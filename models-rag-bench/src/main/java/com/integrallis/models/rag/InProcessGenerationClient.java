/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.models.rag;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.PromptCacheMetrics;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import org.modeljars.ModelJarDescriptor;

/** In-process Models generation client with production timing measurements. */
public final class InProcessGenerationClient implements GenerationClient {
  private final String backendName;
  private final TimingBackend backend;
  private final GenerationLoop generationLoop;
  private final double loadMillis;

  InProcessGenerationClient(String backendName, InferenceBackend backend, double loadMillis) {
    if (backendName == null || backendName.isBlank()) {
      throw new IllegalArgumentException("backendName must not be blank");
    }
    this.backendName = backendName;
    this.backend = instrument(Objects.requireNonNull(backend, "backend"));
    this.generationLoop = new GenerationLoop(this.backend);
    this.loadMillis = loadMillis;
  }

  public static InProcessGenerationClient loadPureJava(Path model, int contextLength) {
    Objects.requireNonNull(model, "model");
    return load("pure-java", contextLength, () -> PureJavaBackend.load(model));
  }

  public static InProcessGenerationClient loadPureJava(
      ModelJarDescriptor descriptor, int contextLength) {
    Objects.requireNonNull(descriptor, "descriptor");
    return load("pure-java", contextLength, () -> PureJavaBackend.load(descriptor));
  }

  public static InProcessGenerationClient loadRustFfm(Path model, int contextLength) {
    Objects.requireNonNull(model, "model");
    return load("rust-ffm", contextLength, () -> RustFfmBackend.load(model));
  }

  public static InProcessGenerationClient loadRustFfm(
      ModelJarDescriptor descriptor, int contextLength) {
    Objects.requireNonNull(descriptor, "descriptor");
    return load("rust-ffm", contextLength, () -> RustFfmBackend.load(descriptor));
  }

  private static InProcessGenerationClient load(
      String backendName,
      int contextLength,
      java.util.function.Supplier<? extends InferenceBackend> backendLoader) {
    if (contextLength < 1) {
      throw new IllegalArgumentException("contextLength must be positive");
    }
    System.setProperty("models.purejava.maxContextLength", Integer.toString(contextLength));
    long start = System.nanoTime();
    InferenceBackend backend = backendLoader.get();
    return new InProcessGenerationClient(backendName, backend, elapsedMillis(start));
  }

  @Override
  public String backend() {
    return backendName;
  }

  @Override
  public String model() {
    return backend.metadata().modelName();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return backend.diagnostics();
  }

  @Override
  public Map<String, String> generationControls() {
    return Map.of(
        "temperature", "0",
        "topK", "1",
        "topP", "1",
        "seed", "42",
        "repetitionPenalty", "1",
        "promptCache", "longest-common-prefix");
  }

  @Override
  public GenerationResult generate(String prompt, int maxTokens) {
    int inputTokens = backend.tokenizer().encode(prompt).length;
    backend.begin();
    SamplingOptions options =
        SamplingOptions.builder()
            .temperature(0)
            .topP(1)
            .topK(1)
            .seed(42)
            .repetitionPenalty(1)
            .maxTokens(maxTokens)
            .build();
    StringBuilder output = new StringBuilder();
    long[] firstTokenNanos = {0};
    int[] outputTokens = {0};
    AtomicReference<Throwable> failure = new AtomicReference<>();
    long pid = ProcessHandle.current().pid();
    Duration cpuBefore = ProcessResourceProbe.cpuDuration(pid);
    long start = System.nanoTime();
    generationLoop.generate(
        prompt,
        options,
        new TokenStream() {
          @Override
          public void onToken(String token) {
            if (firstTokenNanos[0] == 0) {
              firstTokenNanos[0] = System.nanoTime();
            }
            outputTokens[0]++;
            output.append(token);
          }

          @Override
          public void onComplete() {}

          @Override
          public void onError(Throwable error) {
            failure.set(error);
          }
        });
    long end = System.nanoTime();
    if (failure.get() != null) {
      throw new IllegalStateException(backendName + " generation failed", failure.get());
    }
    if (firstTokenNanos[0] == 0 || outputTokens[0] == 0) {
      throw new IllegalStateException(backendName + " generation produced no output token");
    }
    PromptCacheMetrics promptCache = generationLoop.lastPromptCacheMetrics();
    int evaluatedInputTokens =
        promptCache.supported() ? promptCache.cacheWriteInputTokens() : inputTokens;
    Duration cpuAfter = ProcessResourceProbe.cpuDuration(pid);
    return new GenerationResult(
        output.toString(),
        inputTokens,
        promptCache.cacheReadInputTokens(),
        promptCache.cacheWriteInputTokens(),
        outputTokens[0],
        nanosToMillis(firstTokenNanos[0] - start),
        nanosToMillis(end - start),
        tokenRate(evaluatedInputTokens, backend.prefillNanos()),
        loadMillis,
        ProcessResourceProbe.highWaterBytes(pid),
        nanosToMillis(cpuAfter.minus(cpuBefore).toNanos()),
        null);
  }

  @Override
  public void close() {
    backend.close();
  }

  private static double tokenRate(int tokens, long nanos) {
    return nanos > 0 ? tokens * 1_000_000_000.0 / nanos : 0;
  }

  private static double elapsedMillis(long start) {
    return nanosToMillis(System.nanoTime() - start);
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }

  private static TimingBackend instrument(InferenceBackend backend) {
    if (backend instanceof SpeculativeInferenceBackend speculative) {
      return new TimingSpeculativeBackend(speculative);
    }
    if (backend instanceof RewindableInferenceBackend rewindable) {
      return new TimingRewindableBackend(rewindable);
    }
    return new TimingBackend(backend);
  }

  private static class TimingBackend implements InferenceBackend {
    private final InferenceBackend delegate;
    private long prefillNanos;

    private TimingBackend(InferenceBackend delegate) {
      this.delegate = delegate;
    }

    private void begin() {
      prefillNanos = 0;
    }

    private long prefillNanos() {
      return prefillNanos;
    }

    @Override
    public String name() {
      return delegate.name();
    }

    @Override
    public ModelMetadata metadata() {
      return delegate.metadata();
    }

    @Override
    public Tokenizer tokenizer() {
      return delegate.tokenizer();
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return delegate.diagnostics();
    }

    @Override
    public float[] forward(int token, int position) {
      return delegate.forward(token, position);
    }

    @Override
    public float[] forwardTransient(int token, int position) {
      return delegate.forwardTransient(token, position);
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      long start = System.nanoTime();
      float[] logits = delegate.prefill(tokens, startPosition);
      prefillNanos += System.nanoTime() - start;
      return logits;
    }

    @Override
    public void reset() {
      delegate.reset();
    }

    @Override
    public void close() {
      delegate.close();
    }
  }

  private static class TimingRewindableBackend extends TimingBackend
      implements RewindableInferenceBackend {
    private final RewindableInferenceBackend delegate;

    private TimingRewindableBackend(RewindableInferenceBackend delegate) {
      super(delegate);
      this.delegate = delegate;
    }

    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }

    @Override
    public void rewind(int checkpoint) {
      delegate.rewind(checkpoint);
    }
  }

  private static final class TimingSpeculativeBackend extends TimingRewindableBackend
      implements SpeculativeInferenceBackend {
    private final SpeculativeInferenceBackend delegate;

    private TimingSpeculativeBackend(SpeculativeInferenceBackend delegate) {
      super(delegate);
      this.delegate = delegate;
    }

    @Override
    public LogitBatch verify(int[] tokens, int startPosition) {
      return delegate.verify(tokens, startPosition);
    }

    @Override
    public LogitBatch verifyTransient(int[] tokens, int startPosition) {
      return delegate.verifyTransient(tokens, startPosition);
    }
  }
}
