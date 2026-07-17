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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.GenerationLoop;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/** In-process Models generation client with production timing measurements. */
public final class PureJavaGenerationClient implements GenerationClient {
  private final TimingBackend backend;
  private final GenerationLoop generationLoop;
  private final double loadMillis;

  PureJavaGenerationClient(InferenceBackend backend, double loadMillis) {
    this.backend = new TimingBackend(Objects.requireNonNull(backend, "backend"));
    this.generationLoop = new GenerationLoop(this.backend);
    this.loadMillis = loadMillis;
  }

  public static PureJavaGenerationClient load(Path model, int contextLength) {
    Objects.requireNonNull(model, "model");
    if (contextLength < 1) {
      throw new IllegalArgumentException("contextLength must be positive");
    }
    System.setProperty("models.purejava.maxContextLength", Integer.toString(contextLength));
    long start = System.nanoTime();
    PureJavaBackend backend = PureJavaBackend.load(model);
    return new PureJavaGenerationClient(backend, elapsedMillis(start));
  }

  @Override
  public String backend() {
    return "pure-java";
  }

  @Override
  public String model() {
    return backend.metadata().modelName();
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
      throw new IllegalStateException("pure-Java generation failed", failure.get());
    }
    if (firstTokenNanos[0] == 0 || outputTokens[0] == 0) {
      throw new IllegalStateException("pure-Java generation produced no output token");
    }
    Duration cpuAfter = ProcessResourceProbe.cpuDuration(pid);
    return new GenerationResult(
        output.toString(),
        inputTokens,
        outputTokens[0],
        nanosToMillis(firstTokenNanos[0] - start),
        nanosToMillis(end - start),
        tokenRate(inputTokens, backend.prefillNanos()),
        loadMillis,
        ProcessResourceProbe.highWaterBytes(pid),
        nanosToMillis(cpuAfter.minus(cpuBefore).toNanos()));
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

  private static final class TimingBackend implements InferenceBackend {
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
}
