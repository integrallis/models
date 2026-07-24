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
package com.integrallis.models.bench;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.SpeculativeGenerationMetrics;
import com.integrallis.models.runtime.SpeculativeGenerationOptions;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

/** In-process benchmark target for Java and Rust-FFM Models backends. */
final class PureJavaBenchmarkTarget implements BenchmarkTarget {

  private final TimingBackend backend;
  private final GenerationLoop loop;
  private final double loadMillis;
  private final BackendDiagnostics diagnostics;

  private PureJavaBenchmarkTarget(
      TimingBackend backend,
      double loadMillis,
      BackendDiagnostics diagnostics,
      SpeculativeGenerationOptions speculativeOptions) {
    this.backend = backend;
    this.loop = new GenerationLoop(backend, speculativeOptions);
    this.loadMillis = loadMillis;
    this.diagnostics = diagnostics;
  }

  static PureJavaBenchmarkTarget load(
      PureJavaModelSource model,
      int contextLength,
      SpeculativeGenerationOptions speculativeOptions) {
    System.setProperty("models.purejava.maxContextLength", Integer.toString(contextLength));
    long start = System.nanoTime();
    PureJavaBackend loaded = model.load();
    double elapsedMillis = nanosToMillis(System.nanoTime() - start);
    return new PureJavaBenchmarkTarget(
        new TimingBackend(loaded), elapsedMillis, loaded.diagnostics(), speculativeOptions);
  }

  static PureJavaBenchmarkTarget loadRust(
      Path model, int contextLength, SpeculativeGenerationOptions speculativeOptions) {
    System.setProperty("models.purejava.maxContextLength", Integer.toString(contextLength));
    long start = System.nanoTime();
    RustFfmBackend loaded = RustFfmBackend.load(model);
    double elapsedMillis = nanosToMillis(System.nanoTime() - start);
    return new PureJavaBenchmarkTarget(
        new TimingBackend(loaded), elapsedMillis, loaded.diagnostics(), speculativeOptions);
  }

  @Override
  public TrialMeasurement generate(String prompt, int maxTokens) {
    int inputTokens = backend.tokenizer().encode(prompt).length;
    backend.beginTrial(inputTokens);
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
    AtomicReference<Throwable> error = new AtomicReference<>();
    Duration cpuBefore = cpuDuration();
    long start = System.nanoTime();

    try {
      loop.generate(
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
            public void onError(Throwable failure) {
              error.set(failure);
            }
          });
    } catch (RuntimeException failure) {
      error.compareAndSet(null, failure);
    }
    long end = System.nanoTime();
    if (error.get() != null) {
      return TrialMeasurement.failure(error.get().toString());
    }
    if (firstTokenNanos[0] == 0 || outputTokens[0] == 0) {
      return TrialMeasurement.failure("generation completed without an output token");
    }

    SpeculativeGenerationMetrics speculativeMetrics = loop.lastSpeculativeMetrics();
    return TrialMeasurement.success(
        nanosToMillis(firstTokenNanos[0] - start),
        nanosToMillis(end - start),
        rate(inputTokens, backend.prefillNanos()),
        inputTokens,
        outputTokens[0],
        ProcessMemory.highWaterBytes(),
        nanosToMillis(cpuDuration().minus(cpuBefore).toNanos()),
        Hashing.sha256(output.toString()),
        speculativeMetrics);
  }

  @Override
  public double loadMillis() {
    return loadMillis;
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return diagnostics;
  }

  @Override
  public void close() {
    backend.close();
  }

  private static Duration cpuDuration() {
    return ProcessHandle.current().info().totalCpuDuration().orElse(Duration.ZERO);
  }

  private static double rate(int tokens, long nanos) {
    return nanos > 0 ? tokens * 1_000_000_000.0 / nanos : 0;
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }
}
