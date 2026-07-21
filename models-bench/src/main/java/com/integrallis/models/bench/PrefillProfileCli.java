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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.bench.ProfileSupport.GcMetrics;
import com.integrallis.models.bench.ProfileSupport.GcMetricsSource;
import com.integrallis.models.bench.ProfileSupport.ProfileRecording;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.modeljars.ModelJarRegistry;

/** Captures warmed-up prompt prefill without model loading, reset, or decode work. */
final class PrefillProfileCli {

  private static final Set<String> OPTIONS =
      Set.of("model", "modeljar", "prompt", "prompt-file", "context", "warmups", "output");
  private static final String DEFAULT_PROMPT =
      "Explain why profiling prompt prefill separately from autoregressive decode matters.";

  private PrefillProfileCli() {}

  static void run(String[] args) throws Exception {
    Configuration configuration = parse(args);
    System.setProperty(
        "models.purejava.maxContextLength", Integer.toString(configuration.contextLength()));
    try (InferenceBackend backend = configuration.model().load();
        ProfileRecording recording = ProfileSupport.jfr("models-pure-java-prefill")) {
      Result result = profile(backend, configuration, recording);
      System.out.printf(
          "prefill profile: prompt=%d warmups=%d prefill=%.2f tok/s "
              + "checksum=%.9g gc=%d gcPause=%d ms%nrecording: %s%n",
          result.promptTokens(),
          result.warmups(),
          result.tokensPerSecond(),
          result.logitChecksum(),
          result.gcCollections(),
          result.gcPauseMillis(),
          configuration.output().toAbsolutePath());
    }
  }

  static Configuration parse(String[] args) throws IOException {
    return parse(args, ModelJarRegistry.fromClasspath());
  }

  static Configuration parse(String[] args, ModelJarRegistry modelJarRegistry) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    PureJavaModelSource model =
        PureJavaModelSource.resolve(values.get("model"), values.get("modeljar"), modelJarRegistry);
    String prompt =
        values.containsKey("prompt-file")
            ? Files.readString(Path.of(values.get("prompt-file")))
            : values.getOrDefault("prompt", DEFAULT_PROMPT);
    return new Configuration(
        model,
        prompt,
        BenchmarkCliArguments.integer(values, "context", 2_048),
        BenchmarkCliArguments.integer(values, "warmups", 2),
        Path.of(values.getOrDefault("output", "build/reports/inference/prefill-profile.jfr")));
  }

  static Result profile(
      InferenceBackend backend, Configuration configuration, ProfileRecording recording)
      throws IOException {
    return profile(backend, configuration, recording, ProfileSupport::gcMetrics);
  }

  static Result profile(
      InferenceBackend backend,
      Configuration configuration,
      ProfileRecording recording,
      GcMetricsSource gcMetricsSource)
      throws IOException {
    Objects.requireNonNull(backend, "backend");
    Objects.requireNonNull(configuration, "configuration");
    Objects.requireNonNull(recording, "recording");
    Objects.requireNonNull(gcMetricsSource, "gcMetricsSource");

    Tokenizer tokenizer = backend.tokenizer();
    int[] promptTokens = tokenizer.encode(configuration.prompt());
    if (promptTokens.length == 0) {
      throw new IllegalArgumentException("prompt produced no tokens");
    }
    if (promptTokens.length > configuration.contextLength()) {
      throw new IllegalArgumentException(
          "context length "
              + configuration.contextLength()
              + " is smaller than prompt token count "
              + promptTokens.length);
    }

    for (int index = 0; index < configuration.warmups(); index++) {
      backend.reset();
      backend.prefill(promptTokens, 0);
    }
    backend.reset();

    recording.start();
    GcMetrics gcBefore = gcMetricsSource.snapshot();
    long start = System.nanoTime();
    float[] logits;
    long elapsedNanos;
    GcMetrics gcAfter;
    try {
      logits = backend.prefill(promptTokens, 0);
      elapsedNanos = System.nanoTime() - start;
      gcAfter = gcMetricsSource.snapshot();
    } finally {
      recording.stop();
    }
    recording.dump(configuration.output());
    return new Result(
        promptTokens.length,
        configuration.warmups(),
        elapsedNanos,
        checksum(logits),
        Math.max(0, gcAfter.collections() - gcBefore.collections()),
        Math.max(0, gcAfter.pauseMillis() - gcBefore.pauseMillis()));
  }

  private static double checksum(float[] logits) {
    double checksum = 0.0;
    for (float logit : logits) {
      checksum += logit;
    }
    return checksum;
  }

  record Configuration(
      PureJavaModelSource model, String prompt, int contextLength, int warmups, Path output) {

    Configuration {
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(prompt, "prompt");
      Objects.requireNonNull(output, "output");
      if (prompt.isBlank()) {
        throw new IllegalArgumentException("prompt must not be blank");
      }
      if (contextLength <= 0 || warmups < 0) {
        throw new IllegalArgumentException("context and warmup count are invalid");
      }
    }
  }

  record Result(
      int promptTokens,
      int warmups,
      long elapsedNanos,
      double logitChecksum,
      long gcCollections,
      long gcPauseMillis) {

    double tokensPerSecond() {
      return promptTokens * 1_000_000_000.0 / elapsedNanos;
    }
  }
}
