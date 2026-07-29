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

/** Captures warmed-up autoregressive decode at production sequence positions. */
final class DecodeProfileCli {

  private static final Set<String> OPTIONS =
      Set.of(
          "model",
          "prompt",
          "prompt-file",
          "context",
          "token-id",
          "warmup-tokens",
          "measure-tokens",
          "output");
  private static final String DEFAULT_PROMPT =
      "Explain why profiling autoregressive decode separately from prompt prefill matters.";

  private DecodeProfileCli() {}

  static void run(String[] args) throws Exception {
    Configuration configuration = parse(args);
    System.setProperty(
        "models.purejava.maxContextLength", Integer.toString(configuration.contextLength()));
    try (InferenceBackend backend = configuration.model().load();
        ProfileRecording recording = ProfileSupport.jfr("models-pure-java-decode")) {
      Result result = profile(backend, configuration, recording);
      System.out.printf(
          "decode profile: prompt=%d warmup=%d measured=%d decode=%.2f tok/s "
              + "checksum=%.9g gc=%d gcPause=%d ms%nrecording: %s%n",
          result.promptTokens(),
          result.warmupTokens(),
          result.measuredTokens(),
          result.tokensPerSecond(),
          result.logitChecksum(),
          result.gcCollections(),
          result.gcPauseMillis(),
          configuration.output().toAbsolutePath());
    }
  }

  static Configuration parse(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    PureJavaModelSource model = PureJavaModelSource.resolve(values.get("model"));
    String prompt =
        values.containsKey("prompt-file")
            ? Files.readString(Path.of(values.get("prompt-file")))
            : values.getOrDefault("prompt", DEFAULT_PROMPT);
    return new Configuration(
        model,
        prompt,
        BenchmarkCliArguments.integer(values, "context", 2_048),
        BenchmarkCliArguments.integer(values, "token-id", -1),
        BenchmarkCliArguments.integer(values, "warmup-tokens", 64),
        BenchmarkCliArguments.integer(values, "measure-tokens", 256),
        Path.of(values.getOrDefault("output", "build/reports/inference/decode-profile.jfr")));
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

    backend.reset();
    Tokenizer tokenizer = backend.tokenizer();
    int[] promptTokens = tokenizer.encode(configuration.prompt());
    if (promptTokens.length == 0) {
      throw new IllegalArgumentException("prompt produced no tokens");
    }
    int requiredContext =
        Math.addExact(
            promptTokens.length,
            Math.max(configuration.warmupTokens(), configuration.measuredTokens()));
    if (requiredContext > configuration.contextLength()) {
      throw new IllegalArgumentException(
          "context length "
              + configuration.contextLength()
              + " is smaller than required token count "
              + requiredContext);
    }

    int token =
        configuration.tokenId() >= 0
            ? configuration.tokenId()
            : promptTokens[promptTokens.length - 1];
    if (token >= tokenizer.vocabSize()) {
      throw new IllegalArgumentException(
          "token-id " + token + " is outside vocabulary size " + tokenizer.vocabSize());
    }

    backend.prefill(promptTokens, 0);
    int position = promptTokens.length;
    for (int index = 0; index < configuration.warmupTokens(); index++) {
      backend.forwardTransient(token, position++);
    }

    backend.reset();
    backend.prefill(promptTokens, 0);
    position = promptTokens.length;

    recording.start();
    GcMetrics gcBefore = gcMetricsSource.snapshot();
    long start = System.nanoTime();
    long elapsedNanos;
    double checksum = 0.0;
    GcMetrics gcAfter;
    try {
      for (int index = 0; index < configuration.measuredTokens(); index++) {
        float[] logits = backend.forwardTransient(token, position++);
        checksum += logits[index % logits.length];
      }
      elapsedNanos = System.nanoTime() - start;
      gcAfter = gcMetricsSource.snapshot();
    } finally {
      recording.stop();
    }
    recording.dump(configuration.output());
    return new Result(
        promptTokens.length,
        configuration.warmupTokens(),
        configuration.measuredTokens(),
        elapsedNanos,
        checksum,
        Math.max(0, gcAfter.collections() - gcBefore.collections()),
        Math.max(0, gcAfter.pauseMillis() - gcBefore.pauseMillis()));
  }

  record Configuration(
      PureJavaModelSource model,
      String prompt,
      int contextLength,
      int tokenId,
      int warmupTokens,
      int measuredTokens,
      Path output) {

    Configuration {
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(prompt, "prompt");
      Objects.requireNonNull(output, "output");
      if (prompt.isBlank()) {
        throw new IllegalArgumentException("prompt must not be blank");
      }
      if (contextLength <= 0 || tokenId < -1 || warmupTokens < 0 || measuredTokens <= 0) {
        throw new IllegalArgumentException("context and token counts are invalid");
      }
    }
  }

  record Result(
      int promptTokens,
      int warmupTokens,
      int measuredTokens,
      long elapsedNanos,
      double logitChecksum,
      long gcCollections,
      long gcPauseMillis) {

    double tokensPerSecond() {
      return measuredTokens * 1_000_000_000.0 / elapsedNanos;
    }
  }
}
