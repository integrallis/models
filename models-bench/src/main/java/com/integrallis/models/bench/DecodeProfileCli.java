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
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import jdk.jfr.Recording;
import org.modeljars.ModelJarRegistry;

/** Captures warmed-up autoregressive decode at production sequence positions. */
final class DecodeProfileCli {

  private static final Set<String> OPTIONS =
      Set.of(
          "model",
          "modeljar",
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
        ProfileRecording recording = new JfrProfileRecording()) {
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
    return parse(args, ModelJarRegistry.fromClasspath());
  }

  static Configuration parse(String[] args, ModelJarRegistry modelJarRegistry) throws IOException {
    Map<String, String> values = parseOptions(args);
    PureJavaModelSource model =
        PureJavaModelSource.resolve(values.get("model"), values.get("modeljar"), modelJarRegistry);
    String prompt =
        values.containsKey("prompt-file")
            ? Files.readString(Path.of(values.get("prompt-file")))
            : values.getOrDefault("prompt", DEFAULT_PROMPT);
    return new Configuration(
        model,
        prompt,
        integer(values, "context", 2_048),
        integer(values, "token-id", -1),
        integer(values, "warmup-tokens", 64),
        integer(values, "measure-tokens", 256),
        Path.of(values.getOrDefault("output", "build/reports/inference/decode-profile.jfr")));
  }

  static Result profile(
      InferenceBackend backend, Configuration configuration, ProfileRecording recording)
      throws IOException {
    return profile(backend, configuration, recording, DecodeProfileCli::gcMetrics);
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

  private static GcMetrics gcMetrics() {
    long collections = 0;
    long pauseMillis = 0;
    for (var collector : ManagementFactory.getGarbageCollectorMXBeans()) {
      if (collector.getCollectionCount() >= 0) {
        collections += collector.getCollectionCount();
      }
      if (collector.getCollectionTime() >= 0) {
        pauseMillis += collector.getCollectionTime();
      }
    }
    return new GcMetrics(collections, pauseMillis);
  }

  private static Map<String, String> parseOptions(String[] args) {
    if ((args.length & 1) != 0) {
      throw new IllegalArgumentException("options must be provided as --name value pairs");
    }
    Map<String, String> values = new HashMap<>();
    for (int index = 0; index < args.length; index += 2) {
      String option = args[index];
      if (!option.startsWith("--")) {
        throw new IllegalArgumentException("expected option, got: " + option);
      }
      String name = option.substring(2);
      if (!OPTIONS.contains(name)) {
        throw new IllegalArgumentException("unknown option: " + option);
      }
      if (values.put(name, args[index + 1]) != null) {
        throw new IllegalArgumentException("duplicate option: " + option);
      }
    }
    return values;
  }

  private static int integer(Map<String, String> values, String name, int defaultValue) {
    String value = values.get(name);
    if (value == null) {
      return defaultValue;
    }
    try {
      return Integer.parseInt(value);
    } catch (NumberFormatException exception) {
      throw new IllegalArgumentException("--" + name + " must be an integer: " + value, exception);
    }
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

  record GcMetrics(long collections, long pauseMillis) {}

  @FunctionalInterface
  interface GcMetricsSource {
    GcMetrics snapshot();
  }

  interface ProfileRecording extends AutoCloseable {
    void start();

    void stop();

    void dump(Path output) throws IOException;

    @Override
    void close();
  }

  private static final class JfrProfileRecording implements ProfileRecording {
    private final Recording recording;

    private JfrProfileRecording() throws IOException, ParseException {
      recording = new Recording(jdk.jfr.Configuration.getConfiguration("profile"));
      recording.setName("models-pure-java-decode");
    }

    @Override
    public void start() {
      recording.start();
    }

    @Override
    public void stop() {
      recording.stop();
    }

    @Override
    public void dump(Path output) throws IOException {
      Path absolute = output.toAbsolutePath();
      Path parent = absolute.getParent();
      if (parent != null) {
        Files.createDirectories(parent);
      }
      recording.dump(absolute);
    }

    @Override
    public void close() {
      recording.close();
    }
  }
}
