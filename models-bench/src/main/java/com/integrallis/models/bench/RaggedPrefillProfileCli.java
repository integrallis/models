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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.LongSupplier;

/** Compares independent and ragged prompt ingestion while preserving per-session KV state. */
final class RaggedPrefillProfileCli {

  private static final int SCHEMA_VERSION = 1;
  private static final Set<String> OPTIONS =
      Set.of(
          "model",
          "prompt",
          "prompt-file",
          "context",
          "concurrency",
          "warmups",
          "iterations",
          "mode",
          "output");
  private static final String DEFAULT_PROMPT =
      "Explain why independent inference requests must preserve separate KV state.";

  private RaggedPrefillProfileCli() {}

  enum Mode {
    SEQUENTIAL,
    RAGGED;

    static Mode parse(String value) {
      try {
        return valueOf(value.trim().toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException failure) {
        throw new IllegalArgumentException("--mode must be sequential or ragged: " + value);
      }
    }

    String externalName() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  static void run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY,
        Integer.toString(configuration.contextLength()));
    try (PureJavaBackend backend = configuration.model().load()) {
      Result result = profile(backend, configuration, System::nanoTime);
      write(configuration.output(), result);
      System.out.printf(
          Locale.ROOT,
          "ragged prefill: mode=%s concurrency=%d prompt-tokens=%d aggregate=%.2f tok/s "
              + "hash=%s%nreport: %s%n",
          result.mode(),
          result.concurrency(),
          result.promptTokens(),
          result.promptTokensPerSecond(),
          result.outputSha256(),
          configuration.output().toAbsolutePath());
    } finally {
      restoreProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
    }
  }

  static Configuration parse(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    PureJavaModelSource model = PureJavaModelSource.resolve(values.get("model"));
    String prompt =
        values.containsKey("prompt-file")
            ? Files.readString(Path.of(values.get("prompt-file")))
            : values.getOrDefault("prompt", DEFAULT_PROMPT);
    Mode mode = Mode.parse(values.getOrDefault("mode", "ragged"));
    return new Configuration(
        model,
        prompt,
        BenchmarkCliArguments.integer(values, "context", 2_048),
        BenchmarkCliArguments.integer(values, "concurrency", 4),
        BenchmarkCliArguments.integer(values, "warmups", 1),
        BenchmarkCliArguments.integer(values, "iterations", 3),
        mode,
        Path.of(
            values.getOrDefault(
                "output",
                "build/reports/inference/ragged-prefill-" + mode.externalName() + ".json")));
  }

  static Result profile(
      BatchInferenceBackend backend, Configuration configuration, LongSupplier nanoTime)
      throws IOException {
    Objects.requireNonNull(backend, "backend");
    Objects.requireNonNull(configuration, "configuration");
    Objects.requireNonNull(nanoTime, "nanoTime");
    if (configuration.concurrency() > backend.maxBatchSize()) {
      throw new IllegalArgumentException(
          "concurrency "
              + configuration.concurrency()
              + " exceeds backend batch capacity "
              + backend.maxBatchSize());
    }
    if (configuration.mode() == Mode.RAGGED && !backend.supportsRaggedPrefillBatch()) {
      throw new IllegalArgumentException(
          "backend does not implement physical ragged prefill batching: " + backend.name());
    }

    for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
      runRound(backend, configuration, nanoTime, false);
    }

    long processId = ProcessHandle.current().pid();
    JvmMemorySnapshot jvmBefore = JvmMemorySnapshot.capture();
    ProcessMetrics.Snapshot processBefore = ProcessMetrics.capture(processId);
    ProfileSupport.GcMetrics gcBefore = ProfileSupport.gcMetrics();
    int promptTokens = 0;
    long elapsedNanos = 0;
    List<String> traces = new ArrayList<>();
    for (int iteration = 0; iteration < configuration.iterations(); iteration++) {
      RoundResult round = runRound(backend, configuration, nanoTime, true);
      promptTokens = Math.addExact(promptTokens, round.promptTokens());
      elapsedNanos = Math.addExact(elapsedNanos, round.elapsedNanos());
      traces.add(round.trace());
    }
    ProcessMetrics.Snapshot processAfter = ProcessMetrics.capture(processId);
    ProfileSupport.GcMetrics gcAfter = ProfileSupport.gcMetrics();
    JvmMemorySnapshot jvmAfter = JvmMemorySnapshot.capture();
    if (elapsedNanos <= 0) {
      throw new IllegalStateException("measured elapsed time must be positive: " + elapsedNanos);
    }

    return new Result(
        SCHEMA_VERSION,
        Instant.now().toString(),
        configuration.mode().externalName(),
        backend.name(),
        configuration.model().identity(),
        Hashing.sha256(configuration.model().artifact()),
        Files.size(configuration.model().artifact()),
        configuration.contextLength(),
        configuration.concurrency(),
        backend.maxBatchSize(),
        configuration.warmups(),
        configuration.iterations(),
        promptTokens,
        elapsedNanos,
        promptTokens * 1_000_000_000.0 / elapsedNanos,
        Hashing.sha256(String.join("\n", traces)),
        Math.max(0, gcAfter.collections() - gcBefore.collections()),
        Math.max(0, gcAfter.pauseMillis() - gcBefore.pauseMillis()),
        processAfter.cpuMillisSince(processBefore),
        processAfter.highWaterBytes(),
        processAfter.residentBytes(),
        processAfter.anonymousResidentBytes(),
        processAfter.fileResidentBytes(),
        processAfter.sharedMemoryResidentBytes(),
        jvmBefore,
        jvmAfter,
        BenchmarkEnvironment.capture(),
        backend.diagnostics());
  }

  private static RoundResult runRound(
      BatchInferenceBackend backend,
      Configuration configuration,
      LongSupplier nanoTime,
      boolean retainTrace) {
    InferenceSession[] sessions = new InferenceSession[configuration.concurrency()];
    int opened = 0;
    try {
      int[][] prompts = new int[sessions.length][];
      int promptTokens = 0;
      for (; opened < sessions.length; opened++) {
        sessions[opened] = backend.openSession();
        prompts[opened] =
            backend
                .tokenizer()
                .encode(
                    configuration.prompt()
                        + "\nRequest "
                        + opened
                        + ":"
                        + " context".repeat(opened));
        if (prompts[opened].length == 0) {
          throw new IllegalArgumentException("session prompt produced no tokens: " + opened);
        }
        if (prompts[opened].length + 1 > configuration.contextLength()) {
          throw new IllegalArgumentException(
              "context length is too small for session " + opened + ": " + prompts[opened].length);
        }
        promptTokens = Math.addExact(promptTokens, prompts[opened].length);
      }

      long started = nanoTime.getAsLong();
      LogitBatch finalLogits =
          configuration.mode() == Mode.RAGGED
              ? backend.prefillBatchTransient(sessions, prompts)
              : prefillSequentially(backend, sessions, prompts);
      long elapsedNanos = nanoTime.getAsLong() - started;
      if (elapsedNanos <= 0) {
        throw new IllegalStateException("round elapsed time must be positive: " + elapsedNanos);
      }

      int[] finalTokens = new int[sessions.length];
      for (int session = 0; session < sessions.length; session++) {
        finalTokens[session] = finalLogits.argmax(session);
      }
      LogitBatch continuation = backend.forwardBatchTransient(sessions, finalTokens);
      StringBuilder trace = new StringBuilder();
      if (retainTrace) {
        for (int session = 0; session < sessions.length; session++) {
          trace
              .append(session)
              .append(':')
              .append(finalTokens[session])
              .append(':')
              .append(continuation.argmax(session))
              .append('\n');
        }
      }
      return new RoundResult(promptTokens, elapsedNanos, trace.toString());
    } finally {
      for (int index = opened - 1; index >= 0; index--) {
        sessions[index].close();
      }
    }
  }

  private static LogitBatch prefillSequentially(
      BatchInferenceBackend backend, InferenceSession[] sessions, int[][] prompts) {
    int vocabularySize = backend.metadata().vocabSize();
    float[] logits = new float[Math.multiplyExact(sessions.length, vocabularySize)];
    for (int session = 0; session < sessions.length; session++) {
      float[] row =
          backend.prefill(sessions[session], prompts[session], sessions[session].checkpoint());
      System.arraycopy(row, 0, logits, session * vocabularySize, vocabularySize);
    }
    return new LogitBatch(sessions.length, vocabularySize, logits);
  }

  private static void write(Path output, Result result) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    new ObjectMapper()
        .enable(SerializationFeature.INDENT_OUTPUT)
        .writeValue(output.toFile(), result);
  }

  private static void restoreProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }

  record Configuration(
      PureJavaModelSource model,
      String prompt,
      int contextLength,
      int concurrency,
      int warmups,
      int iterations,
      Mode mode,
      Path output) {

    Configuration {
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(prompt, "prompt");
      Objects.requireNonNull(mode, "mode");
      Objects.requireNonNull(output, "output");
      if (prompt.isBlank()) {
        throw new IllegalArgumentException("prompt must not be blank");
      }
      if (contextLength <= 0 || concurrency <= 0 || warmups < 0 || iterations <= 0) {
        throw new IllegalArgumentException("ragged prefill profile options are invalid");
      }
    }
  }

  record Result(
      int schemaVersion,
      String timestamp,
      String mode,
      String backend,
      String model,
      String artifactSha256,
      long artifactSizeBytes,
      int contextLength,
      int concurrency,
      int maxBatchSize,
      int warmups,
      int iterations,
      int promptTokens,
      long elapsedNanos,
      double promptTokensPerSecond,
      String outputSha256,
      long gcCollections,
      long gcPauseMillis,
      double cpuMillis,
      long peakRssBytes,
      long currentRssBytes,
      long anonymousRssBytes,
      long fileRssBytes,
      long sharedMemoryRssBytes,
      JvmMemorySnapshot jvmMemoryBefore,
      JvmMemorySnapshot jvmMemoryAfter,
      BenchmarkEnvironment environment,
      BackendDiagnostics diagnostics) {}

  private record RoundResult(int promptTokens, long elapsedNanos, String trace) {}
}
