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
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

/** Measures weight-reusing decode across independent in-process inference sessions. */
final class SessionBatchProfileCli {

  private static final int SCHEMA_VERSION = 3;
  private static final Set<String> OPTIONS =
      Set.of(
          "model",
          "prompt",
          "prompt-file",
          "context",
          "concurrency",
          "warmup-steps",
          "measure-steps",
          "mode",
          "output");
  private static final String DEFAULT_PROMPT =
      "Explain why continuous batching improves aggregate inference throughput.";

  private SessionBatchProfileCli() {}

  enum Mode {
    SEQUENTIAL,
    BATCHED;

    static Mode parse(String value) {
      try {
        return valueOf(value.trim().toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException exception) {
        throw new IllegalArgumentException("--mode must be sequential or batched: " + value);
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
          "session batch: mode=%s concurrency=%d tokens=%d aggregate=%.2f tok/s "
              + "per-request-tpot=%.3f ms hash=%s%nreport: %s%n",
          result.mode(),
          result.concurrency(),
          result.generatedTokens(),
          result.aggregateTokensPerSecond(),
          result.perRequestTokenMillis(),
          result.outputTokenSha256(),
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
    return new Configuration(
        model,
        prompt,
        BenchmarkCliArguments.integer(values, "context", 2_048),
        BenchmarkCliArguments.integer(values, "concurrency", 4),
        BenchmarkCliArguments.integer(values, "warmup-steps", 16),
        BenchmarkCliArguments.integer(values, "measure-steps", 64),
        Mode.parse(values.getOrDefault("mode", "batched")),
        Path.of(
            values.getOrDefault("output", "build/reports/inference/session-batch-profile.json")));
  }

  static Result profile(
      BatchInferenceBackend backend, Configuration configuration, LongSupplier nanoTime)
      throws IOException {
    return profile(backend, configuration, nanoTime, JvmMemorySnapshot::capture);
  }

  static Result profile(
      BatchInferenceBackend backend,
      Configuration configuration,
      LongSupplier nanoTime,
      Supplier<JvmMemorySnapshot> memorySnapshot)
      throws IOException {
    Objects.requireNonNull(backend, "backend");
    Objects.requireNonNull(configuration, "configuration");
    Objects.requireNonNull(nanoTime, "nanoTime");
    Objects.requireNonNull(memorySnapshot, "memorySnapshot");
    int maxBatchSize = backend.maxBatchSize();
    if (configuration.concurrency() > maxBatchSize) {
      throw new IllegalArgumentException(
          "concurrency "
              + configuration.concurrency()
              + " exceeds backend batch capacity "
              + maxBatchSize);
    }

    runPhase(
        backend,
        configuration,
        configuration.warmupSteps(),
        false,
        MeasurementBoundary::unmeasured);

    long processId = ProcessHandle.current().pid();
    JvmMemorySnapshot jvmMemoryBefore = memorySnapshot.get();
    ProcessMetrics.Snapshot processBefore = ProcessMetrics.capture(processId);
    ProfileSupport.GcMetrics gcBefore = ProfileSupport.gcMetrics();
    long start = nanoTime.getAsLong();
    PhaseResult measured =
        runPhase(
            backend,
            configuration,
            configuration.measuredSteps(),
            true,
            () -> MeasurementBoundary.capture(nanoTime, processId, memorySnapshot));
    MeasurementBoundary boundary = measured.boundary();
    long elapsedNanos = boundary.completedNanos() - start;
    if (elapsedNanos <= 0) {
      throw new IllegalStateException("measured elapsed time must be positive: " + elapsedNanos);
    }

    int generatedTokens =
        Math.multiplyExact(configuration.concurrency(), configuration.measuredSteps());
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
        maxBatchSize,
        configuration.warmupSteps(),
        configuration.measuredSteps(),
        generatedTokens,
        elapsedNanos,
        generatedTokens * 1_000_000_000.0 / elapsedNanos,
        elapsedNanos / (configuration.measuredSteps() * 1_000_000.0),
        outputHash(measured.outputTokens()),
        Math.max(0, boundary.gc().collections() - gcBefore.collections()),
        Math.max(0, boundary.gc().pauseMillis() - gcBefore.pauseMillis()),
        boundary.process().cpuMillisSince(processBefore),
        boundary.process().highWaterBytes(),
        boundary.process().residentBytes(),
        boundary.process().anonymousResidentBytes(),
        boundary.process().fileResidentBytes(),
        boundary.process().sharedMemoryResidentBytes(),
        jvmMemoryBefore,
        boundary.jvmMemory(),
        BenchmarkEnvironment.capture(),
        backend.diagnostics());
  }

  private static PhaseResult runPhase(
      BatchInferenceBackend backend,
      Configuration configuration,
      int steps,
      boolean retainTokens,
      Supplier<MeasurementBoundary> boundary) {
    InferenceSession[] sessions = new InferenceSession[configuration.concurrency()];
    int opened = 0;
    try {
      for (; opened < sessions.length; opened++) {
        sessions[opened] = backend.openSession();
      }
      int[] tokens = prefillSessions(backend, configuration, sessions, steps);
      int[][] outputTokens = retainTokens ? new int[sessions.length][steps] : new int[0][];
      for (int step = 0; step < steps; step++) {
        if (configuration.mode() == Mode.BATCHED) {
          LogitBatch logits = backend.forwardBatchTransient(sessions, tokens);
          for (int session = 0; session < sessions.length; session++) {
            tokens[session] = logits.argmax(session);
            if (retainTokens) {
              outputTokens[session][step] = tokens[session];
            }
          }
        } else {
          for (int session = 0; session < sessions.length; session++) {
            float[] logits =
                backend.forwardTransient(
                    sessions[session], tokens[session], sessions[session].checkpoint());
            tokens[session] = argmax(logits);
            if (retainTokens) {
              outputTokens[session][step] = tokens[session];
            }
          }
        }
      }
      return new PhaseResult(outputTokens, boundary.get());
    } finally {
      for (int index = opened - 1; index >= 0; index--) {
        sessions[index].close();
      }
    }
  }

  private static int[] prefillSessions(
      BatchInferenceBackend backend,
      Configuration configuration,
      InferenceSession[] sessions,
      int steps) {
    Tokenizer tokenizer = backend.tokenizer();
    int[] tokens = new int[sessions.length];
    for (int session = 0; session < sessions.length; session++) {
      String prompt = configuration.prompt() + "\nRequest " + session + '.';
      int[] promptTokens = tokenizer.encode(prompt);
      if (promptTokens.length == 0) {
        throw new IllegalArgumentException("session prompt produced no tokens: " + session);
      }
      int requiredContext = Math.addExact(promptTokens.length, steps);
      if (requiredContext > configuration.contextLength()) {
        throw new IllegalArgumentException(
            "context length "
                + configuration.contextLength()
                + " is smaller than required token count "
                + requiredContext
                + " for session "
                + session);
      }
      tokens[session] = argmax(backend.prefill(sessions[session], promptTokens, 0));
    }
    return tokens;
  }

  private static int argmax(float[] values) {
    if (values.length == 0) {
      throw new IllegalArgumentException("logits must not be empty");
    }
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static String outputHash(int[][] tokens) {
    StringBuilder canonical = new StringBuilder();
    for (int session = 0; session < tokens.length; session++) {
      canonical.append(session).append(':').append(Arrays.toString(tokens[session])).append('\n');
    }
    return Hashing.sha256(canonical.toString());
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
      int warmupSteps,
      int measuredSteps,
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
      if (contextLength <= 0 || concurrency <= 0 || warmupSteps < 0 || measuredSteps <= 0) {
        throw new IllegalArgumentException("context, concurrency, and step counts are invalid");
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
      int warmupSteps,
      int measuredSteps,
      int generatedTokens,
      long elapsedNanos,
      double aggregateTokensPerSecond,
      double perRequestTokenMillis,
      String outputTokenSha256,
      long gcCollections,
      long gcPauseMillis,
      double cpuMillis,
      long peakRssBytes,
      long currentRssBytes,
      long anonymousRssBytes,
      long fileRssBytes,
      long sharedMemoryRssBytes,
      JvmMemorySnapshot jvmMemoryBefore,
      JvmMemorySnapshot jvmMemoryActive,
      BenchmarkEnvironment environment,
      BackendDiagnostics diagnostics) {}

  private record PhaseResult(int[][] outputTokens, MeasurementBoundary boundary) {}

  private record MeasurementBoundary(
      long completedNanos,
      ProcessMetrics.Snapshot process,
      ProfileSupport.GcMetrics gc,
      JvmMemorySnapshot jvmMemory) {

    private static MeasurementBoundary capture(
        LongSupplier nanoTime, long processId, Supplier<JvmMemorySnapshot> memorySnapshot) {
      long completedNanos = nanoTime.getAsLong();
      ProcessMetrics.Snapshot process = ProcessMetrics.capture(processId);
      ProfileSupport.GcMetrics gc = ProfileSupport.gcMetrics();
      return new MeasurementBoundary(completedNanos, process, gc, memorySnapshot.get());
    }

    private static MeasurementBoundary unmeasured() {
      return new MeasurementBoundary(
          0, ProcessMetrics.capture(-1), new ProfileSupport.GcMetrics(0, 0), null);
    }
  }
}
