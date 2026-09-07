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
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ContinuousBatchingMetrics;
import com.integrallis.models.runtime.ContinuousBatchingOptions;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.TextGenerationSession;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/** Profiles the public generation-session API with and without continuous batching. */
final class RuntimeBatchProfileCli {

  private static final int SCHEMA_VERSION = 2;
  private static final Set<String> OPTIONS =
      Set.of(
          "model",
          "prompt",
          "prompt-file",
          "context",
          "concurrency",
          "warmups",
          "iterations",
          "max-tokens",
          "batch-prefill-across-sessions",
          "mode",
          "batch-delay-ms",
          "output");
  private static final String DEFAULT_PROMPT =
      "Explain in one sentence why independent inference requests must keep separate KV state.";

  private RuntimeBatchProfileCli() {}

  enum Mode {
    SERIALIZED,
    CONTINUOUS;

    static Mode parse(String value) {
      try {
        return valueOf(value.trim().toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException failure) {
        throw new IllegalArgumentException("--mode must be serialized or continuous: " + value);
      }
    }

    String externalName() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  static void run(String[] args) throws Exception {
    Configuration configuration = parse(args);
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY,
        Integer.toString(configuration.contextLength()));
    try {
      Result result = profile(configuration.model().load(), configuration);
      write(configuration.output(), result);
      System.out.printf(
          Locale.ROOT,
          "runtime sessions: mode=%s concurrency=%d requests=%d tokens=%d aggregate=%.2f tok/s "
              + "ttft-p50=%.2f ms total-p95=%.2f ms hash=%s%nreport: %s%n",
          result.mode(),
          result.concurrency(),
          result.successfulRequests(),
          result.completionTokens(),
          result.aggregateTokensPerSecond(),
          result.ttftP50Millis(),
          result.totalP95Millis(),
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
    Mode mode = Mode.parse(values.getOrDefault("mode", "continuous"));
    return new Configuration(
        model,
        prompt,
        BenchmarkCliArguments.integer(values, "context", 2_048),
        BenchmarkCliArguments.integer(values, "concurrency", 4),
        BenchmarkCliArguments.integer(values, "warmups", 1),
        BenchmarkCliArguments.integer(values, "iterations", 3),
        BenchmarkCliArguments.integer(values, "max-tokens", 16),
        booleanValue(values, "batch-prefill-across-sessions", false),
        mode,
        Duration.ofMillis(BenchmarkCliArguments.integer(values, "batch-delay-ms", 1)),
        Path.of(
            values.getOrDefault(
                "output",
                "build/reports/inference/runtime-sessions-" + mode.externalName() + ".json")));
  }

  static Result profile(BatchInferenceBackend backend, Configuration configuration)
      throws Exception {
    Objects.requireNonNull(backend, "backend");
    Objects.requireNonNull(configuration, "configuration");
    if (configuration.concurrency() > backend.maxBatchSize()) {
      throw new IllegalArgumentException(
          "concurrency "
              + configuration.concurrency()
              + " exceeds backend batch capacity "
              + backend.maxBatchSize());
    }

    ContinuousBatchingOptions batching =
        ContinuousBatchingOptions.builder()
            .maximumBatchSize(configuration.concurrency())
            .batchPrefillAcrossSessions(configuration.batchPrefillAcrossSessions())
            .batchFormationDelay(configuration.batchFormationDelay())
            .build();
    long processId = ProcessHandle.current().pid();
    try (InferencePipeline pipeline =
        configuration.mode() == Mode.CONTINUOUS
            ? new InferencePipeline(backend, batching)
            : new InferencePipeline(backend)) {
      for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
        runRound(pipeline, configuration, "warmup", warmup);
      }

      ContinuousBatchingMetrics schedulerBefore = schedulerMetrics(pipeline);
      JvmMemorySnapshot jvmBefore = JvmMemorySnapshot.capture();
      ProcessMetrics.Snapshot processBefore = ProcessMetrics.capture(processId);
      ProfileSupport.GcMetrics gcBefore = ProfileSupport.gcMetrics();
      long started = System.nanoTime();
      List<RequestMeasurement> requests = new ArrayList<>();
      for (int iteration = 0; iteration < configuration.iterations(); iteration++) {
        requests.addAll(runRound(pipeline, configuration, "measurement", iteration));
      }
      long elapsedNanos = Math.max(1, System.nanoTime() - started);
      ProcessMetrics.Snapshot processAfter = ProcessMetrics.capture(processId);
      ProfileSupport.GcMetrics gcAfter = ProfileSupport.gcMetrics();
      JvmMemorySnapshot jvmAfter = JvmMemorySnapshot.capture();
      ContinuousBatchingMetrics schedulerAfter = schedulerMetrics(pipeline);

      int successfulRequests =
          Math.toIntExact(requests.stream().filter(RequestMeasurement::successful).count());
      int completionTokens = requests.stream().mapToInt(RequestMeasurement::completionTokens).sum();
      List<Double> ttft =
          requests.stream()
              .filter(RequestMeasurement::successful)
              .map(RequestMeasurement::ttftMillis)
              .toList();
      List<Double> total =
          requests.stream()
              .filter(RequestMeasurement::successful)
              .map(RequestMeasurement::totalMillis)
              .toList();
      BackendDiagnostics diagnostics = pipeline.diagnostics();
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
          configuration.warmups(),
          configuration.iterations(),
          configuration.maxTokens(),
          configuration.batchPrefillAcrossSessions(),
          successfulRequests,
          requests.size() - successfulRequests,
          completionTokens,
          elapsedNanos,
          completionTokens * 1_000_000_000.0 / elapsedNanos,
          percentileOrZero(ttft, 0.50),
          percentileOrZero(ttft, 0.95),
          percentileOrZero(total, 0.50),
          percentileOrZero(total, 0.95),
          outputHash(requests),
          schedulerDelta(schedulerBefore, schedulerAfter),
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
          diagnostics,
          List.copyOf(requests));
    }
  }

  private static List<RequestMeasurement> runRound(
      InferencePipeline pipeline, Configuration configuration, String phase, int round)
      throws InterruptedException, ExecutionException {
    CountDownLatch ready = new CountDownLatch(configuration.concurrency());
    CountDownLatch start = new CountDownLatch(1);
    List<Future<RequestMeasurement>> futures = new ArrayList<>(configuration.concurrency());
    try (var callers = Executors.newFixedThreadPool(configuration.concurrency())) {
      for (int request = 0; request < configuration.concurrency(); request++) {
        int requestIndex = request;
        futures.add(
            callers.submit(
                () -> {
                  try (TextGenerationSession session = pipeline.openGenerationSession()) {
                    ready.countDown();
                    start.await();
                    long requestStarted = System.nanoTime();
                    String prompt =
                        configuration.prompt()
                            + "\nPhase "
                            + phase
                            + ", round "
                            + round
                            + ", request "
                            + requestIndex
                            + '.';
                    StringBuilder output = new StringBuilder();
                    AtomicLong firstTokenAt = new AtomicLong();
                    AtomicReference<Throwable> failure = new AtomicReference<>();
                    session.generate(
                        prompt,
                        SamplingOptions.builder()
                            .temperature(0)
                            .maxTokens(configuration.maxTokens())
                            .build(),
                        new TokenStream() {
                          @Override
                          public void onToken(String token) {
                            firstTokenAt.compareAndSet(0, System.nanoTime());
                            output.append(token);
                          }

                          @Override
                          public void onComplete() {}

                          @Override
                          public void onError(Throwable generationFailure) {
                            failure.compareAndSet(null, generationFailure);
                          }
                        });
                    long requestCompleted = System.nanoTime();
                    GenerationMetrics metrics = session.lastGenerationMetrics();
                    boolean successful = metrics.successful() && failure.get() == null;
                    String answer = output.toString();
                    return new RequestMeasurement(
                        round,
                        requestIndex,
                        successful,
                        firstTokenAt.get() == 0
                            ? 0
                            : (firstTokenAt.get() - requestStarted) / 1_000_000.0,
                        (requestCompleted - requestStarted) / 1_000_000.0,
                        metrics.usage().promptTokens(),
                        metrics.usage().completionTokens(),
                        Hashing.sha256(answer),
                        answer);
                  }
                }));
      }
      ready.await();
      start.countDown();
      List<RequestMeasurement> measurements = new ArrayList<>(futures.size());
      for (Future<RequestMeasurement> future : futures) {
        measurements.add(future.get());
      }
      return List.copyOf(measurements);
    }
  }

  private static ContinuousBatchingMetrics schedulerMetrics(InferencePipeline pipeline) {
    return pipeline.continuousBatchingMetrics().orElse(null);
  }

  private static SchedulerMeasurement schedulerDelta(
      ContinuousBatchingMetrics before, ContinuousBatchingMetrics after) {
    if (after == null) {
      return null;
    }
    long batchInvocations = after.batchInvocations() - before.batchInvocations();
    long sessionSteps = after.sessionSteps() - before.sessionSteps();
    return new SchedulerMeasurement(
        after.completedRequests() - before.completedRequests(),
        after.failedRequests() - before.failedRequests(),
        after.rejectedRequests() - before.rejectedRequests(),
        batchInvocations,
        sessionSteps,
        after.largestBatch(),
        batchInvocations == 0 ? 0 : (double) sessionSteps / batchInvocations);
  }

  private static double percentileOrZero(List<Double> values, double percentile) {
    return values.isEmpty() ? 0 : BenchmarkStatistics.percentile(values, percentile);
  }

  private static String outputHash(List<RequestMeasurement> requests) {
    StringBuilder canonical = new StringBuilder();
    for (RequestMeasurement request : requests) {
      canonical
          .append(request.round())
          .append(':')
          .append(request.request())
          .append(':')
          .append(request.outputSha256())
          .append('\n');
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

  private static boolean booleanValue(
      Map<String, String> values, String name, boolean defaultValue) {
    String value = values.get(name);
    if (value == null) {
      return defaultValue;
    }
    if ("true".equalsIgnoreCase(value)) {
      return true;
    }
    if ("false".equalsIgnoreCase(value)) {
      return false;
    }
    throw new IllegalArgumentException("--" + name + " must be true or false: " + value);
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
      int maxTokens,
      boolean batchPrefillAcrossSessions,
      Mode mode,
      Duration batchFormationDelay,
      Path output) {

    Configuration {
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(prompt, "prompt");
      Objects.requireNonNull(mode, "mode");
      Objects.requireNonNull(batchFormationDelay, "batchFormationDelay");
      Objects.requireNonNull(output, "output");
      if (prompt.isBlank()) {
        throw new IllegalArgumentException("prompt must not be blank");
      }
      if (contextLength <= 0
          || concurrency <= 0
          || warmups < 0
          || iterations <= 0
          || maxTokens <= 0
          || batchFormationDelay.isNegative()) {
        throw new IllegalArgumentException("runtime session profile options are invalid");
      }
      if (mode == Mode.SERIALIZED && batchPrefillAcrossSessions) {
        throw new IllegalArgumentException("batchPrefillAcrossSessions requires continuous mode");
      }
    }
  }

  record RequestMeasurement(
      int round,
      int request,
      boolean successful,
      double ttftMillis,
      double totalMillis,
      int promptTokens,
      int completionTokens,
      String outputSha256,
      String output) {}

  record SchedulerMeasurement(
      long completedRequests,
      long failedRequests,
      long rejectedRequests,
      long batchInvocations,
      long sessionSteps,
      int largestBatch,
      double meanBatchSize) {}

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
      int warmups,
      int iterations,
      int maxTokens,
      boolean batchPrefillAcrossSessions,
      int successfulRequests,
      int failedRequests,
      int completionTokens,
      long elapsedNanos,
      double aggregateTokensPerSecond,
      double ttftP50Millis,
      double ttftP95Millis,
      double totalP50Millis,
      double totalP95Millis,
      String outputSha256,
      SchedulerMeasurement scheduler,
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
      BackendDiagnostics diagnostics,
      List<RequestMeasurement> requests) {}
}
