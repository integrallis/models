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
import com.integrallis.models.runtime.SpeculativeGenerationOptions;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;

/** Runs comparable in-process Models, Ollama, and llama.cpp inference measurements. */
public final class InferenceBenchmarkCli {

  private static final Set<String> BACKENDS =
      Set.of("pure-java", "rust-ffm", "ollama", "llama.cpp");
  private static final String PROMPT_STRATEGY = "sha256-nonce-prefix-v1";
  private static final Set<String> OPTIONS =
      Set.of(
          "backend",
          "model",
          "modeljar",
          "model-id",
          "artifact",
          "endpoint",
          "prompt",
          "prompt-file",
          "max-tokens",
          "warmups",
          "iterations",
          "context",
          "backend-version",
          "threads",
          "pid",
          "load-ms",
          "speculation",
          "ngram-size",
          "draft-probe",
          "draft-min",
          "draft-max",
          "speculation-history",
          "speculation-window",
          "speculation-min-acceptance",
          "speculation-cooldown",
          "output");
  private static final String DEFAULT_PROMPT =
      "Explain why a reproducible local inference benchmark must use identical model bytes, "
          + "deterministic generation settings, explicit warmup, repeated trials, and a complete "
          + "hardware and software manifest. Give a concise answer suitable for a Java engineer.";

  private InferenceBenchmarkCli() {}

  public static void main(String[] args) throws Exception {
    if (args.length > 0 && "profile-prefill".equals(args[0])) {
      PrefillProfileCli.run(Arrays.copyOfRange(args, 1, args.length));
      return;
    }
    if (args.length > 0 && "profile-decode".equals(args[0])) {
      DecodeProfileCli.run(Arrays.copyOfRange(args, 1, args.length));
      return;
    }
    if (args.length > 0 && "compare".equals(args[0])) {
      BenchmarkComparisonCli.run(Arrays.copyOfRange(args, 1, args.length));
      return;
    }
    if (args.length > 0 && "determinism".equals(args[0])) {
      DeterminismAuditCli.run(Arrays.copyOfRange(args, 1, args.length));
      return;
    }
    BenchmarkConfiguration configuration = parse(args);
    BenchmarkReport report = run(configuration);
    write(configuration.output(), report);
    printSummary(report, configuration.output());
  }

  static BenchmarkConfiguration parse(String[] args) throws IOException {
    return parse(args, ModelJarRegistry.fromClasspath());
  }

  static BenchmarkConfiguration parse(String[] args, ModelJarRegistry modelJarRegistry)
      throws IOException {
    Objects.requireNonNull(modelJarRegistry, "modelJarRegistry");
    Map<String, String> values = parseOptions(args);
    String backend = required(values, "backend");
    if (!BACKENDS.contains(backend)) {
      throw new IllegalArgumentException("backend must be one of " + BACKENDS + ": " + backend);
    }
    String model;
    Path artifact;
    Optional<ModelJarDescriptor> modelJarDescriptor;
    if (isInProcess(backend)) {
      if (values.containsKey("artifact")) {
        throw new IllegalArgumentException("--artifact cannot override an in-process model source");
      }
      if ("rust-ffm".equals(backend) && values.containsKey("modeljar")) {
        throw new IllegalArgumentException(
            "--modeljar requires rust-ffm catalog metadata and is not enabled yet");
      }
      PureJavaModelSource source =
          PureJavaModelSource.resolve(
              values.get("model"), values.get("modeljar"), modelJarRegistry);
      model = source.identity();
      artifact = source.artifact();
      modelJarDescriptor = source.descriptor();
    } else {
      if (values.containsKey("modeljar")) {
        throw new IllegalArgumentException("--modeljar is supported only by pure-java");
      }
      model = required(values, "model");
      artifact = values.containsKey("artifact") ? Path.of(values.get("artifact")) : null;
      modelJarDescriptor = Optional.empty();
    }
    String modelId = values.getOrDefault("model-id", safeModelId(model));
    if (artifact != null && !Files.isRegularFile(artifact)) {
      throw new IllegalArgumentException("artifact does not exist: " + artifact);
    }
    String prompt =
        values.containsKey("prompt-file")
            ? Files.readString(Path.of(values.get("prompt-file")))
            : values.getOrDefault("prompt", DEFAULT_PROMPT);
    URI endpoint =
        URI.create(
            values.getOrDefault(
                "endpoint",
                "ollama".equals(backend) ? "http://127.0.0.1:11434" : "http://127.0.0.1:8080"));
    int maxTokens = integer(values, "max-tokens", 64);
    int warmups = integer(values, "warmups", 2);
    int iterations = integer(values, "iterations", 10);
    int contextLength = integer(values, "context", 2_048);
    String backendVersion =
        values.getOrDefault("backend-version", isInProcess(backend) ? "development" : "unknown");
    int threads = integer(values, "threads", Runtime.getRuntime().availableProcessors());
    int availableProcessors = Runtime.getRuntime().availableProcessors();
    if (isInProcess(backend) && threads != availableProcessors) {
      throw new IllegalArgumentException(
          backend
              + " uses the JVM processor allocation; --threads must be "
              + availableProcessors
              + " for this process");
    }
    long backendPid = longValue(values, "pid", 0);
    double loadMillis = doubleValue(values, "load-ms", 0);
    SpeculativeGenerationOptions speculativeOptions = speculativeOptions(values, backend);
    String outputSuffix = speculativeOptions.enabled() ? "-ngram" : "";
    Path output =
        Path.of(
            values.getOrDefault(
                "output",
                "build/reports/inference/" + modelId + "-" + backend + outputSuffix + ".json"));
    return new BenchmarkConfiguration(
        backend,
        modelId,
        model,
        artifact,
        modelJarDescriptor,
        endpoint,
        prompt,
        maxTokens,
        warmups,
        iterations,
        contextLength,
        backendVersion,
        threads,
        backendPid,
        loadMillis,
        speculativeOptions,
        output);
  }

  private static BenchmarkReport run(BenchmarkConfiguration configuration) throws IOException {
    String artifactSha256 =
        configuration.artifact() == null ? null : Hashing.sha256(configuration.artifact());
    long artifactSize = configuration.artifact() == null ? 0 : Files.size(configuration.artifact());
    List<TrialMeasurement> trials = new ArrayList<>();

    try (BenchmarkTarget target = target(configuration)) {
      for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
        TrialMeasurement result =
            target.generate(
                benchmarkPrompt(configuration.prompt(), "warmup", warmup),
                configuration.maxTokens());
        if (!result.successful()) {
          throw new IllegalStateException("warmup failed: " + result.error());
        }
      }
      for (int iteration = 0; iteration < configuration.iterations(); iteration++) {
        TrialMeasurement result =
            target.generate(
                benchmarkPrompt(configuration.prompt(), "measurement", iteration),
                configuration.maxTokens());
        trials.add(result);
        printTrial(iteration, configuration.iterations(), result);
      }

      PerformanceSummary summary = BenchmarkStatistics.summarize(target.loadMillis(), trials);
      return new BenchmarkReport(
          BenchmarkReport.CURRENT_SCHEMA_VERSION,
          Instant.now().toString(),
          configuration.backend(),
          configuration.backendVersion(),
          configuration.modelId(),
          configuration.model(),
          artifactSha256,
          artifactSize,
          new BenchmarkRun(
              Hashing.sha256(configuration.prompt()),
              PROMPT_STRATEGY,
              configuration.maxTokens(),
              configuration.warmups(),
              configuration.iterations(),
              configuration.contextLength(),
              configuration.threads(),
              0,
              1,
              1,
              1,
              true,
              42),
          BenchmarkEnvironment.capture(),
          target.diagnostics(),
          configuration.speculativeOptions(),
          summary,
          BenchmarkPolicy.classify(summary),
          List.copyOf(trials));
    }
  }

  private static BenchmarkTarget target(BenchmarkConfiguration configuration) {
    if ("pure-java".equals(configuration.backend())) {
      return PureJavaBenchmarkTarget.load(
          new PureJavaModelSource(
              configuration.model(), configuration.artifact(), configuration.modelJarDescriptor()),
          configuration.contextLength(),
          configuration.speculativeOptions());
    }
    if ("rust-ffm".equals(configuration.backend())) {
      return PureJavaBenchmarkTarget.loadRust(
          configuration.artifact(),
          configuration.contextLength(),
          configuration.speculativeOptions());
    }
    return new HttpBenchmarkTarget(
        configuration.backend(),
        configuration.model(),
        configuration.endpoint(),
        configuration.contextLength(),
        configuration.threads(),
        configuration.backendPid(),
        configuration.loadMillis());
  }

  private static void write(Path output, BenchmarkReport report) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    new ObjectMapper()
        .enable(SerializationFeature.INDENT_OUTPUT)
        .writeValue(output.toFile(), report);
  }

  private static void printSummary(BenchmarkReport report, Path output) {
    PerformanceSummary summary = report.summary();
    System.out.printf(
        "%s %s: tier=%s load=%.1fms p95-ttft=%.1fms p50-decode=%.2f tok/s "
            + "p95-tpot=%.1fms peak-rss=%d bytes%nreport: %s%n",
        report.modelId(),
        report.backend(),
        report.performanceTier(),
        summary.loadMillis(),
        summary.p95TtftMillis(),
        summary.p50DecodeTokensPerSecond(),
        summary.p95TpotMillis(),
        summary.peakRssBytes(),
        output.toAbsolutePath());
  }

  private static void printTrial(int iteration, int iterations, TrialMeasurement result) {
    System.out.printf(
        "trial %d/%d: %s ttft=%.1fms decode=%.2f tok/s",
        iteration + 1,
        iterations,
        result.successful() ? "ok" : "failed",
        result.ttftMillis(),
        result.decodeTokensPerSecond());
    if (result.speculation() != null && result.speculation().active()) {
      System.out.printf(
          " drafts=%d/%d accepted=%.1f%% verify=%.1fms",
          result.speculation().acceptedTokens(),
          result.speculation().proposedTokens(),
          result.speculation().acceptanceRate() * 100.0,
          result.speculation().verificationNanos() / 1_000_000.0);
    }
    System.out.println();
  }

  private static Map<String, String> parseOptions(String[] args) {
    Map<String, String> values = new HashMap<>();
    for (int index = 0; index < args.length; index += 2) {
      String option = args[index];
      if (!option.startsWith("--") || index + 1 >= args.length) {
        throw new IllegalArgumentException("options must use --name value pairs: " + option);
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
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be an integer: " + value, failure);
    }
  }

  private static long longValue(Map<String, String> values, String name, long defaultValue) {
    String value = values.get(name);
    if (value == null) {
      return defaultValue;
    }
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be an integer: " + value, failure);
    }
  }

  private static double doubleValue(Map<String, String> values, String name, double defaultValue) {
    String value = values.get(name);
    if (value == null) {
      return defaultValue;
    }
    try {
      return Double.parseDouble(value);
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be a number: " + value, failure);
    }
  }

  private static SpeculativeGenerationOptions speculativeOptions(
      Map<String, String> values, String backend) {
    String mode = values.getOrDefault("speculation", "none");
    if ("none".equals(mode)) {
      return SpeculativeGenerationOptions.disabled();
    }
    if (!"ngram".equals(mode)) {
      throw new IllegalArgumentException("--speculation must be none or ngram: " + mode);
    }
    if (!isInProcess(backend)) {
      throw new IllegalArgumentException(
          "--speculation ngram is supported only by in-process Models backends");
    }

    SpeculativeGenerationOptions defaults = SpeculativeGenerationOptions.builder().build();
    return SpeculativeGenerationOptions.builder()
        .ngramSize(integer(values, "ngram-size", defaults.ngramSize()))
        .confidenceProbeTokens(integer(values, "draft-probe", defaults.confidenceProbeTokens()))
        .minimumDraftTokens(integer(values, "draft-min", defaults.minimumDraftTokens()))
        .maximumDraftTokens(integer(values, "draft-max", defaults.maximumDraftTokens()))
        .historyWindow(integer(values, "speculation-history", defaults.historyWindow()))
        .adaptationWindow(integer(values, "speculation-window", defaults.adaptationWindow()))
        .minimumAcceptanceRate(
            (float)
                doubleValue(values, "speculation-min-acceptance", defaults.minimumAcceptanceRate()))
        .cooldownTokens(integer(values, "speculation-cooldown", defaults.cooldownTokens()))
        .build();
  }

  private static boolean isInProcess(String backend) {
    return "pure-java".equals(backend) || "rust-ffm".equals(backend);
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }

  private static String safeModelId(String model) {
    Path modelPath = Path.of(model);
    Path fileName = modelPath.getFileName();
    String source = fileName == null ? modelPath.toString() : fileName.toString();
    String modelId = source.toLowerCase().replaceAll("[^a-z0-9]+", "_").replaceAll("^_+|_+$", "");
    return modelId.isBlank() ? "model" : modelId;
  }

  static String benchmarkPrompt(String basePrompt, String phase, int index) {
    if (basePrompt == null
        || basePrompt.isBlank()
        || phase == null
        || phase.isBlank()
        || index < 0) {
      throw new IllegalArgumentException("benchmark prompt inputs are invalid");
    }
    String identity =
        PROMPT_STRATEGY + '\0' + phase + '\0' + index + '\0' + Hashing.sha256(basePrompt);
    return Hashing.sha256(identity).substring(0, 16) + '\n' + basePrompt;
  }
}
