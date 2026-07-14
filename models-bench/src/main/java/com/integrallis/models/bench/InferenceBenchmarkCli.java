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
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Runs comparable pure-Java, Ollama, and llama.cpp inference measurements. */
public final class InferenceBenchmarkCli {

  private static final Set<String> BACKENDS = Set.of("pure-java", "ollama", "llama.cpp");
  private static final Set<String> OPTIONS =
      Set.of(
          "backend",
          "model",
          "model-id",
          "artifact",
          "endpoint",
          "prompt",
          "prompt-file",
          "max-tokens",
          "warmups",
          "iterations",
          "context",
          "threads",
          "output");
  private static final String DEFAULT_PROMPT =
      "Explain why a reproducible local inference benchmark must use identical model bytes, "
          + "deterministic generation settings, explicit warmup, repeated trials, and a complete "
          + "hardware and software manifest. Give a concise answer suitable for a Java engineer.";

  private InferenceBenchmarkCli() {}

  public static void main(String[] args) throws Exception {
    BenchmarkConfiguration configuration = parse(args);
    BenchmarkReport report = run(configuration);
    write(configuration.output(), report);
    printSummary(report, configuration.output());
  }

  static BenchmarkConfiguration parse(String[] args) throws IOException {
    Map<String, String> values = parseOptions(args);
    String backend = required(values, "backend");
    if (!BACKENDS.contains(backend)) {
      throw new IllegalArgumentException("backend must be one of " + BACKENDS + ": " + backend);
    }
    String model = required(values, "model");
    String modelId = values.getOrDefault("model-id", safeModelId(model));
    Path artifact;
    if (values.containsKey("artifact")) {
      artifact = Path.of(values.get("artifact"));
    } else if ("pure-java".equals(backend)) {
      artifact = Path.of(model);
    } else {
      throw new IllegalArgumentException(
          "--artifact is required for HTTP backends so identical model bytes can be verified");
    }
    if (!Files.isRegularFile(artifact)) {
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
    int threads = integer(values, "threads", 1);
    if ("pure-java".equals(backend) && threads != 1) {
      throw new IllegalArgumentException(
          "pure-java currently only supports --threads 1; multi-thread inference is not active");
    }
    Path output =
        Path.of(
            values.getOrDefault(
                "output", "build/reports/inference/" + modelId + "-" + backend + ".json"));
    return new BenchmarkConfiguration(
        backend,
        modelId,
        model,
        artifact,
        endpoint,
        prompt,
        maxTokens,
        warmups,
        iterations,
        contextLength,
        threads,
        output);
  }

  private static BenchmarkReport run(BenchmarkConfiguration configuration) throws IOException {
    BenchmarkArtifact artifact = BenchmarkArtifact.inspect(configuration.artifact());
    List<TrialMeasurement> trials = new ArrayList<>();

    try (BenchmarkTarget target = target(configuration, artifact)) {
      for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
        TrialMeasurement result = target.generate(configuration.prompt(), configuration.maxTokens());
        if (!result.successful()) {
          throw new IllegalStateException("warmup failed: " + result.error());
        }
      }
      for (int iteration = 0; iteration < configuration.iterations(); iteration++) {
        TrialMeasurement result = target.generate(configuration.prompt(), configuration.maxTokens());
        trials.add(result);
        System.out.printf(
            "trial %d/%d: %s ttft=%.1fms decode=%.2f tok/s%n",
            iteration + 1,
            configuration.iterations(),
            result.successful() ? "ok" : "failed",
            result.ttftMillis(),
            result.decodeTokensPerSecond());
      }

      LoadMeasurement load = target.loadMeasurement();
      PerformanceSummary summary = BenchmarkStatistics.summarize(load.millis(), trials);
      return new BenchmarkReport(
          3,
          Instant.now().toString(),
          configuration.backend(),
          configuration.modelId(),
          configuration.model(),
          artifact.sha256(),
          artifact.sizeBytes(),
          target.identity(),
          load,
          new BenchmarkRun(
              Hashing.sha256(configuration.prompt()),
              configuration.maxTokens(),
              configuration.warmups(),
              configuration.iterations(),
              configuration.contextLength(),
              configuration.threads(),
              0,
              42),
          BenchmarkEnvironment.capture(),
          summary,
          BenchmarkPolicy.classify(summary),
          List.copyOf(trials));
    }
  }

  private static BenchmarkTarget target(
      BenchmarkConfiguration configuration, BenchmarkArtifact artifact) throws IOException {
    if ("pure-java".equals(configuration.backend())) {
      return PureJavaBenchmarkTarget.load(artifact, configuration.contextLength());
    }
    return HttpBenchmarkTarget.connect(
        configuration.backend(),
        configuration.model(),
        configuration.endpoint(),
        artifact,
        configuration.contextLength(),
        configuration.threads());
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
        "%s %s: tier=%s load=%.1fms(%s) p95-ttft=%.1fms p50-decode=%.2f tok/s "
            + "p95-tpot=%.1fms peak-rss=%d bytes%nreport: %s%n",
        report.modelId(),
        report.backend(),
        report.performanceTier(),
        summary.loadMillis(),
        report.load().scope(),
        summary.p95TtftMillis(),
        summary.p50DecodeTokensPerSecond(),
        summary.p95TpotMillis(),
        summary.peakRssBytes(),
        output.toAbsolutePath());
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

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }

  private static String safeModelId(String model) {
    String filename = Path.of(model).getFileName().toString();
    return filename
        .toLowerCase()
        .replaceAll("[^a-z0-9]+", "_")
        .replaceAll("^_+|_+$", "");
  }
}
