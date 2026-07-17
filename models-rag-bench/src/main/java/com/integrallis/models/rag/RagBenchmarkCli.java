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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Runs controlled plain Java, LangChain4j, and Spring AI RAG workloads. */
public final class RagBenchmarkCli {
  private static final Set<String> FRAMEWORKS = Set.of("plain-java", "langchain4j", "spring-ai");
  private static final Set<String> BACKENDS = Set.of("pure-java", "ollama", "llama.cpp");
  private static final Set<String> OPTIONS =
      Set.of(
          "framework",
          "backend",
          "backend-version",
          "model",
          "model-id",
          "artifact",
          "endpoint",
          "prompt-template",
          "context",
          "threads",
          "pid",
          "top-k",
          "max-tokens",
          "warmups",
          "iterations",
          "case",
          "output");

  private RagBenchmarkCli() {}

  public static void main(String[] args) throws Exception {
    RagBenchmarkConfiguration configuration = parse(args);
    RagBenchmarkReport report = run(configuration);
    write(configuration.output(), report);
    printSummary(report, configuration.output());
  }

  static RagBenchmarkConfiguration parse(String[] args) {
    Map<String, String> values = parseOptions(args);
    String framework = required(values, "framework");
    if (!FRAMEWORKS.contains(framework)) {
      throw new IllegalArgumentException("framework must be one of " + FRAMEWORKS);
    }
    String backend = required(values, "backend");
    if (!BACKENDS.contains(backend)) {
      throw new IllegalArgumentException("backend must be one of " + BACKENDS);
    }
    String model = required(values, "model");
    Path artifact =
        values.containsKey("artifact")
            ? Path.of(values.get("artifact"))
            : "pure-java".equals(backend) ? Path.of(model) : null;
    if (artifact != null && !Files.isRegularFile(artifact)) {
      throw new IllegalArgumentException("artifact does not exist: " + artifact);
    }
    String modelId = values.getOrDefault("model-id", safeId(model));
    URI endpoint =
        URI.create(
            values.getOrDefault(
                "endpoint",
                "ollama".equals(backend) ? "http://127.0.0.1:11434" : "http://127.0.0.1:8080"));
    RagPromptTemplate promptTemplate =
        RagPromptTemplate.parse(values.getOrDefault("prompt-template", "raw"));
    int context = positiveInteger(values, "context", 2_048);
    int threads = positiveInteger(values, "threads", Runtime.getRuntime().availableProcessors());
    long backendPid = nonNegativeLong(values, "pid", 0);
    int topK = positiveInteger(values, "top-k", 1);
    int maxTokens = positiveInteger(values, "max-tokens", 64);
    int warmups = nonNegativeInteger(values, "warmups", 1);
    int iterations = positiveInteger(values, "iterations", 3);
    List<String> caseIds =
        values.containsKey("case")
            ? Arrays.stream(values.get("case").split(","))
                .map(String::trim)
                .filter(value -> !value.isEmpty())
                .toList()
            : List.of();
    Path output =
        Path.of(
            values.getOrDefault(
                "output",
                "build/reports/rag/" + modelId + "-" + framework + "-" + backend + ".json"));
    return new RagBenchmarkConfiguration(
        framework,
        backend,
        values.getOrDefault(
            "backend-version", "pure-java".equals(backend) ? "development" : "unknown"),
        modelId,
        model,
        artifact,
        endpoint,
        promptTemplate,
        context,
        threads,
        backendPid,
        topK,
        maxTokens,
        warmups,
        iterations,
        caseIds,
        output);
  }

  private static RagBenchmarkReport run(RagBenchmarkConfiguration configuration) throws Exception {
    RagCorpus corpus = RagCorpus.loadDefault();
    List<RagCase> cases = selectedCases(corpus, configuration.caseIds());
    Map<String, RagCase> casesById = new LinkedHashMap<>();
    cases.forEach(testCase -> casesById.put(testCase.id(), testCase));
    List<RagRun> runs = new ArrayList<>();
    List<RagBenchmarkFailure> failures = new ArrayList<>();

    try (GenerationClient generation = generationClient(configuration);
        LuceneRagRetriever retriever = new LuceneRagRetriever(corpus.documents());
        RagApplication application =
            application(
                configuration.framework(),
                retriever,
                generation,
                configuration.topK(),
                configuration.promptTemplate())) {
      for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
        for (RagCase testCase : cases) {
          application.run(testCase, configuration.maxTokens());
        }
      }
      for (int iteration = 0; iteration < configuration.iterations(); iteration++) {
        for (RagCase testCase : cases) {
          try {
            RagRun run = application.run(testCase, configuration.maxTokens());
            runs.add(run);
            printTrial(iteration, configuration.iterations(), run);
          } catch (RuntimeException failure) {
            failures.add(new RagBenchmarkFailure(iteration, testCase.id(), failure.toString()));
            System.out.printf(
                "iteration %d/%d case=%s failed: %s%n",
                iteration + 1, configuration.iterations(), testCase.id(), failure);
          }
        }
      }
    }

    int totalAttempts = cases.size() * configuration.iterations();
    RagBenchmarkSummary summary = RagStatistics.summarize(runs, totalAttempts, casesById);
    Path artifact = configuration.artifact();
    return new RagBenchmarkReport(
        RagBenchmarkReport.CURRENT_SCHEMA_VERSION,
        Instant.now().toString(),
        configuration.framework(),
        configuration.backend(),
        configuration.backendVersion(),
        configuration.modelId(),
        configuration.model(),
        artifact == null ? null : sha256(artifact),
        artifact == null ? 0 : Files.size(artifact),
        new RagBenchmarkSettings(
            corpus.fingerprint(),
            cases.stream().map(RagCase::id).toList(),
            configuration.promptTemplate().id(),
            configuration.topK(),
            configuration.maxTokens(),
            configuration.warmups(),
            configuration.iterations(),
            configuration.contextLength(),
            configuration.threads(),
            0,
            1,
            1,
            1,
            42,
            false),
        RagBenchmarkEnvironment.capture(),
        summary,
        RagPerformancePolicy.classify(summary.policyMetrics()),
        runs,
        failures);
  }

  private static GenerationClient generationClient(RagBenchmarkConfiguration configuration) {
    if ("pure-java".equals(configuration.backend())) {
      return PureJavaGenerationClient.load(configuration.artifact(), configuration.contextLength());
    }
    return new HttpGenerationClient(
        configuration.backend(),
        configuration.model(),
        configuration.endpoint(),
        configuration.contextLength(),
        configuration.threads(),
        configuration.backendPid());
  }

  private static RagApplication application(
      String framework,
      RagRetriever retriever,
      GenerationClient client,
      int topK,
      RagPromptTemplate promptTemplate) {
    return switch (framework) {
      case "plain-java" -> new PlainJavaRagApplication(retriever, client, topK, promptTemplate);
      case "langchain4j" -> new LangChain4jRagApplication(retriever, client, topK, promptTemplate);
      case "spring-ai" -> new SpringAiRagApplication(retriever, client, topK, promptTemplate);
      default -> throw new IllegalArgumentException("unknown framework: " + framework);
    };
  }

  private static List<RagCase> selectedCases(RagCorpus corpus, List<String> selected) {
    if (selected.isEmpty()) {
      return corpus.cases();
    }
    Map<String, RagCase> byId = new HashMap<>();
    corpus.cases().forEach(testCase -> byId.put(testCase.id(), testCase));
    return selected.stream()
        .map(
            id -> {
              RagCase testCase = byId.get(id);
              if (testCase == null) {
                throw new IllegalArgumentException("unknown RAG case: " + id);
              }
              return testCase;
            })
        .toList();
  }

  private static void write(Path output, RagBenchmarkReport report) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    new ObjectMapper()
        .enable(SerializationFeature.INDENT_OUTPUT)
        .writeValue(output.toFile(), report);
  }

  private static void printTrial(int iteration, int iterations, RagRun run) {
    System.out.printf(
        "iteration %d/%d case=%s correct=%s retrieval=%.1fms ttft=%.1fms "
            + "decode=%.2f tok/s e2e=%.1fms%n",
        iteration + 1,
        iterations,
        run.caseId(),
        run.evaluation().correct(),
        run.retrievalMillis(),
        run.generation().ttftMillis(),
        run.generation().decodeTokensPerSecond(),
        run.endToEndMillis());
  }

  private static void printSummary(RagBenchmarkReport report, Path output) {
    RagBenchmarkSummary summary = report.summary();
    System.out.printf(
        "%s/%s/%s: tier=%s success=%d/%d p95-retrieval=%.1fms p95-ttft=%.1fms "
            + "p50-decode=%.2f tok/s p95-e2e=%.1fms correct=%.1f%%%nreport: %s%n",
        report.framework(),
        report.backend(),
        report.modelId(),
        report.performanceTier(),
        summary.successfulAttempts(),
        summary.totalAttempts(),
        summary.retrievalMillis().p95(),
        summary.ttftMillis().p95(),
        summary.p50DecodeTokensPerSecond(),
        summary.endToEndMillis().p95(),
        summary.correctAnswerRate() * 100,
        output.toAbsolutePath());
  }

  private static Map<String, String> parseOptions(String[] args) {
    if (args.length % 2 != 0) {
      throw new IllegalArgumentException("every option requires a value");
    }
    Map<String, String> values = new HashMap<>();
    for (int index = 0; index < args.length; index += 2) {
      String raw = args[index];
      if (!raw.startsWith("--")) {
        throw new IllegalArgumentException("expected option, got: " + raw);
      }
      String option = raw.substring(2);
      if (!OPTIONS.contains(option)) {
        throw new IllegalArgumentException("unknown option: --" + option);
      }
      values.put(option, args[index + 1]);
    }
    return values;
  }

  private static String required(Map<String, String> values, String option) {
    String value = values.get(option);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + option + " is required");
    }
    return value;
  }

  private static int positiveInteger(Map<String, String> values, String option, int fallback) {
    int value = integer(values, option, fallback);
    if (value < 1) {
      throw new IllegalArgumentException("--" + option + " must be positive");
    }
    return value;
  }

  private static int nonNegativeInteger(Map<String, String> values, String option, int fallback) {
    int value = integer(values, option, fallback);
    if (value < 0) {
      throw new IllegalArgumentException("--" + option + " must be non-negative");
    }
    return value;
  }

  private static int integer(Map<String, String> values, String option, int fallback) {
    try {
      return values.containsKey(option) ? Integer.parseInt(values.get(option)) : fallback;
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + option + " must be an integer", failure);
    }
  }

  private static long nonNegativeLong(Map<String, String> values, String option, long fallback) {
    try {
      long value = values.containsKey(option) ? Long.parseLong(values.get(option)) : fallback;
      if (value < 0) {
        throw new IllegalArgumentException("--" + option + " must be non-negative");
      }
      return value;
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + option + " must be an integer", failure);
    }
  }

  private static String safeId(String model) {
    Path modelPath = Path.of(model);
    Path fileName = modelPath.getFileName();
    String name = fileName == null ? model : fileName.toString();
    return name.replaceAll("(?i)\\.gguf$", "").replaceAll("[^A-Za-z0-9._-]", "-");
  }

  private static String sha256(Path path) throws IOException {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      try (InputStream input = Files.newInputStream(path);
          DigestInputStream hashing = new DigestInputStream(input, digest)) {
        hashing.transferTo(OutputStreamDiscarder.INSTANCE);
      }
      return HexFormat.of().formatHex(digest.digest());
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
  }

  private static final class OutputStreamDiscarder extends java.io.OutputStream {
    private static final OutputStreamDiscarder INSTANCE = new OutputStreamDiscarder();

    @Override
    public void write(int ignored) {}

    @Override
    public void write(byte[] bytes, int offset, int length) {}
  }
}
