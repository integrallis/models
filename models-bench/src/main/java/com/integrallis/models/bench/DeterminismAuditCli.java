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
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.vectors.core.PanamaConstants;
import com.integrallis.vectors.core.VectorizationProvider;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Audits exact greedy-logit repeatability for one pure-Java model and JVM configuration. */
final class DeterminismAuditCli {

  private static final int SCHEMA_VERSION = 1;
  private static final String CONTEXT_LENGTH_PROPERTY = "models.purejava.maxContextLength";
  private static final String DEFAULT_PROMPT =
      "Question: Which organ pumps blood through the human body?\nAnswer:";
  private static final Set<String> OPTIONS =
      Set.of(
          "model",
          "model-id",
          "prompt",
          "prompt-file",
          "tokens",
          "iterations",
          "context",
          "prefill",
          "output");

  private DeterminismAuditCli() {}

  static void run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    System.setProperty(CONTEXT_LENGTH_PROPERTY, Integer.toString(configuration.contextLength()));

    String artifactSha256 = Hashing.sha256(configuration.model());
    long artifactSize = Files.size(configuration.model());
    long loadStart = System.nanoTime();
    Report report;
    try (PureJavaBackend backend = PureJavaBackend.load(configuration.model())) {
      double loadMillis = (System.nanoTime() - loadStart) / 1_000_000.0;
      AuditResult audit =
          audit(
              backend,
              configuration.prompt(),
              configuration.generatedTokens(),
              configuration.iterations(),
              configuration.promptMode());
      report =
          new Report(
              SCHEMA_VERSION,
              Instant.now().toString(),
              configuration.modelId(),
              configuration.model().toAbsolutePath().toString(),
              artifactSha256,
              artifactSize,
              loadMillis,
              Hashing.sha256(configuration.prompt()),
              configuration.prompt(),
              configuration.generatedTokens(),
              configuration.iterations(),
              configuration.contextLength(),
              configuration.promptMode(),
              BenchmarkEnvironment.capture(),
              ExecutionConfiguration.capture(),
              audit.deterministic(),
              audit.promptTokens(),
              audit.trials());
    }

    write(configuration.output(), report);
    printSummary(report, configuration.output());
    if (!report.deterministic()) {
      throw new IllegalStateException(
          "exact logits were not deterministic; see " + configuration.output().toAbsolutePath());
    }
  }

  static Configuration parse(String[] args) throws IOException {
    Map<String, String> values = parseOptions(args);
    Path model = Path.of(required(values, "model"));
    if (!Files.isRegularFile(model)) {
      throw new IllegalArgumentException("model does not exist: " + model);
    }
    String modelId = values.getOrDefault("model-id", safeModelId(model));
    String prompt =
        values.containsKey("prompt-file")
            ? Files.readString(Path.of(values.get("prompt-file")))
            : values.getOrDefault("prompt", DEFAULT_PROMPT);
    if (prompt.isBlank()) {
      throw new IllegalArgumentException("prompt must not be blank");
    }
    int generatedTokens = positiveInteger(values, "tokens", 4);
    int iterations = positiveInteger(values, "iterations", 3);
    int contextLength = positiveInteger(values, "context", 128);
    PromptMode promptMode = PromptMode.parse(values.getOrDefault("prefill", "sequential"));
    Path output =
        Path.of(
            values.getOrDefault(
                "output", "build/reports/determinism/" + modelId + "-determinism.json"));
    return new Configuration(
        model, modelId, prompt, generatedTokens, iterations, contextLength, promptMode, output);
  }

  static AuditResult audit(
      InferenceBackend backend,
      String prompt,
      int generatedTokens,
      int iterations,
      PromptMode promptMode) {
    if (prompt == null || prompt.isBlank()) {
      throw new IllegalArgumentException("prompt must not be blank");
    }
    if (generatedTokens < 1) {
      throw new IllegalArgumentException("generatedTokens must be positive");
    }
    if (iterations < 1) {
      throw new IllegalArgumentException("iterations must be positive");
    }

    int[] promptTokens = backend.tokenizer().encode(prompt);
    if (promptTokens.length == 0) {
      throw new IllegalArgumentException("prompt produced no tokens");
    }

    List<Trial> trials = new ArrayList<>(iterations);
    for (int iteration = 0; iteration < iterations; iteration++) {
      backend.reset();
      float[] logits = prompt(backend, promptTokens, promptMode);
      List<Step> steps = new ArrayList<>(generatedTokens);
      int position = promptTokens.length;
      for (int stepIndex = 0; stepIndex < generatedTokens; stepIndex++) {
        Step step = analyze(stepIndex, logits);
        steps.add(step);
        if (stepIndex + 1 < generatedTokens) {
          logits = backend.forwardTransient(step.winnerToken(), position++);
        }
      }
      String sequenceSha256 =
          Hashing.sha256(
              steps.stream()
                  .map(step -> step.winnerToken() + ":" + step.logitsSha256())
                  .reduce("", (left, right) -> left + right + ";"));
      trials.add(new Trial(iteration, sequenceSha256, List.copyOf(steps)));
    }

    String reference = trials.getFirst().sequenceSha256();
    boolean deterministic =
        trials.stream().allMatch(trial -> reference.equals(trial.sequenceSha256()));
    return new AuditResult(promptTokens.clone(), deterministic, List.copyOf(trials));
  }

  private static float[] prompt(
      InferenceBackend backend, int[] promptTokens, PromptMode promptMode) {
    if (promptMode == PromptMode.BACKEND) {
      return backend.prefill(promptTokens, 0);
    }
    float[] logits = null;
    for (int position = 0; position < promptTokens.length; position++) {
      logits = backend.forwardTransient(promptTokens[position], position);
    }
    return logits;
  }

  private static Step analyze(int stepIndex, float[] logits) {
    if (logits == null || logits.length < 2) {
      throw new IllegalStateException("at least two logits are required");
    }
    int winner = -1;
    int runnerUp = -1;
    for (int index = 0; index < logits.length; index++) {
      float value = logits[index];
      if (!Float.isFinite(value)) {
        throw new IllegalStateException(
            "non-finite logit at step " + stepIndex + ", token " + index + ": " + value);
      }
      if (winner < 0 || value > logits[winner]) {
        runnerUp = winner;
        winner = index;
      } else if (runnerUp < 0 || value > logits[runnerUp]) {
        runnerUp = index;
      }
    }
    return new Step(
        stepIndex,
        winner,
        logits[winner],
        runnerUp,
        logits[runnerUp],
        logits[winner] - logits[runnerUp],
        Hashing.sha256(logits));
  }

  private static void write(Path output, Report report) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    new ObjectMapper()
        .enable(SerializationFeature.INDENT_OUTPUT)
        .writeValue(output.toFile(), report);
  }

  private static void printSummary(Report report, Path output) {
    for (Trial trial : report.trials()) {
      String tokens =
          trial.steps().stream()
              .map(step -> Integer.toString(step.winnerToken()))
              .reduce((left, right) -> left + "," + right)
              .orElse("");
      float minimumMargin =
          trial.steps().stream().map(Step::margin).min(Float::compare).orElse(Float.NaN);
      System.out.printf(
          Locale.ROOT,
          "trial %d/%d: tokens=[%s] min-margin=%.9g hash=%s%n",
          trial.index() + 1,
          report.iterations(),
          tokens,
          minimumMargin,
          trial.sequenceSha256());
    }
    System.out.printf(
        Locale.ROOT,
        "deterministic=%s provider=%s panama=%s report=%s%n",
        report.deterministic(),
        report.execution().provider(),
        report.execution().panama(),
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
    if (values.containsKey("prompt") && values.containsKey("prompt-file")) {
      throw new IllegalArgumentException("--prompt and --prompt-file are mutually exclusive");
    }
    return values;
  }

  private static int positiveInteger(Map<String, String> values, String name, int defaultValue) {
    String configured = values.get(name);
    if (configured == null) {
      return defaultValue;
    }
    try {
      int value = Integer.parseInt(configured);
      if (value < 1) {
        throw new IllegalArgumentException("--" + name + " must be positive: " + configured);
      }
      return value;
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException(
          "--" + name + " must be an integer: " + configured, failure);
    }
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }

  private static String safeModelId(Path model) {
    Path fileName = model.getFileName();
    String source = fileName == null ? model.toString() : fileName.toString();
    String modelId = source.toLowerCase().replaceAll("[^a-z0-9]+", "_").replaceAll("^_+|_+$", "");
    return modelId.isBlank() ? "model" : modelId;
  }

  enum PromptMode {
    SEQUENTIAL,
    BACKEND;

    static PromptMode parse(String configured) {
      try {
        return valueOf(configured.trim().toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException failure) {
        throw new IllegalArgumentException(
            "--prefill must be sequential or backend: " + configured, failure);
      }
    }
  }

  record Configuration(
      Path model,
      String modelId,
      String prompt,
      int generatedTokens,
      int iterations,
      int contextLength,
      PromptMode promptMode,
      Path output) {}

  record Step(
      int index,
      int winnerToken,
      float winnerLogit,
      int runnerUpToken,
      float runnerUpLogit,
      float margin,
      String logitsSha256) {}

  record Trial(int index, String sequenceSha256, List<Step> steps) {}

  record AuditResult(int[] promptTokens, boolean deterministic, List<Trial> trials) {}

  record ExecutionConfiguration(
      String provider,
      boolean panama,
      String panamaFailure,
      int maxVectorBits,
      int preferredVectorBits,
      boolean fastVectorFma,
      boolean fastScalarFma,
      boolean groupedProjections,
      int prefillBatchSize,
      boolean ggufParallel,
      long ggufParallelThreshold,
      String ggufExecutor,
      int ggufThreads,
      int ggufChunksPerThread,
      Map<String, String> configuredProperties) {

    private static final List<String> PROPERTY_NAMES =
        List.of(
            "models.purejava.groupedProjections",
            "models.purejava.prefillBatchSize",
            "vectors.forceScalar",
            "vectors.maxBits",
            "vectors.useVectorFMA",
            "vectors.useScalarFMA",
            "vectors.gguf.parallel",
            "vectors.gguf.parallelThreshold",
            "vectors.gguf.executor",
            "vectors.gguf.threads",
            "vectors.gguf.chunksPerThread");

    static ExecutionConfiguration capture() {
      int processors = Runtime.getRuntime().availableProcessors();
      return new ExecutionConfiguration(
          VectorizationProvider.getProviderName(),
          VectorizationProvider.isPanamaEnabled(),
          VectorizationProvider.getPanamaFailure()
              .map(failure -> failure.getClass().getName() + ": " + failure.getMessage())
              .orElse(null),
          PanamaConstants.MAX_BITS,
          PanamaConstants.PREFERRED_BITS,
          PanamaConstants.HAS_FAST_VECTOR_FMA,
          PanamaConstants.HAS_FAST_SCALAR_FMA,
          Boolean.parseBoolean(System.getProperty("models.purejava.groupedProjections", "true")),
          Integer.getInteger("models.purejava.prefillBatchSize", 32),
          Boolean.parseBoolean(System.getProperty("vectors.gguf.parallel", "true")),
          Math.max(1L, Long.getLong("vectors.gguf.parallelThreshold", 1_048_576L)),
          System.getProperty("vectors.gguf.executor", "persistent").trim().toLowerCase(Locale.ROOT),
          Math.min(Integer.getInteger("vectors.gguf.threads", processors), processors),
          Integer.getInteger("vectors.gguf.chunksPerThread", 2),
          captureConfiguredProperties());
    }

    private static Map<String, String> captureConfiguredProperties() {
      Map<String, String> properties = new LinkedHashMap<>();
      for (String name : PROPERTY_NAMES) {
        properties.put(name, System.getProperty(name, "(default)"));
      }
      return Map.copyOf(properties);
    }
  }

  record Report(
      int schemaVersion,
      String generatedAt,
      String modelId,
      String artifact,
      String artifactSha256,
      long artifactSizeBytes,
      double loadMillis,
      String promptSha256,
      String prompt,
      int generatedTokens,
      int iterations,
      int contextLength,
      PromptMode promptMode,
      BenchmarkEnvironment environment,
      ExecutionConfiguration execution,
      boolean deterministic,
      int[] promptTokens,
      List<Trial> trials) {}
}
