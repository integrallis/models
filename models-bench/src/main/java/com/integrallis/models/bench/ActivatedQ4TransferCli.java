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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.ToolApplicabilityHead;
import com.integrallis.models.runtime.ToolApplicabilityScore;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/** Calibrates on all exposed Q4 cases, then screens an independently frozen live window. */
final class ActivatedQ4TransferCli {
  static final String EXPERIMENT = "qwen3-17b-q4-transfer-v18";
  static final String LIVE_RECORDS_SHA256 =
      "4d70a040341480ccbae9af5a00d83bc4e7365ebead67e390cb5c3763ed83b30a";
  static final String LIVE_MANIFEST_SHA256 =
      "f1815fa298e8b4e205b061a26ed8a1043e7aa688f7a4b40d034e7c329d14507a";
  static final String LIVE_CASE_SET_SHA256 =
      "ce93b078003095c0db03d4cb2099f2b1e0ddeac7df6c8e819260ade29a1690fc";
  private static final String LIVE_REVISION = "c15b2a151662cac9839c96d7dfb1493b5329c975";
  private static final int CASES = 75;
  private static final int CALLS = 50;
  private static final int NO_CALLS = 25;
  private static final Set<String> KINDS = Set.of("simple", "multiple", "irrelevance");
  private static final Set<String> CALIBRATION_OPTIONS =
      Set.of("model", "adapter", "head", "records", "output", "models-revision");
  private static final Set<String> SCREEN_OPTIONS =
      Set.of(
          "model",
          "adapter",
          "head",
          "records",
          "window-manifest",
          "window-manifest-sha256",
          "calibration",
          "calibration-sha256",
          "calibration-records",
          "output",
          "models-revision");

  private ActivatedQ4TransferCli() {}

  record Configuration(
      Path model, Path adapter, Path head, Path records, Path output, String modelsRevision) {}

  record CalibrationReport(
      int schemaVersion,
      String experiment,
      String phase,
      String createdAt,
      String modelsRevision,
      String sourceSha256,
      String modelSha256,
      String headFileSha256,
      String headArtifactSha256,
      ActivatedAdapterMetadata adapter,
      BenchmarkEnvironment environment,
      double threshold,
      ActivatedQ4CalibrationCli.Score result,
      boolean passed,
      String verdict,
      List<ActivatedQ4CalibrationCli.Observation> observations) {}

  record ScreenReport(
      int schemaVersion,
      String experiment,
      String phase,
      String createdAt,
      String modelsRevision,
      String sourceSha256,
      String sourceManifestSha256,
      String sourceCaseSetSha256,
      String modelSha256,
      String headFileSha256,
      String headArtifactSha256,
      String calibrationFileSha256,
      ActivatedAdapterMetadata adapter,
      BenchmarkEnvironment environment,
      double threshold,
      ActivatedQ4CalibrationCli.Score calibration,
      ActivatedQ4CalibrationCli.Score screen,
      boolean passed,
      String verdict,
      List<ActivatedQ4CalibrationCli.Observation> observations) {}

  static int runCalibration(String[] args) throws IOException {
    Configuration configuration = parse(args, CALIBRATION_OPTIONS);
    requireCommonHashes(configuration);
    requireHash(
        configuration.records(), ActivatedDecisionProfileCli.SOURCE_SHA256, "calibration records");
    ObjectMapper mapper = ActivatedDecisionProfileCli.mapper();
    List<ActivatedDecisionProfileCli.SourceCase> exposed =
        ActivatedDecisionProfileCli.loadCases(mapper, configuration.records()).cases().stream()
            .map(
                item ->
                    new ActivatedDecisionProfileCli.SourceCase(
                        item.id(), item.kind(), "calibration", item.callExpected(), item.prompt()))
            .toList();
    ScoredRun run = scoreCases(configuration, exposed, "calibration", Double.NaN);
    ActivatedQ4CalibrationCli.ThresholdSelection selection =
        ActivatedQ4CalibrationCli.selectThreshold(run.observations());
    List<ActivatedQ4CalibrationCli.Observation> observations =
        predict(run.observations(), selection.threshold());
    long physicallyShared = physicalCount(observations);
    boolean passed = phasePassed(selection.score(), observations.size(), physicallyShared);
    CalibrationReport report =
        new CalibrationReport(
            1,
            EXPERIMENT,
            "calibration",
            Instant.now().toString(),
            configuration.modelsRevision(),
            ActivatedDecisionProfileCli.SOURCE_SHA256,
            ActivatedHiddenDecisionProfileCli.MODEL_SHA256,
            ActivatedHiddenDecisionProfileCli.HEAD_FILE_SHA256,
            ActivatedHiddenDecisionProfileCli.HEAD_ARTIFACT_SHA256,
            run.adapter(),
            BenchmarkEnvironment.capture(),
            selection.threshold(),
            selection.score(),
            passed,
            passed ? "PASS" : "FAIL",
            observations);
    write(mapper, configuration.output(), report);
    printResult(
        report.verdict(),
        report.threshold(),
        report.result(),
        physicallyShared,
        observations.size(),
        configuration.output());
    return passed ? 0 : 1;
  }

  static int runScreen(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, SCREEN_OPTIONS);
    Configuration configuration = parse(values);
    requireCommonHashes(configuration);
    requireHash(configuration.records(), LIVE_RECORDS_SHA256, "live records");
    Path manifest = requiredFile(values, "window-manifest");
    String manifestHash = requiredSha256(values, "window-manifest-sha256");
    if (!LIVE_MANIFEST_SHA256.equals(manifestHash)) {
      throw new IllegalArgumentException("V18 live manifest hash differs from the frozen protocol");
    }
    requireHash(manifest, manifestHash, "live manifest");

    ObjectMapper mapper = ActivatedDecisionProfileCli.mapper();
    verifyLiveManifest(mapper, manifest);
    List<ActivatedDecisionProfileCli.SourceCase> live =
        loadLiveCases(mapper, configuration.records());
    requireLiveShape(live);

    Path calibrationPath = requiredFile(values, "calibration");
    String calibrationHash = requiredSha256(values, "calibration-sha256");
    requireHash(calibrationPath, calibrationHash, "calibration report");
    Path calibrationRecords = requiredFile(values, "calibration-records");
    requireHash(
        calibrationRecords,
        ActivatedDecisionProfileCli.SOURCE_SHA256,
        "calibration source records");
    Map<String, ActivatedDecisionProfileCli.SourceCase> calibrationCases =
        ActivatedDecisionProfileCli.loadCases(mapper, calibrationRecords).cases().stream()
            .collect(
                Collectors.toUnmodifiableMap(
                    ActivatedDecisionProfileCli.SourceCase::id, item -> item));
    CalibrationInput calibration =
        loadCalibration(mapper, calibrationPath, configuration, calibrationHash, calibrationCases);
    ScoredRun run = scoreCases(configuration, live, "screen", calibration.threshold());
    List<ActivatedQ4CalibrationCli.Observation> observations =
        predict(run.observations(), calibration.threshold());
    ActivatedQ4CalibrationCli.Score screen =
        ActivatedQ4CalibrationCli.score(observations, calibration.threshold(), "screen");
    long physicallyShared = physicalCount(observations);
    boolean passed = phasePassed(screen, observations.size(), physicallyShared);
    ScreenReport report =
        new ScreenReport(
            1,
            EXPERIMENT,
            "screen",
            Instant.now().toString(),
            configuration.modelsRevision(),
            LIVE_RECORDS_SHA256,
            manifestHash,
            LIVE_CASE_SET_SHA256,
            ActivatedHiddenDecisionProfileCli.MODEL_SHA256,
            ActivatedHiddenDecisionProfileCli.HEAD_FILE_SHA256,
            ActivatedHiddenDecisionProfileCli.HEAD_ARTIFACT_SHA256,
            calibrationHash,
            run.adapter(),
            BenchmarkEnvironment.capture(),
            calibration.threshold(),
            calibration.score(),
            screen,
            passed,
            passed ? "PASS" : "FAIL",
            observations);
    write(mapper, configuration.output(), report);
    printResult(
        report.verdict(),
        report.threshold(),
        report.screen(),
        physicallyShared,
        observations.size(),
        configuration.output());
    return passed ? 0 : 1;
  }

  static boolean phasePassed(
      ActivatedQ4CalibrationCli.Score score, long observations, long physicallyShared) {
    return observations == CASES
        && physicallyShared == CASES
        && score.calls() == CALLS
        && score.noCalls() == NO_CALLS
        && score.correctCalls() >= 47
        && score.correctNoCalls() >= 24
        && score.balancedAccuracy() >= 0.95;
  }

  static List<ActivatedDecisionProfileCli.SourceCase> loadLiveCases(ObjectMapper mapper, Path path)
      throws IOException {
    Map<String, ActivatedDecisionProfileCli.SourceCase> unique = new LinkedHashMap<>();
    try (var lines = Files.lines(path)) {
      for (String line : lines.toList()) {
        JsonNode item = mapper.readTree(line);
        if (!"screen".equals(item.path("phase").asText())) {
          throw new IllegalArgumentException("V18 live record phase must be screen");
        }
        String id = requiredText(item, "id");
        String kind = requiredText(item, "kind");
        if (!KINDS.contains(kind)) {
          throw new IllegalArgumentException("unexpected V18 live case kind: " + kind);
        }
        ActivatedDecisionProfileCli.requireNoControlMarkers(
            item.path("messages"), id + ".messages");
        ActivatedDecisionProfileCli.requireNoControlMarkers(item.path("tools"), id + ".tools");
        List<ChatMessage> messages = messages(item.path("messages"));
        List<ToolSpec> tools = tools(mapper, item.path("tools"));
        ModelPrompt rendered = ChatTemplate.CHATML_NO_THINK.render(messages, tools);
        JsonNode expected = item.path("expected");
        if (!expected.isArray()) {
          throw new IllegalArgumentException("expected calls must be an array for " + id);
        }
        var sourceCase =
            new ActivatedDecisionProfileCli.SourceCase(
                id, kind, "screen", !expected.isEmpty(), rendered);
        if (unique.put(id, sourceCase) != null) {
          throw new IllegalArgumentException("duplicate V18 live case: " + id);
        }
      }
    }
    return unique.values().stream()
        .sorted(
            Comparator.comparing(ActivatedDecisionProfileCli.SourceCase::kind)
                .thenComparing(ActivatedDecisionProfileCli.SourceCase::id))
        .toList();
  }

  private static ScoredRun scoreCases(
      Configuration configuration,
      List<ActivatedDecisionProfileCli.SourceCase> cases,
      String partition,
      double threshold) {
    if (cases.isEmpty() || cases.stream().anyMatch(item -> !partition.equals(item.partition()))) {
      throw new IllegalArgumentException("V18 cases must all belong to " + partition);
    }
    ToolApplicabilityHead head;
    try {
      head =
          ActivatedHiddenDecisionProfileCli.loadHead(
              ActivatedDecisionProfileCli.mapper(), configuration.head());
    } catch (IOException failure) {
      throw new IllegalStateException("cannot load V18 applicability head", failure);
    }
    List<ActivatedQ4CalibrationCli.Observation> observations = new ArrayList<>();
    PureJavaBackend backend =
        PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
    ActivatedToolCallingModel activatedModel;
    try {
      activatedModel = new ActivatedToolCallingModel(backend, 1);
    } catch (RuntimeException | Error failure) {
      try {
        backend.close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
    ActivatedAdapterMetadata adapter;
    try (ActivatedToolCallingModel activated = activatedModel) {
      adapter = activated.adapter();
      int ordinal = 0;
      for (ActivatedDecisionProfileCli.SourceCase item : cases) {
        ordinal++;
        long started = System.nanoTime();
        try (ActivatedToolTurn turn =
            activated.openToolTurn(
                item.prompt(), ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          ToolApplicabilityScore applicability = turn.scoreToolApplicability(head);
          long elapsedMillis = (System.nanoTime() - started) / 1_000_000L;
          boolean predicted = Double.isFinite(threshold) && applicability.score() > threshold;
          var observation =
              new ActivatedQ4CalibrationCli.Observation(
                  item.id(),
                  item.kind(),
                  item.partition(),
                  item.callExpected(),
                  backend.tokenizer().encode(item.prompt()).length,
                  turn.sharedPrefixTokens(),
                  turn.sharedPrefixBytes(),
                  turn.physicallySharesPrefix(),
                  applicability.score(),
                  predicted,
                  elapsedMillis);
          observations.add(observation);
          System.out.printf(
              "%2d/%d %-24s expected=%-5s score=%11.5f shared=%s %d ms%n",
              ordinal,
              cases.size(),
              item.id(),
              item.callExpected(),
              observation.score(),
              observation.physicallyShared(),
              elapsedMillis);
        }
      }
    }
    return new ScoredRun(adapter, List.copyOf(observations));
  }

  private static CalibrationInput loadCalibration(
      ObjectMapper mapper,
      Path path,
      Configuration configuration,
      String calibrationHash,
      Map<String, ActivatedDecisionProfileCli.SourceCase> calibrationCases)
      throws IOException {
    JsonNode value = mapper.readTree(path.toFile());
    if (value.path("schemaVersion").asInt() != 1
        || !EXPERIMENT.equals(value.path("experiment").asText())
        || !"calibration".equals(value.path("phase").asText())
        || !value.path("passed").asBoolean()
        || !"PASS".equals(value.path("verdict").asText())
        || !configuration.modelsRevision().equals(value.path("modelsRevision").asText())
        || !ActivatedDecisionProfileCli.SOURCE_SHA256.equals(value.path("sourceSha256").asText())
        || !ActivatedHiddenDecisionProfileCli.MODEL_SHA256.equals(
            value.path("modelSha256").asText())
        || !ActivatedHiddenDecisionProfileCli.HEAD_FILE_SHA256.equals(
            value.path("headFileSha256").asText())
        || !ActivatedHiddenDecisionProfileCli.HEAD_ARTIFACT_SHA256.equals(
            value.path("headArtifactSha256").asText())
        || !ActivatedHiddenDecisionProfileCli.ADAPTER_SHA256.equals(
            value.path("adapter").path("adapterSha256").asText())) {
      throw new IllegalArgumentException(
          "V18 calibration report identity or passing verdict changed: " + calibrationHash);
    }
    double threshold = value.path("threshold").asDouble(Double.NaN);
    if (!Double.isFinite(threshold)) {
      throw new IllegalArgumentException("V18 calibration threshold must be finite");
    }
    JsonNode values = value.path("observations");
    if (!values.isArray() || values.size() != CASES) {
      throw new IllegalArgumentException("V18 calibration must contain 75 observations");
    }
    List<ActivatedQ4CalibrationCli.Observation> observations = new ArrayList<>();
    Set<String> ids = new java.util.HashSet<>();
    for (JsonNode item : values) {
      String id = requiredText(item, "id");
      double score = item.path("score").asDouble(Double.NaN);
      ActivatedDecisionProfileCli.SourceCase expected = calibrationCases.get(id);
      if (expected == null
          || !ids.add(id)
          || !"calibration".equals(item.path("partition").asText())
          || !expected.kind().equals(item.path("kind").asText())
          || expected.callExpected() != item.path("callExpected").asBoolean()
          || !item.path("physicallyShared").asBoolean()
          || !Double.isFinite(score)
          || item.path("callPredicted").asBoolean() != (score > threshold)) {
        throw new IllegalArgumentException("V18 calibration observation changed: " + id);
      }
      observations.add(
          new ActivatedQ4CalibrationCli.Observation(
              id,
              expected.kind(),
              "calibration",
              expected.callExpected(),
              item.path("promptTokens").asInt(),
              item.path("sharedPrefixTokens").asInt(),
              item.path("sharedPrefixBytes").asLong(),
              true,
              score,
              item.path("callPredicted").asBoolean(),
              item.path("elapsedMillis").asLong()));
    }
    ActivatedQ4CalibrationCli.ThresholdSelection recomputed =
        ActivatedQ4CalibrationCli.selectThreshold(observations);
    ActivatedQ4CalibrationCli.Score declared = readScore(value.path("result"));
    if (Double.doubleToLongBits(recomputed.threshold()) != Double.doubleToLongBits(threshold)
        || !recomputed.score().equals(declared)
        || observations.size() != calibrationCases.size()
        || !phasePassed(declared, observations.size(), physicalCount(observations))) {
      throw new IllegalArgumentException("V18 calibration report is not reproducible");
    }
    return new CalibrationInput(threshold, declared);
  }

  private static ActivatedQ4CalibrationCli.Score readScore(JsonNode value) {
    var score =
        new ActivatedQ4CalibrationCli.Score(
            value.path("calls").asInt(-1),
            value.path("noCalls").asInt(-1),
            value.path("correctCalls").asInt(-1),
            value.path("correctNoCalls").asInt(-1),
            value.path("callAccuracy").asDouble(Double.NaN),
            value.path("noCallAccuracy").asDouble(Double.NaN),
            value.path("balancedAccuracy").asDouble(Double.NaN));
    if (score.calls() < 0
        || score.noCalls() < 0
        || score.correctCalls() < 0
        || score.correctNoCalls() < 0
        || !Double.isFinite(score.callAccuracy())
        || !Double.isFinite(score.noCallAccuracy())
        || !Double.isFinite(score.balancedAccuracy())) {
      throw new IllegalArgumentException("V18 calibration score is invalid");
    }
    return score;
  }

  private static List<ActivatedQ4CalibrationCli.Observation> predict(
      List<ActivatedQ4CalibrationCli.Observation> observations, double threshold) {
    return observations.stream()
        .map(
            item ->
                new ActivatedQ4CalibrationCli.Observation(
                    item.id(),
                    item.kind(),
                    item.partition(),
                    item.callExpected(),
                    item.promptTokens(),
                    item.sharedPrefixTokens(),
                    item.sharedPrefixBytes(),
                    item.physicallyShared(),
                    item.score(),
                    item.score() > threshold,
                    item.elapsedMillis()))
        .toList();
  }

  private static void verifyLiveManifest(ObjectMapper mapper, Path path) throws IOException {
    JsonNode manifest = mapper.readTree(path.toFile());
    if (manifest.path("schemaVersion").asInt() != 1
        || !"frozen-before-v18-scoring".equals(manifest.path("status").asText())
        || manifest.path("selection").path("seed").asInt() != 20260915
        || manifest.path("selection").path("perKind").asInt() != 25
        || !LIVE_CASE_SET_SHA256.equals(manifest.path("selection").path("caseSetSha256").asText())
        || !LIVE_RECORDS_SHA256.equals(manifest.path("records").path("sha256").asText())
        || !LIVE_REVISION.equals(manifest.path("source").path("revision").asText())
        || manifest.path("staticEvaluationOverlapQueries").asInt(-1) != 0
        || manifest.path("completeTrainingSourceOverlapQueries").asInt(-1) != 0) {
      throw new IllegalArgumentException("V18 live manifest contract changed");
    }
  }

  private static void requireLiveShape(List<ActivatedDecisionProfileCli.SourceCase> cases) {
    Map<String, Long> counts =
        cases.stream()
            .collect(
                Collectors.groupingBy(
                    ActivatedDecisionProfileCli.SourceCase::kind,
                    HashMap::new,
                    Collectors.counting()));
    long calls =
        cases.stream().filter(ActivatedDecisionProfileCli.SourceCase::callExpected).count();
    if (cases.size() != CASES
        || counts.size() != 3
        || KINDS.stream().anyMatch(kind -> counts.getOrDefault(kind, 0L) != 25)
        || calls != CALLS) {
      throw new IllegalArgumentException("V18 live window must contain 25 cases per kind");
    }
  }

  private static List<ChatMessage> messages(JsonNode value) {
    if (!value.isArray() || value.isEmpty()) {
      throw new IllegalArgumentException("messages must be a nonempty array");
    }
    List<ChatMessage> result = new ArrayList<>();
    for (JsonNode item : value) {
      String role = requiredText(item, "role");
      String content = requiredText(item, "content");
      result.add(
          switch (role) {
            case "system" -> ChatMessage.system(content);
            case "user" -> ChatMessage.user(content);
            case "assistant" -> ChatMessage.assistant(content);
            default -> throw new IllegalArgumentException("unsupported source role: " + role);
          });
    }
    return List.copyOf(result);
  }

  private static List<ToolSpec> tools(ObjectMapper mapper, JsonNode value) {
    if (!value.isArray() || value.isEmpty()) {
      throw new IllegalArgumentException("tools must be a nonempty array");
    }
    List<ToolSpec> result = new ArrayList<>();
    for (JsonNode item : value) {
      JsonNode function = item.path("function");
      JsonNode parameters = function.path("parameters");
      if (!parameters.isObject()) {
        throw new IllegalArgumentException("tool parameters must be a JSON object");
      }
      try {
        result.add(
            new ToolSpec(
                requiredText(function, "name"),
                function.path("description").asText(""),
                mapper.writeValueAsString(parameters)));
      } catch (com.fasterxml.jackson.core.JsonProcessingException failure) {
        throw new IllegalArgumentException("cannot serialize tool schema", failure);
      }
    }
    return List.copyOf(result);
  }

  private static void requireCommonHashes(Configuration configuration) throws IOException {
    requireHash(configuration.model(), ActivatedHiddenDecisionProfileCli.MODEL_SHA256, "model");
    requireHash(
        configuration.adapter().resolve("adapter_model.safetensors"),
        ActivatedHiddenDecisionProfileCli.ADAPTER_SHA256,
        "adapter");
    requireHash(configuration.head(), ActivatedHiddenDecisionProfileCli.HEAD_FILE_SHA256, "head");
  }

  private static Configuration parse(String[] args, Set<String> options) {
    return parse(BenchmarkCliArguments.parse(args, options));
  }

  private static Configuration parse(Map<String, String> values) {
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact Git SHA");
    }
    return new Configuration(
        requiredFile(values, "model"),
        requiredDirectory(values, "adapter"),
        requiredFile(values, "head"),
        requiredFile(values, "records"),
        Path.of(required(values, "output")),
        revision);
  }

  private static void write(ObjectMapper mapper, Path output, Object report) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    mapper.writeValue(output.toFile(), report);
  }

  private static void printResult(
      String verdict,
      double threshold,
      ActivatedQ4CalibrationCli.Score score,
      long physicallyShared,
      long observations,
      Path output) {
    System.out.printf(
        "%s threshold=%.8f balanced=%.6f calls=%d/%d no-calls=%d/%d physical=%d/%d "
            + "report=%s%n",
        verdict,
        threshold,
        score.balancedAccuracy(),
        score.correctCalls(),
        score.calls(),
        score.correctNoCalls(),
        score.noCalls(),
        physicallyShared,
        observations,
        output.toAbsolutePath());
  }

  private static long physicalCount(List<ActivatedQ4CalibrationCli.Observation> observations) {
    return observations.stream()
        .filter(ActivatedQ4CalibrationCli.Observation::physicallyShared)
        .count();
  }

  private static void requireHash(Path path, String expected, String name) throws IOException {
    String actual = Hashing.sha256(path);
    if (!expected.equals(actual)) {
      throw new IllegalArgumentException("V18 " + name + " SHA-256 changed: " + actual);
    }
  }

  private static Path requiredFile(Map<String, String> values, String name) {
    Path path = Path.of(required(values, name));
    if (!Files.isRegularFile(path)) {
      throw new IllegalArgumentException("--" + name + " is not a regular file: " + path);
    }
    return path;
  }

  private static Path requiredDirectory(Map<String, String> values, String name) {
    Path path = Path.of(required(values, name));
    if (!Files.isDirectory(path)) {
      throw new IllegalArgumentException("--" + name + " is not a directory: " + path);
    }
    return path;
  }

  private static String requiredSha256(Map<String, String> values, String name) {
    String value = required(values, name);
    if (!value.matches("[0-9a-f]{64}")) {
      throw new IllegalArgumentException("--" + name + " must be a lowercase SHA-256");
    }
    return value;
  }

  private static String requiredText(JsonNode item, String field) {
    String value = item.path(field).asText();
    if (value.isBlank()) {
      throw new IllegalArgumentException(field + " must not be blank");
    }
    return value;
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }

  private record ScoredRun(
      ActivatedAdapterMetadata adapter, List<ActivatedQ4CalibrationCli.Observation> observations) {}

  private record CalibrationInput(double threshold, ActivatedQ4CalibrationCli.Score score) {}
}
