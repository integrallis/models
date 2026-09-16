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
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.ToolApplicabilityHead;
import com.integrallis.models.runtime.ToolApplicabilityScore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Calibrates and screens the frozen V16 head against production Java Q4 hidden states. */
final class ActivatedQ4CalibrationCli {
  static final String EXPERIMENT = "qwen3-17b-q4-calibration-v17";
  static final String PARTITION_SEED = "qwen3-17b-shared-decision-v15:partition-v1";
  private static final Set<String> CALIBRATION_OPTIONS =
      Set.of("model", "adapter", "head", "records", "output", "models-revision");
  private static final Set<String> SCREEN_OPTIONS =
      Set.of(
          "model",
          "adapter",
          "head",
          "records",
          "calibration",
          "calibration-sha256",
          "output",
          "models-revision");

  private ActivatedQ4CalibrationCli() {}

  record Configuration(
      Path model, Path adapter, Path head, Path records, Path output, String modelsRevision) {}

  record Observation(
      String id,
      String kind,
      String partition,
      boolean callExpected,
      int promptTokens,
      int sharedPrefixTokens,
      long sharedPrefixBytes,
      boolean physicallyShared,
      double score,
      boolean callPredicted,
      long elapsedMillis) {}

  record Score(
      int calls,
      int noCalls,
      int correctCalls,
      int correctNoCalls,
      double callAccuracy,
      double noCallAccuracy,
      double balancedAccuracy) {}

  record ThresholdSelection(double threshold, Score score) {}

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
      ActivatedDecisionProfileCli.SplitContract split,
      double threshold,
      Score result,
      boolean passed,
      String verdict,
      List<Observation> observations) {}

  record ScreenReport(
      int schemaVersion,
      String experiment,
      String phase,
      String createdAt,
      String modelsRevision,
      String sourceSha256,
      String modelSha256,
      String headFileSha256,
      String headArtifactSha256,
      String calibrationFileSha256,
      ActivatedAdapterMetadata adapter,
      BenchmarkEnvironment environment,
      ActivatedDecisionProfileCli.SplitContract split,
      double threshold,
      Score calibration,
      Score screen,
      Score combined,
      boolean passed,
      String verdict,
      List<Observation> observations) {}

  static int runCalibration(String[] args) throws IOException {
    Configuration configuration = parse(args, CALIBRATION_OPTIONS);
    ObjectMapper mapper = ActivatedDecisionProfileCli.mapper();
    Inputs inputs = loadInputs(configuration, mapper);
    ScoredRun run = scoreCases(configuration, inputs, "calibration", Double.NaN);
    ThresholdSelection selection = selectThreshold(run.observations());
    List<Observation> observations = predict(run.observations(), selection.threshold());
    long physicallyShared = observations.stream().filter(Observation::physicallyShared).count();
    boolean passed = calibrationPassed(selection.score(), observations.size(), physicallyShared);
    CalibrationReport report =
        new CalibrationReport(
            1,
            EXPERIMENT,
            "calibration",
            Instant.now().toString(),
            configuration.modelsRevision(),
            ActivatedHiddenDecisionProfileCli.SOURCE_SHA256,
            ActivatedHiddenDecisionProfileCli.MODEL_SHA256,
            ActivatedHiddenDecisionProfileCli.HEAD_FILE_SHA256,
            ActivatedHiddenDecisionProfileCli.HEAD_ARTIFACT_SHA256,
            run.adapter(),
            BenchmarkEnvironment.capture(),
            inputs.loadedCases().split(),
            selection.threshold(),
            selection.score(),
            passed,
            passed ? "PASS" : "FAIL",
            observations);
    write(mapper, configuration.output(), report);
    System.out.printf(
        "%s threshold=%.8f balanced=%.6f calls=%d/%d no-calls=%d/%d physical=%d/%d report=%s%n",
        report.verdict(),
        report.threshold(),
        report.result().balancedAccuracy(),
        report.result().correctCalls(),
        report.result().calls(),
        report.result().correctNoCalls(),
        report.result().noCalls(),
        physicallyShared,
        observations.size(),
        configuration.output().toAbsolutePath());
    return passed ? 0 : 1;
  }

  static int runScreen(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, SCREEN_OPTIONS);
    Configuration configuration = parse(values);
    Path calibrationPath = requiredFile(values, "calibration");
    String calibrationSha256 = requiredSha256(values, "calibration-sha256");
    requireHash(calibrationPath, calibrationSha256, "calibration artifact");

    ObjectMapper mapper = ActivatedDecisionProfileCli.mapper();
    Inputs inputs = loadInputs(configuration, mapper);
    CalibrationInput calibration =
        loadCalibration(mapper, calibrationPath, calibrationSha256, configuration, inputs);
    ScoredRun run = scoreCases(configuration, inputs, "screen", calibration.threshold());
    List<Observation> observations = predict(run.observations(), calibration.threshold());
    Score screen = score(observations, calibration.threshold(), "screen");
    Score combined = combine(calibration.score(), screen);
    long physicallyShared = observations.stream().filter(Observation::physicallyShared).count();
    boolean passed =
        screenPassed(calibration.score(), screen, observations.size(), physicallyShared);
    ScreenReport report =
        new ScreenReport(
            1,
            EXPERIMENT,
            "screen",
            Instant.now().toString(),
            configuration.modelsRevision(),
            ActivatedHiddenDecisionProfileCli.SOURCE_SHA256,
            ActivatedHiddenDecisionProfileCli.MODEL_SHA256,
            ActivatedHiddenDecisionProfileCli.HEAD_FILE_SHA256,
            ActivatedHiddenDecisionProfileCli.HEAD_ARTIFACT_SHA256,
            calibrationSha256,
            run.adapter(),
            BenchmarkEnvironment.capture(),
            inputs.loadedCases().split(),
            calibration.threshold(),
            calibration.score(),
            screen,
            combined,
            passed,
            passed ? "PASS" : "FAIL",
            observations);
    write(mapper, configuration.output(), report);
    System.out.printf(
        "%s threshold=%.8f screen=%.6f screen-calls=%d/%d screen-no-calls=%d/%d "
            + "combined-calls=%d/%d combined-no-calls=%d/%d physical=%d/%d report=%s%n",
        report.verdict(),
        report.threshold(),
        report.screen().balancedAccuracy(),
        report.screen().correctCalls(),
        report.screen().calls(),
        report.screen().correctNoCalls(),
        report.screen().noCalls(),
        report.combined().correctCalls(),
        report.combined().calls(),
        report.combined().correctNoCalls(),
        report.combined().noCalls(),
        physicallyShared,
        observations.size(),
        configuration.output().toAbsolutePath());
    return passed ? 0 : 1;
  }

  static ThresholdSelection selectThreshold(List<Observation> observations) {
    requirePartition(observations, "calibration");
    List<Double> scores =
        observations.stream().map(Observation::score).distinct().sorted().toList();
    if (scores.isEmpty()) {
      throw new IllegalArgumentException("calibration scores must not be empty");
    }
    List<Double> candidates = new ArrayList<>();
    candidates.add(Math.nextDown(scores.getFirst()));
    for (int index = 1; index < scores.size(); index++) {
      double lower = scores.get(index - 1);
      double upper = scores.get(index);
      candidates.add(lower / 2.0 + upper / 2.0);
    }
    candidates.add(scores.getLast());

    double selected = candidates.getFirst();
    Score selectedScore = score(observations, selected, "calibration");
    for (double candidate : candidates.subList(1, candidates.size())) {
      Score candidateScore = score(observations, candidate, "calibration");
      if (better(candidateScore, candidate, selectedScore, selected)) {
        selected = candidate;
        selectedScore = candidateScore;
      }
    }
    return new ThresholdSelection(selected, selectedScore);
  }

  static Score score(List<Observation> observations, double threshold, String partition) {
    requirePartition(observations, partition);
    int calls = Math.toIntExact(observations.stream().filter(Observation::callExpected).count());
    int noCalls = observations.size() - calls;
    if (calls == 0 || noCalls == 0) {
      throw new IllegalArgumentException(partition + " partition must contain both classes");
    }
    int correctCalls =
        Math.toIntExact(
            observations.stream()
                .filter(Observation::callExpected)
                .filter(item -> item.score() > threshold)
                .count());
    int correctNoCalls =
        Math.toIntExact(
            observations.stream()
                .filter(item -> !item.callExpected())
                .filter(item -> item.score() <= threshold)
                .count());
    double callAccuracy = (double) correctCalls / calls;
    double noCallAccuracy = (double) correctNoCalls / noCalls;
    return new Score(
        calls,
        noCalls,
        correctCalls,
        correctNoCalls,
        callAccuracy,
        noCallAccuracy,
        (callAccuracy + noCallAccuracy) / 2.0);
  }

  static boolean calibrationPassed(Score score, long observations, long physicallyShared) {
    return observations == 30
        && physicallyShared == 30
        && score.calls() == 20
        && score.noCalls() == 10
        && score.correctCalls() >= 19
        && score.correctNoCalls() >= 9
        && score.balancedAccuracy() >= 0.925;
  }

  static boolean screenPassed(
      Score calibration, Score screen, long observations, long physicallyShared) {
    Score combined = combine(calibration, screen);
    return observations == 45
        && physicallyShared == 45
        && screen.calls() == 30
        && screen.noCalls() == 15
        && screen.correctCalls() >= 28
        && screen.correctNoCalls() >= 14
        && screen.balancedAccuracy() > 0.93
        && combined.calls() == 50
        && combined.noCalls() == 25
        && combined.correctCalls() >= 47
        && combined.correctNoCalls() >= 24;
  }

  private static Inputs loadInputs(Configuration configuration, ObjectMapper mapper)
      throws IOException {
    requireHash(configuration.model(), ActivatedHiddenDecisionProfileCli.MODEL_SHA256, "model");
    requireHash(
        configuration.adapter().resolve("adapter_model.safetensors"),
        ActivatedHiddenDecisionProfileCli.ADAPTER_SHA256,
        "adapter");
    requireHash(configuration.head(), ActivatedHiddenDecisionProfileCli.HEAD_FILE_SHA256, "head");
    requireHash(
        configuration.records(), ActivatedHiddenDecisionProfileCli.SOURCE_SHA256, "records");
    ToolApplicabilityHead head =
        ActivatedHiddenDecisionProfileCli.loadHead(mapper, configuration.head());
    ActivatedDecisionProfileCli.LoadedCases loadedCases =
        ActivatedDecisionProfileCli.loadCases(mapper, configuration.records());
    if (!PARTITION_SEED.equals(loadedCases.split().seed())
        || loadedCases.split().calibrationPerKind() != 10) {
      throw new IllegalArgumentException("V17 partition contract changed");
    }
    return new Inputs(head, loadedCases);
  }

  private static ScoredRun scoreCases(
      Configuration configuration, Inputs inputs, String partition, double threshold) {
    List<ActivatedDecisionProfileCli.SourceCase> selected =
        inputs.loadedCases().cases().stream()
            .filter(item -> partition.equals(item.partition()))
            .toList();
    int expectedCount = "calibration".equals(partition) ? 30 : 45;
    if (selected.size() != expectedCount) {
      throw new IllegalArgumentException(
          partition + " partition must contain exactly " + expectedCount + " cases");
    }

    List<Observation> observations = new ArrayList<>();
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
      for (ActivatedDecisionProfileCli.SourceCase item : selected) {
        ordinal++;
        long started = System.nanoTime();
        try (ActivatedToolTurn turn =
            activated.openToolTurn(
                item.prompt(), ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          ToolApplicabilityScore applicability = turn.scoreToolApplicability(inputs.head());
          long elapsedMillis = (System.nanoTime() - started) / 1_000_000L;
          boolean predicted = Double.isFinite(threshold) && applicability.score() > threshold;
          Observation observation =
              new Observation(
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
              "%2d/%d %-18s expected=%-5s score=%11.5f shared=%s %d ms%n",
              ordinal,
              selected.size(),
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
      ObjectMapper mapper, Path path, String fileSha256, Configuration configuration, Inputs inputs)
      throws IOException {
    JsonNode value = mapper.readTree(path.toFile());
    if (value.path("schemaVersion").asInt() != 1
        || !EXPERIMENT.equals(value.path("experiment").asText())
        || !"calibration".equals(value.path("phase").asText())
        || !value.path("passed").asBoolean()
        || !"PASS".equals(value.path("verdict").asText())
        || !configuration.modelsRevision().equals(value.path("modelsRevision").asText())
        || !ActivatedHiddenDecisionProfileCli.SOURCE_SHA256.equals(
            value.path("sourceSha256").asText())
        || !ActivatedHiddenDecisionProfileCli.MODEL_SHA256.equals(
            value.path("modelSha256").asText())
        || !ActivatedHiddenDecisionProfileCli.HEAD_FILE_SHA256.equals(
            value.path("headFileSha256").asText())
        || !ActivatedHiddenDecisionProfileCli.HEAD_ARTIFACT_SHA256.equals(
            value.path("headArtifactSha256").asText())
        || !ActivatedHiddenDecisionProfileCli.ADAPTER_SHA256.equals(
            value.path("adapter").path("adapterSha256").asText())
        || !PARTITION_SEED.equals(value.path("split").path("seed").asText())
        || value.path("split").path("calibrationPerKind").asInt() != 10) {
      throw new IllegalArgumentException(
          "V17 calibration artifact identity or passing verdict changed: " + fileSha256);
    }
    JsonNode observations = value.path("observations");
    if (!observations.isArray()
        || observations.size() != 30
        || java.util.stream.StreamSupport.stream(observations.spliterator(), false)
            .anyMatch(
                item ->
                    !"calibration".equals(item.path("partition").asText())
                        || !item.path("physicallyShared").asBoolean())) {
      throw new IllegalArgumentException("V17 calibration observations changed");
    }
    double threshold = value.path("threshold").asDouble(Double.NaN);
    if (!Double.isFinite(threshold)) {
      throw new IllegalArgumentException("V17 calibration threshold must be finite");
    }
    Map<String, ActivatedDecisionProfileCli.SourceCase> expectedCases =
        inputs.loadedCases().cases().stream()
            .filter(item -> "calibration".equals(item.partition()))
            .collect(
                java.util.stream.Collectors.toUnmodifiableMap(
                    ActivatedDecisionProfileCli.SourceCase::id, item -> item));
    List<Observation> parsed = new ArrayList<>();
    for (JsonNode item : observations) {
      String id = item.path("id").asText();
      ActivatedDecisionProfileCli.SourceCase expected = expectedCases.get(id);
      double itemScore = item.path("score").asDouble(Double.NaN);
      if (expected == null
          || parsed.stream().anyMatch(existing -> existing.id().equals(id))
          || !expected.kind().equals(item.path("kind").asText())
          || expected.callExpected() != item.path("callExpected").asBoolean()
          || !Double.isFinite(itemScore)
          || item.path("callPredicted").asBoolean() != (itemScore > threshold)) {
        throw new IllegalArgumentException("V17 calibration observation changed: " + id);
      }
      parsed.add(
          new Observation(
              id,
              expected.kind(),
              "calibration",
              expected.callExpected(),
              item.path("promptTokens").asInt(),
              item.path("sharedPrefixTokens").asInt(),
              item.path("sharedPrefixBytes").asLong(),
              true,
              itemScore,
              item.path("callPredicted").asBoolean(),
              item.path("elapsedMillis").asLong()));
    }
    if (parsed.size() != expectedCases.size()) {
      throw new IllegalArgumentException("V17 calibration case identity changed");
    }
    ThresholdSelection recomputed = selectThreshold(parsed);
    if (Double.doubleToLongBits(recomputed.threshold()) != Double.doubleToLongBits(threshold)) {
      throw new IllegalArgumentException("V17 calibration threshold is not reproducible");
    }
    Score calibration = readScore(value.path("result"));
    if (!calibration.equals(recomputed.score())) {
      throw new IllegalArgumentException("V17 calibration score is not reproducible");
    }
    if (!calibrationPassed(calibration, observations.size(), observations.size())) {
      throw new IllegalArgumentException("V17 calibration result no longer passes");
    }
    if (!inputs.loadedCases().split().seed().equals(value.path("split").path("seed").asText())) {
      throw new IllegalArgumentException("V17 loaded split differs from calibration artifact");
    }
    return new CalibrationInput(threshold, calibration);
  }

  private static Score readScore(JsonNode value) {
    Score score =
        new Score(
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
      throw new IllegalArgumentException("V17 calibration score is invalid");
    }
    return score;
  }

  private static Score combine(Score first, Score second) {
    int calls = first.calls() + second.calls();
    int noCalls = first.noCalls() + second.noCalls();
    int correctCalls = first.correctCalls() + second.correctCalls();
    int correctNoCalls = first.correctNoCalls() + second.correctNoCalls();
    double callAccuracy = (double) correctCalls / calls;
    double noCallAccuracy = (double) correctNoCalls / noCalls;
    return new Score(
        calls,
        noCalls,
        correctCalls,
        correctNoCalls,
        callAccuracy,
        noCallAccuracy,
        (callAccuracy + noCallAccuracy) / 2.0);
  }

  private static boolean better(
      Score candidate, double threshold, Score selected, double selectedThreshold) {
    int balanced = Double.compare(candidate.balancedAccuracy(), selected.balancedAccuracy());
    if (balanced != 0) {
      return balanced > 0;
    }
    if (candidate.correctNoCalls() != selected.correctNoCalls()) {
      return candidate.correctNoCalls() > selected.correctNoCalls();
    }
    if (candidate.correctCalls() != selected.correctCalls()) {
      return candidate.correctCalls() > selected.correctCalls();
    }
    return threshold > selectedThreshold;
  }

  private static List<Observation> predict(List<Observation> observations, double threshold) {
    return observations.stream()
        .map(
            item ->
                new Observation(
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

  private static void requirePartition(List<Observation> observations, String partition) {
    if (observations.isEmpty()
        || observations.stream().anyMatch(item -> !partition.equals(item.partition()))) {
      throw new IllegalArgumentException(
          "observations must contain only the " + partition + " partition");
    }
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

  private static void requireHash(Path path, String expected, String name) throws IOException {
    String actual = Hashing.sha256(path);
    if (!expected.equals(actual)) {
      throw new IllegalArgumentException("V17 " + name + " SHA-256 changed: " + actual);
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

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }

  private record Inputs(
      ToolApplicabilityHead head, ActivatedDecisionProfileCli.LoadedCases loadedCases) {}

  private record ScoredRun(ActivatedAdapterMetadata adapter, List<Observation> observations) {}

  private record CalibrationInput(double threshold, Score score) {}
}
