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
import com.fasterxml.jackson.databind.SerializationFeature;
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.ToolDecisionScore;
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
import java.util.Objects;
import java.util.Set;

/** Profiles and freezes the V15 activated-adapter call/no-call threshold in the JVM runtime. */
final class ActivatedDecisionProfileCli {
  static final String EXPERIMENT = "qwen3-17b-shared-decision-v15";
  static final String SOURCE_SHA256 =
      "944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c";
  static final int CALL_TOKEN_ID = 4913;
  static final int NO_CALL_TOKEN_ID = 19536;
  private static final String PARTITION_SEED = EXPERIMENT + ":partition-v1";
  private static final Set<String> OPTIONS =
      Set.of("model", "adapter", "records", "output", "models-revision");
  private static final Set<String> KINDS = Set.of("simple", "multiple", "irrelevance");

  private ActivatedDecisionProfileCli() {}

  record Configuration(
      Path model, Path adapter, Path records, Path output, String modelsRevision) {}

  record SourceCase(
      String id, String kind, String partition, boolean callExpected, ModelPrompt prompt) {}

  record Observation(
      String id,
      String kind,
      String partition,
      boolean callExpected,
      int promptTokens,
      int sharedPrefixTokens,
      boolean physicallyShared,
      float callLogit,
      float noCallLogit,
      float margin,
      long elapsedMillis) {}

  record Score(
      int calls,
      int noCalls,
      int correctCalls,
      int correctNoCalls,
      double callAccuracy,
      double noCallAccuracy,
      double balancedAccuracy) {}

  record Calibration(float threshold, Score calibration, Score screen) {}

  record SplitContract(
      String seed,
      int calibrationPerKind,
      Map<String, List<String>> calibrationIds,
      Map<String, List<String>> screenIds) {}

  record Report(
      int schemaVersion,
      String experiment,
      String createdAt,
      String modelsRevision,
      String sourceSha256,
      String modelSha256,
      ActivatedAdapterMetadata adapter,
      BenchmarkEnvironment environment,
      SplitContract split,
      int callTokenId,
      int noCallTokenId,
      Calibration result,
      boolean screenPassed,
      String verdict,
      List<Observation> observations) {}

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path model = requiredFile(values, "model");
    Path adapter = requiredDirectory(values, "adapter");
    Path records = requiredFile(values, "records");
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    Path output =
        Path.of(
            values.getOrDefault(
                "output", "build/reports/tool-qualification/activated-decision-v15.json"));
    return new Configuration(model, adapter, records, output, revision);
  }

  static int run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    String sourceSha256 = Hashing.sha256(configuration.records());
    if (!SOURCE_SHA256.equals(sourceSha256)) {
      throw new IllegalArgumentException(
          "V15 requires the frozen V9 exposed records: " + sourceSha256);
    }

    ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    LoadedCases loaded = loadCases(mapper, configuration.records());
    List<Observation> observations = new ArrayList<>();
    ActivatedAdapterMetadata adapter;
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
    try (ActivatedToolCallingModel model = activatedModel) {
      adapter = model.adapter();
      int ordinal = 0;
      for (SourceCase item : loaded.cases()) {
        ordinal++;
        long started = System.nanoTime();
        try (ActivatedToolTurn turn =
            model.openToolTurn(item.prompt(), ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          ToolDecisionScore score = turn.scoreToolDecision(CALL_TOKEN_ID, NO_CALL_TOKEN_ID);
          long elapsedMillis = (System.nanoTime() - started) / 1_000_000L;
          Observation observation =
              new Observation(
                  item.id(),
                  item.kind(),
                  item.partition(),
                  item.callExpected(),
                  backend.tokenizer().encode(item.prompt()).length,
                  turn.sharedPrefixTokens(),
                  turn.physicallySharesPrefix(),
                  score.callLogit(),
                  score.noCallLogit(),
                  score.callMargin(),
                  elapsedMillis);
          observations.add(observation);
          System.out.printf(
              "%2d/%d %-18s %-11s expected=%-5s margin=%10.4f shared=%s %d ms%n",
              ordinal,
              loaded.cases().size(),
              item.id(),
              item.partition(),
              item.callExpected(),
              score.callMargin(),
              observation.physicallyShared(),
              elapsedMillis);
        }
      }
    }

    if (observations.stream().anyMatch(item -> !item.physicallyShared())) {
      throw new IllegalStateException("a V15 observation did not physically share its KV prefix");
    }
    Calibration calibration = calibrate(observations);
    boolean screenPassed =
        calibration.screen().correctCalls() >= 28
            && calibration.screen().correctNoCalls() >= 14
            && calibration.screen().balancedAccuracy() > 0.93;
    Report report =
        new Report(
            1,
            EXPERIMENT,
            Instant.now().toString(),
            configuration.modelsRevision(),
            sourceSha256,
            Hashing.sha256(configuration.model()),
            adapter,
            BenchmarkEnvironment.capture(),
            loaded.split(),
            CALL_TOKEN_ID,
            NO_CALL_TOKEN_ID,
            calibration,
            screenPassed,
            screenPassed ? "PASS" : "FAIL",
            List.copyOf(observations));
    Path parent = configuration.output().toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    mapper.writeValue(configuration.output().toFile(), report);
    System.out.printf(
        "%s threshold=%.4f calibration=%.4f screen=%.4f calls=%d/%d no-calls=%d/%d report=%s%n",
        report.verdict(),
        calibration.threshold(),
        calibration.calibration().balancedAccuracy(),
        calibration.screen().balancedAccuracy(),
        calibration.screen().correctCalls(),
        calibration.screen().calls(),
        calibration.screen().correctNoCalls(),
        calibration.screen().noCalls(),
        configuration.output().toAbsolutePath());
    return screenPassed ? 0 : 1;
  }

  static Calibration calibrate(List<Observation> observations) {
    List<Observation> calibration =
        observations.stream().filter(item -> item.partition().equals("calibration")).toList();
    List<Observation> screen =
        observations.stream().filter(item -> item.partition().equals("screen")).toList();
    requireBothClasses(calibration, "calibration");
    requireBothClasses(screen, "screen");
    List<Float> margins =
        calibration.stream().map(Observation::margin).distinct().sorted().toList();
    List<Float> thresholds = new ArrayList<>();
    thresholds.add(Math.nextDown(margins.getFirst()));
    for (int index = 1; index < margins.size(); index++) {
      float lower = margins.get(index - 1);
      float upper = margins.get(index);
      thresholds.add(lower / 2.0f + upper / 2.0f);
    }
    thresholds.add(margins.getLast());

    float selected = thresholds.getFirst();
    Score selectedScore = score(calibration, selected);
    for (float threshold : thresholds.subList(1, thresholds.size())) {
      Score candidate = score(calibration, threshold);
      if (better(candidate, threshold, selectedScore, selected)) {
        selected = threshold;
        selectedScore = candidate;
      }
    }
    return new Calibration(selected, selectedScore, score(screen, selected));
  }

  private static boolean better(
      Score candidate, float threshold, Score selected, float selectedThreshold) {
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

  private static Score score(List<Observation> observations, float threshold) {
    int calls = Math.toIntExact(observations.stream().filter(Observation::callExpected).count());
    int noCalls = observations.size() - calls;
    int correctCalls =
        Math.toIntExact(
            observations.stream()
                .filter(Observation::callExpected)
                .filter(item -> item.margin() > threshold)
                .count());
    int correctNoCalls =
        Math.toIntExact(
            observations.stream()
                .filter(item -> !item.callExpected())
                .filter(item -> item.margin() <= threshold)
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

  private static void requireBothClasses(List<Observation> observations, String partition) {
    if (observations.isEmpty()
        || observations.stream().noneMatch(Observation::callExpected)
        || observations.stream().allMatch(Observation::callExpected)) {
      throw new IllegalArgumentException(partition + " partition must contain both classes");
    }
  }

  private static LoadedCases loadCases(ObjectMapper mapper, Path path) throws IOException {
    Map<String, JsonNode> unique = new LinkedHashMap<>();
    try (var lines = Files.lines(path)) {
      for (String line : lines.toList()) {
        JsonNode item = mapper.readTree(line);
        if (!"adapter".equals(item.path("mode").asText())) {
          continue;
        }
        String id = requiredText(item, "id");
        if (unique.put(id, item) != null) {
          throw new IllegalArgumentException("duplicate adapter record: " + id);
        }
      }
    }
    if (unique.size() != 75) {
      throw new IllegalArgumentException("V15 requires exactly 75 unique adapter records");
    }

    Map<String, List<JsonNode>> byKind = new HashMap<>();
    for (JsonNode item : unique.values()) {
      String kind = requiredText(item, "kind");
      if (!KINDS.contains(kind)) {
        throw new IllegalArgumentException("unexpected V15 case kind: " + kind);
      }
      byKind.computeIfAbsent(kind, ignored -> new ArrayList<>()).add(item);
    }
    Map<String, List<String>> calibrationIds = new LinkedHashMap<>();
    Map<String, List<String>> screenIds = new LinkedHashMap<>();
    Set<String> calibration = new java.util.HashSet<>();
    for (String kind : List.of("simple", "multiple", "irrelevance")) {
      List<JsonNode> items = byKind.getOrDefault(kind, List.of());
      if (items.size() != 25) {
        throw new IllegalArgumentException("V15 requires exactly 25 " + kind + " records");
      }
      List<String> ids =
          items.stream()
              .map(item -> requiredText(item, "id"))
              .sorted(
                  Comparator.comparing(
                          (String id) -> Hashing.sha256(PARTITION_SEED + "\0" + kind + "\0" + id))
                      .thenComparing(Comparator.naturalOrder()))
              .toList();
      List<String> selected = List.copyOf(ids.subList(0, 10));
      List<String> heldOut = List.copyOf(ids.subList(10, ids.size()));
      calibration.addAll(selected);
      calibrationIds.put(kind, selected);
      screenIds.put(kind, heldOut);
    }

    List<SourceCase> cases = new ArrayList<>();
    for (JsonNode item : unique.values()) {
      String id = requiredText(item, "id");
      String kind = requiredText(item, "kind");
      List<ChatMessage> messages = messages(item.path("messages"));
      List<ToolSpec> tools = tools(mapper, item.path("tools"));
      ModelPrompt prompt = ChatTemplate.CHATML_NO_THINK.render(messages, tools);
      String frozenPrompt = requiredText(item, "prompt");
      if (!prompt.text().equals(frozenPrompt)) {
        throw new IllegalArgumentException(
            "Java prompt differs from frozen V9 prompt for "
                + id
                + ": "
                + firstDifference(frozenPrompt, prompt.text()));
      }
      JsonNode expected = item.path("expected");
      if (!expected.isArray()) {
        throw new IllegalArgumentException("expected calls must be an array for " + id);
      }
      cases.add(
          new SourceCase(
              id,
              kind,
              calibration.contains(id) ? "calibration" : "screen",
              !expected.isEmpty(),
              prompt));
    }
    cases.sort(Comparator.comparing(SourceCase::partition).thenComparing(SourceCase::id));
    return new LoadedCases(
        List.copyOf(cases), new SplitContract(PARTITION_SEED, 10, calibrationIds, screenIds));
  }

  private static List<ChatMessage> messages(JsonNode value) {
    if (!value.isArray() || value.isEmpty()) {
      throw new IllegalArgumentException("messages must be a nonempty array");
    }
    List<ChatMessage> messages = new ArrayList<>();
    for (JsonNode item : value) {
      String role = requiredText(item, "role");
      String content = requiredText(item, "content");
      messages.add(
          switch (role) {
            case "system" -> ChatMessage.system(content);
            case "user" -> ChatMessage.user(content);
            default ->
                throw new IllegalArgumentException("unsupported source message role: " + role);
          });
    }
    return List.copyOf(messages);
  }

  private static List<ToolSpec> tools(ObjectMapper mapper, JsonNode value) {
    if (!value.isArray() || value.isEmpty()) {
      throw new IllegalArgumentException("tools must be a nonempty array");
    }
    List<ToolSpec> tools = new ArrayList<>();
    for (JsonNode item : value) {
      JsonNode function = item.path("function");
      JsonNode parameters = function.path("parameters");
      if (!parameters.isObject()) {
        throw new IllegalArgumentException("tool parameters must be a JSON object");
      }
      try {
        tools.add(
            new ToolSpec(
                requiredText(function, "name"),
                function.path("description").asText(""),
                mapper.writeValueAsString(parameters)));
      } catch (com.fasterxml.jackson.core.JsonProcessingException failure) {
        throw new IllegalArgumentException("cannot serialize tool schema", failure);
      }
    }
    return List.copyOf(tools);
  }

  private static String requiredText(JsonNode item, String field) {
    String value = item.path(field).asText();
    if (value.isBlank()) {
      throw new IllegalArgumentException(field + " must not be blank");
    }
    return value;
  }

  private static String firstDifference(String expected, String actual) {
    int limit = Math.min(expected.length(), actual.length());
    int index = 0;
    while (index < limit && expected.charAt(index) == actual.charAt(index)) {
      index++;
    }
    int from = Math.max(0, index - 40);
    int expectedTo = Math.min(expected.length(), index + 80);
    int actualTo = Math.min(actual.length(), index + 80);
    return "index="
        + index
        + " expectedLength="
        + expected.length()
        + " actualLength="
        + actual.length()
        + " expected='"
        + visible(expected.substring(from, expectedTo))
        + "' actual='"
        + visible(actual.substring(from, actualTo))
        + "'";
  }

  private static String visible(String value) {
    return value.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n");
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

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }

  private record LoadedCases(List<SourceCase> cases, SplitContract split) {
    private LoadedCases {
      cases = List.copyOf(Objects.requireNonNull(cases, "cases"));
      Objects.requireNonNull(split, "split");
    }
  }
}
