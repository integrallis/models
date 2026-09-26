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
import com.integrallis.models.router.*;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;

/**
 * Paired offline selection evaluation from complete, pinned provider observations. This measures
 * selection against actual reference answers, not prompt classification accuracy. It never calls a
 * provider, and replay latency is explicitly distinguished from live router latency.
 */
public final class RouterSelectionEvaluationCli {
  private static final ObjectMapper JSON = new ObjectMapper();

  private RouterSelectionEvaluationCli() {}

  public static void run(String[] args) throws Exception {
    if (args.length != 2)
      throw new IllegalArgumentException("router-evaluate MANIFEST.json REPORT.json");
    Path manifest = Path.of(args[0]).toAbsolutePath();
    Map<String, Object> report = evaluate(manifest);
    Path output = Path.of(args[1]);
    if (Files.exists(output))
      throw new IllegalArgumentException("refusing to overwrite evaluation evidence: " + output);
    JSON.writerWithDefaultPrettyPrinter().writeValue(output.toFile(), report);
  }

  static Map<String, Object> evaluate(Path manifestPath) throws Exception {
    JsonNode manifest = JSON.readTree(manifestPath.toFile());
    require(manifest.path("schemaVersion").asInt() == 1, "schemaVersion must be 1");
    for (String field :
        List.of(
            "protocolRevision",
            "modelsRevision",
            "classifierRevision",
            "hardware",
            "runtime",
            "generationSettings")) {
      require(!manifest.path(field).asText().isBlank(), "missing pinned " + field);
    }
    JsonNode cases = readPinned(manifestPath, manifest.path("cases"));
    JsonNode observations = readPinned(manifestPath, manifest.path("observations"));
    require(
        cases.isArray() && observations.isArray(), "cases and observations must be JSON arrays");
    Map<String, JsonNode> datasets = new LinkedHashMap<>();
    for (JsonNode dataset : manifest.path("datasets")) {
      String id = text(dataset, "id");
      for (String field : List.of("source", "revision", "license")) text(dataset, field);
      require(datasets.put(id, dataset) == null, "duplicate dataset: " + id);
    }
    require(datasets.size() >= 2, "at least two public datasets are required");
    Map<String, JsonNode> caseById = new LinkedHashMap<>();
    Set<String> prompts = new HashSet<>();
    Map<String, Set<String>> splits = new HashMap<>();
    for (JsonNode item : cases) {
      String id = text(item, "id"), dataset = text(item, "dataset"), split = text(item, "split");
      require(datasets.containsKey(dataset), "undeclared dataset: " + dataset);
      require(Set.of("calibration", "eval").contains(split), "unknown split");
      require(caseById.put(id, item) == null, "duplicate case: " + id);
      require(
          prompts.add(text(item, "prompt").strip()),
          "duplicate prompt or calibration/evaluation leakage");
      require(
          item.path("answers").isArray() && !item.path("answers").isEmpty(),
          "reference answers required");
      text(item, "task");
      require(
          item.has("predictedTask")
              && (item.path("predictedTask").isNull()
                  || (item.path("predictedTask").isTextual()
                      && !item.path("predictedTask").asText().isBlank())),
          "predictedTask must be explicit text or null for abstention");
      nonNegative(item, "classifierMillis");
      nonNegative(item, "inputBound");
      nonNegative(item, "outputBound");
      require(
          item.path("inputBound").isIntegralNumber()
              && item.path("inputBound").canConvertToInt()
              && item.path("outputBound").isIntegralNumber()
              && item.path("outputBound").canConvertToInt(),
          "token bounds must be integers");
      new RoutingTokenBounds(
          item.path("inputBound").intValue(), item.path("outputBound").intValue());
      require(
          item.path("budget").isNumber() && item.path("budget").decimalValue().signum() >= 0,
          "budget required");
      nonNegative(item, "deadlineMillis");
      require(item.path("deadlineMillis").asDouble() > 0, "positive deadline required");
      splits.computeIfAbsent(dataset, ignored -> new HashSet<>()).add(split);
    }
    for (String dataset : datasets.keySet())
      require(
          Set.of("calibration", "eval").equals(splits.get(dataset)),
          "each dataset needs disjoint calibration and evaluation cases");
    Map<String, JsonNode> descriptors = new LinkedHashMap<>();
    for (JsonNode model : manifest.path("models")) {
      String id = text(model, "id");
      text(model, "revision");
      text(model, "clientRevision");
      nonNegative(model, "inputPrice");
      nonNegative(model, "outputPrice");
      require(model.path("local").isBoolean(), "model locality must be explicit");
      require(
          model.path("contextWindow").isIntegralNumber()
              && model.path("contextWindow").canConvertToInt(),
          "model contextWindow required");
      require(descriptors.put(id, model) == null, "duplicate model");
    }
    require(descriptors.size() >= 2, "at least two models required");
    Map<String, Map<String, JsonNode>> runs = new HashMap<>();
    for (JsonNode run : observations) {
      String caseId = text(run, "caseId"), modelId = text(run, "modelId");
      require(
          caseById.containsKey(caseId) && descriptors.containsKey(modelId),
          "unknown observation key");
      require(
          runs.computeIfAbsent(caseId, ignored -> new HashMap<>()).put(modelId, run) == null,
          "duplicate observation");
      require(run.path("success").isBoolean(), "success must be explicit");
      nonNegative(run, "latencyMillis");
      nonNegative(run, "ttftMillis");
      require(
          run.path("ttftMillis").asDouble() <= run.path("latencyMillis").asDouble(),
          "TTFT exceeds completion time");
      require(
          run.has("response") && run.path("response").isTextual(),
          "raw response required, including on failure");
      text(run, "receipt");
      if (run.hasNonNull("inputTokens") || run.hasNonNull("outputTokens")) {
        nonNegative(run, "inputTokens");
        nonNegative(run, "outputTokens");
        require(
            run.path("inputTokens").isIntegralNumber()
                && run.path("inputTokens").canConvertToLong()
                && run.path("outputTokens").isIntegralNumber()
                && run.path("outputTokens").canConvertToLong(),
            "usage must be integers");
      }
    }
    for (String id : caseById.keySet())
      require(
          runs.containsKey(id) && runs.get(id).keySet().equals(descriptors.keySet()),
          "missing paired model observations for " + id);

    List<ModelCandidate> candidates = new ArrayList<>();
    for (var entry : descriptors.entrySet()) {
      String id = entry.getKey();
      JsonNode model = entry.getValue();
      Map<String, List<Double>> calibration = new TreeMap<>();
      List<Double> ttft = new ArrayList<>();
      for (JsonNode item : cases)
        if (item.path("split").asText().equals("calibration")) {
          JsonNode run = runs.get(item.path("id").asText()).get(id);
          calibration
              .computeIfAbsent(item.path("task").asText(), ignored -> new ArrayList<>())
              .add(score(item, run));
          ttft.add(run.path("ttftMillis").asDouble());
        }
      Map<String, Double> quality = new TreeMap<>();
      calibration.forEach((task, scores) -> quality.put(task, mean(scores)));
      candidates.add(
          ModelCandidate.builder(id)
              .local(model.path("local").asBoolean())
              .capabilities(strings(model.path("capabilities")))
              .contextWindow(model.path("contextWindow").asInt())
              .costPerMillionTokens(
                  model.path("inputPrice").asDouble(), model.path("outputPrice").asDouble())
              .quality(quality)
              .timeToFirstTokenMillis((long) percentile(ttft, 0.5))
              .build());
    }
    ModelCandidate cheapest =
        candidates.stream()
            .min(
                Comparator.comparingDouble(ModelCandidate::blendedCostPerMillionTokens)
                    .thenComparing(ModelCandidate::id))
            .orElseThrow();
    ModelCandidate best =
        candidates.stream()
            .max(
                Comparator.comparingDouble((ModelCandidate c) -> c.qualityFor(null))
                    .thenComparing(ModelCandidate::id))
            .orElseThrow();
    LinkedHashMap<String, List<ModelCandidate>> arms = new LinkedHashMap<>();
    for (ModelCandidate candidate : candidates)
      arms.put("model:" + candidate.id(), List.of(candidate));
    arms.put("static-cheapest", List.of(cheapest));
    arms.put("static-best-calibration", List.of(best));
    for (String arm :
        List.of("router", "without-classifier", "without-quality", "without-performance"))
      arms.put(arm, candidates);
    List<Map<String, Object>> rows = new ArrayList<>();
    Map<String, String> selected = new HashMap<>();
    Map<String, Integer> changed = new LinkedHashMap<>();
    for (var arm : arms.entrySet()) {
      for (JsonNode item : cases)
        if (item.path("split").asText().equals("eval")) {
          String id = item.path("id").asText(), task = item.path("task").asText();
          RoutingPolicy policy =
              switch (arm.getKey()) {
                case "without-quality" ->
                    new RoutingPolicy(
                        0.30,
                        0,
                        0.15,
                        0.10,
                        0.05,
                        OptionalDouble.empty(),
                        OptionalDouble.empty(),
                        OptionalDouble.empty(),
                        false);
                case "without-performance" ->
                    new RoutingPolicy(
                        0.30,
                        0.40,
                        0,
                        0.10,
                        0.05,
                        OptionalDouble.empty(),
                        OptionalDouble.empty(),
                        OptionalDouble.empty(),
                        false);
                default -> RoutingPolicy.BALANCED;
              };
          String predicted =
              (arm.getKey().equals("without-classifier") || item.path("predictedTask").isNull())
                  ? null
                  : item.path("predictedTask").asText();
          ModelFleet.Builder<JsonNode> builder = ModelFleet.<JsonNode>builder().policy(policy);
          for (ModelCandidate candidate : arm.getValue())
            builder.model(candidate, runs.get(id).get(candidate.id()));
          ModelFleet<JsonNode> fleet = builder.build();
          RoutingBudget budget = new RoutingBudget(item.path("budget").decimalValue());
          RoutingExecutionOptions options =
              RoutingExecutionOptions.builder()
                  .sharedBudget(budget)
                  .tokenBounds(
                      new RoutingTokenBounds(
                          item.path("inputBound").asInt(), item.path("outputBound").asInt()))
                  .build();
          RoutingRequirements requirements =
              new RoutingRequirements(
                  strings(item.path("capabilities")),
                  item.path("localOnly").asBoolean()
                      ? RoutingDataBoundary.LOCAL_ONLY
                      : RoutingDataBoundary.REMOTE_ALLOWED);
          List<String> attempted = new ArrayList<>();
          double[] latency = {
            arm.getKey().startsWith("model:")
                    || arm.getKey().startsWith("static-")
                    || arm.getKey().equals("without-classifier")
                ? 0
                : item.path("classifierMillis").asDouble()
          };
          String chosen = "", failure = "";
          double quality = 0;
          try {
            var result =
                fleet.execute(
                    RoutingRequest.builder(item.path("prompt").asText())
                        .taskType(predicted)
                        .build(),
                    requirements,
                    RoutingContinuity.none(),
                    options,
                    run -> {
                      attempted.add(run.path("modelId").asText());
                      latency[0] += run.path("latencyMillis").asDouble();
                      if (!run.path("success").asBoolean())
                        throw new IllegalStateException("recorded provider failure");
                      return run;
                    },
                    run ->
                        run.hasNonNull("inputTokens")
                            ? new RoutingUsage(
                                run.path("inputTokens").asLong(), run.path("outputTokens").asLong())
                            : null);
            chosen = result.decision().selected().id();
            quality = score(item, result.value());
          } catch (RoutingExecutionException
              | RoutingBudgetExceededException
              | NoEligibleModelException denied) {
            failure = denied.getClass().getSimpleName();
          }
          boolean deadlineViolation = latency[0] > item.path("deadlineMillis").asDouble();
          boolean budgetViolation =
              budget.snapshot().spent().compareTo(budget.snapshot().limit()) > 0;
          Map<String, Object> row = new LinkedHashMap<>();
          row.put("arm", arm.getKey());
          row.put("caseId", id);
          row.put("dataset", item.path("dataset").asText());
          row.put("task", task);
          row.put("selected", chosen);
          row.put("attempted", attempted);
          row.put("quality", quality);
          row.put("failure", failure);
          row.put("replayedLatencyMillis", latency[0]);
          row.put("costIncludingFailedAttempts", budget.snapshot().spent());
          row.put("deadlineViolation", deadlineViolation);
          row.put("budgetViolation", budgetViolation);
          row.put(
              "policyCompliantSuccess",
              quality == 1 && !deadlineViolation && !budgetViolation && failure.isEmpty());
          rows.add(row);
          if (arm.getKey().equals("router")) selected.put(id, chosen);
          if (arm.getKey().startsWith("without-"))
            changed.merge(
                arm.getKey(), Objects.equals(selected.get(id), chosen) ? 0 : 1, Integer::sum);
        }
    }
    Map<String, Object> summaries = new LinkedHashMap<>();
    for (String arm : arms.keySet()) {
      Map<String, Object> groups = new TreeMap<>();
      for (Map<String, Object> row : rows)
        if (row.get("arm").equals(arm)) {
          groups.put(row.get("dataset") + "/" + row.get("task"), null);
        }
      for (String group : groups.keySet()) {
        List<Map<String, Object>> members =
            rows.stream()
                .filter(
                    row ->
                        row.get("arm").equals(arm)
                            && (row.get("dataset") + "/" + row.get("task")).equals(group))
                .toList();
        List<Double> latencies =
            members.stream().map(row -> (Double) row.get("replayedLatencyMillis")).toList();
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("count", members.size());
        stats.put(
            "quality",
            members.stream()
                .mapToDouble(row -> (Double) row.get("quality"))
                .average()
                .orElseThrow());
        stats.put("p50Millis", percentile(latencies, .50));
        stats.put("p95Millis", percentile(latencies, .95));
        stats.put("p99Millis", percentile(latencies, .99));
        stats.put(
            "cost",
            members.stream()
                .map(row -> (BigDecimal) row.get("costIncludingFailedAttempts"))
                .reduce(BigDecimal.ZERO, BigDecimal::add));
        for (String flag :
            List.of("deadlineViolation", "budgetViolation", "policyCompliantSuccess"))
          stats.put(
              flag, members.stream().filter(row -> Boolean.TRUE.equals(row.get(flag))).count());
        stats.put(
            "failedOrRejected",
            members.stream().filter(row -> !row.get("failure").equals("")).count());
        groups.put(group, stats);
      }
      summaries.put(arm, groups);
    }
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("schemaVersion", 1);
    report.put("mode", "paired-observation-replay");
    report.put(
        "limitations",
        "Not a live deadline, provider-cancellation, or classification benchmark. Exact stripped answer matching only. Failed/unknown usage is charged at its declared bound. Zero changed selections is no evidence of an ablation's value. Local infrastructure cost is not modeled.");
    report.put("manifestSha256", digest(Files.readAllBytes(manifestPath)));
    report.put("configuration", manifest);
    report.put("calibrationCandidates", candidates);
    report.put("ablationChangedSelections", changed);
    report.put("summaries", summaries);
    report.put("rows", rows);
    return report;
  }

  private static JsonNode readPinned(Path manifest, JsonNode resource) throws Exception {
    Path parent = Objects.requireNonNull(manifest.toAbsolutePath().getParent(), "manifest parent");
    Path path = parent.resolve(text(resource, "path")).normalize();
    byte[] data = Files.readAllBytes(path);
    require(digest(data).equals(text(resource, "sha256")), "digest mismatch: " + path);
    return JSON.readTree(data);
  }

  private static String digest(byte[] data) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data));
  }

  private static String text(JsonNode item, String field) {
    String value = item.path(field).asText();
    require(!value.isBlank(), "missing " + field);
    return value;
  }

  private static void nonNegative(JsonNode item, String field) {
    require(
        item.path(field).isNumber()
            && Double.isFinite(item.path(field).asDouble())
            && item.path(field).asDouble() >= 0,
        "invalid " + field);
  }

  private static Set<String> strings(JsonNode node) {
    Set<String> result = new HashSet<>();
    for (JsonNode item : node) result.add(item.asText());
    return Set.copyOf(result);
  }

  private static double score(JsonNode item, JsonNode run) {
    if (!run.path("success").asBoolean()) return 0;
    for (JsonNode answer : item.path("answers"))
      if (answer.asText().strip().equals(run.path("response").asText().strip())) return 1;
    return 0;
  }

  private static double mean(List<Double> values) {
    return values.stream().mapToDouble(Double::doubleValue).average().orElseThrow();
  }

  private static double percentile(List<Double> values, double quantile) {
    List<Double> sorted = values.stream().sorted().toList();
    return sorted.get(Math.max(0, (int) Math.ceil(quantile * sorted.size()) - 1));
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalArgumentException(message);
  }
}
