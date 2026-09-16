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
import com.integrallis.models.backend.purejava.GgufEmbeddingBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Diagnoses whether a small in-process embedding model improves activated-adapter applicability.
 *
 * <p>This command consumes only previously exposed development records. Its output is research
 * evidence, never qualification evidence.
 */
final class SemanticApplicabilityDiagnosticCli {
  private static final Set<String> OPTIONS =
      Set.of("embedding-model", "records", "decision-profile", "output");

  private SemanticApplicabilityDiagnosticCli() {}

  record Observation(
      String id,
      String kind,
      String partition,
      boolean callExpected,
      double semanticSimilarity,
      double adapterMargin) {}

  record Score(
      int calls,
      int noCalls,
      int correctCalls,
      int correctNoCalls,
      double callAccuracy,
      double noCallAccuracy,
      double balancedAccuracy) {}

  record Selection(
      double semanticThreshold, double marginThreshold, Score calibration, Score screen) {}

  record Report(
      int schemaVersion,
      String experiment,
      String createdAt,
      String embeddingModelSha256,
      String recordsSha256,
      String decisionProfileSha256,
      Selection selection,
      List<Observation> observations) {}

  static int run(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path embeddingModel = requiredFile(values, "embedding-model");
    Path records = requiredFile(values, "records");
    Path decisionProfile = requiredFile(values, "decision-profile");
    Path output = Path.of(required(values, "output"));

    ObjectMapper mapper = new ObjectMapper();
    Map<String, Decision> decisions = loadDecisions(mapper, decisionProfile);
    List<JsonNode> sourceRecords = loadRecords(mapper, records);
    if (!sourceRecords.stream()
        .map(item -> requiredText(item, "id"))
        .collect(java.util.stream.Collectors.toSet())
        .equals(decisions.keySet())) {
      throw new IllegalArgumentException("records and decision profile contain different case IDs");
    }

    List<Observation> observations = new ArrayList<>(sourceRecords.size());
    try (GgufEmbeddingBackend embedding =
        GgufEmbeddingBackend.builder(PureJavaBackend.load(embeddingModel)).build()) {
      int index = 0;
      for (JsonNode source : sourceRecords) {
        String id = requiredText(source, "id");
        Decision decision = decisions.get(id);
        double similarity = maximumSimilarity(embedding, source);
        observations.add(
            new Observation(
                id,
                decision.kind(),
                decision.partition(),
                decision.callExpected(),
                similarity,
                decision.margin()));
        index++;
        System.out.printf(
            Locale.ROOT,
            "%d/%d %-24s semantic=% .5f margin=% .5f%n",
            index,
            sourceRecords.size(),
            id,
            similarity,
            decision.margin());
      }
    }

    Selection selection = select(observations);
    Report report =
        new Report(
            1,
            "qwen3-17b-semantic-applicability-diagnostic",
            Instant.now().toString(),
            Hashing.sha256(embeddingModel),
            Hashing.sha256(records),
            Hashing.sha256(decisionProfile),
            selection,
            List.copyOf(observations));
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    Files.writeString(output, mapper.writeValueAsString(report) + System.lineSeparator());
    System.out.printf(
        Locale.ROOT,
        "semantic > %.6f AND adapter margin > %.6f%n",
        selection.semanticThreshold(),
        selection.marginThreshold());
    print("calibration", selection.calibration());
    print("screen", selection.screen());
    System.out.println("diagnostic report: " + output.toAbsolutePath());
    return 0;
  }

  static String userIntent(JsonNode record) {
    JsonNode messages = requiredArray(record, "messages");
    List<String> contents = new ArrayList<>();
    for (JsonNode message : messages) {
      if ("user".equals(message.path("role").asText())) {
        contents.add(requiredText(message, "content").strip());
      }
    }
    if (contents.isEmpty()) {
      throw new IllegalArgumentException("record contains no user message");
    }
    return String.join("\n", contents);
  }

  static List<String> toolCapabilities(JsonNode record) {
    JsonNode tools = requiredArray(record, "tools");
    List<String> result = new ArrayList<>();
    for (JsonNode tool : tools) {
      JsonNode function = tool.path("function");
      if (!function.isObject()) {
        throw new IllegalArgumentException("tool does not contain a function object");
      }
      List<String> sentences = new ArrayList<>();
      sentences.add(words(requiredText(function, "name")) + ".");
      optionalText(function, "description").ifPresent(value -> sentences.add(sentence(value)));
      JsonNode parameters = function.path("parameters");
      JsonNode properties = parameters.path("properties");
      if (properties.isObject()) {
        properties
            .properties()
            .forEach(
                entry -> {
                  sentences.add(words(entry.getKey()) + ".");
                  optionalText(entry.getValue(), "description")
                      .ifPresent(value -> sentences.add(sentence(value)));
                });
      }
      JsonNode required = parameters.path("required");
      if (required.isArray() && !required.isEmpty()) {
        List<String> names = new ArrayList<>();
        required.forEach(item -> names.add(words(item.asText())));
        sentences.add("Required: " + String.join(", ", names) + ".");
      }
      result.add(String.join(" ", sentences));
    }
    if (result.isEmpty()) {
      throw new IllegalArgumentException("record contains no tools");
    }
    return List.copyOf(result);
  }

  static Selection select(List<Observation> observations) {
    List<Observation> calibration = partition(observations, "calibration");
    List<Observation> screen = partition(observations, "screen");
    List<Double> semanticThresholds =
        thresholds(calibration.stream().map(Observation::semanticSimilarity).toList());
    List<Double> marginThresholds =
        thresholds(calibration.stream().map(Observation::adapterMargin).toList());

    Selection selected = null;
    for (double semanticThreshold : semanticThresholds) {
      for (double marginThreshold : marginThresholds) {
        Score calibrationScore = score(calibration, semanticThreshold, marginThreshold);
        Score screenScore = score(screen, semanticThreshold, marginThreshold);
        Selection candidate =
            new Selection(semanticThreshold, marginThreshold, calibrationScore, screenScore);
        if (selected == null || better(candidate, selected)) {
          selected = candidate;
        }
      }
    }
    if (selected == null) {
      throw new IllegalArgumentException("no calibration thresholds were available");
    }
    return selected;
  }

  static boolean predictsCall(
      Observation observation, double semanticThreshold, double marginThreshold) {
    return observation.semanticSimilarity() > semanticThreshold
        && observation.adapterMargin() > marginThreshold;
  }

  private static double maximumSimilarity(GgufEmbeddingBackend embedding, JsonNode source) {
    float[] query = embedding.embed(userIntent(source));
    double maximum = -1.0;
    for (String capability : toolCapabilities(source)) {
      maximum = Math.max(maximum, dot(query, embedding.embed(capability)));
    }
    return maximum;
  }

  private static double dot(float[] left, float[] right) {
    if (left.length != right.length) {
      throw new IllegalArgumentException("embedding dimensions differ");
    }
    double result = 0;
    for (int index = 0; index < left.length; index++) {
      result += (double) left[index] * right[index];
    }
    return result;
  }

  private static boolean better(Selection candidate, Selection current) {
    return Comparator.comparingDouble((Selection item) -> item.calibration().balancedAccuracy())
            .thenComparingInt(item -> item.calibration().correctNoCalls())
            .thenComparingInt(item -> item.calibration().correctCalls())
            .thenComparingDouble(Selection::semanticThreshold)
            .thenComparingDouble(Selection::marginThreshold)
            .compare(candidate, current)
        > 0;
  }

  private static Score score(
      List<Observation> observations, double semanticThreshold, double marginThreshold) {
    int calls = 0;
    int noCalls = 0;
    int correctCalls = 0;
    int correctNoCalls = 0;
    for (Observation observation : observations) {
      boolean predicted = predictsCall(observation, semanticThreshold, marginThreshold);
      if (observation.callExpected()) {
        calls++;
        correctCalls += predicted ? 1 : 0;
      } else {
        noCalls++;
        correctNoCalls += predicted ? 0 : 1;
      }
    }
    if (calls == 0 || noCalls == 0) {
      throw new IllegalArgumentException("both applicability classes are required");
    }
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

  private static List<Double> thresholds(List<Double> values) {
    List<Double> distinct = values.stream().distinct().sorted().toList();
    if (distinct.isEmpty()) {
      throw new IllegalArgumentException("threshold input is empty");
    }
    List<Double> candidates = new ArrayList<>();
    candidates.add(Math.nextDown(distinct.getFirst()));
    for (int index = 1; index < distinct.size(); index++) {
      double lower = distinct.get(index - 1);
      double upper = distinct.get(index);
      candidates.add(lower + (upper - lower) / 2.0);
    }
    candidates.add(distinct.getLast());
    return candidates;
  }

  private static List<Observation> partition(List<Observation> observations, String name) {
    List<Observation> result =
        observations.stream().filter(item -> name.equals(item.partition())).toList();
    if (result.isEmpty()) {
      throw new IllegalArgumentException("missing " + name + " observations");
    }
    return result;
  }

  private static List<JsonNode> loadRecords(ObjectMapper mapper, Path path) throws IOException {
    List<JsonNode> result = new ArrayList<>();
    try (var lines = Files.lines(path)) {
      for (String line : lines.toList()) {
        if (!line.isBlank()) {
          result.add(mapper.readTree(line));
        }
      }
    }
    if (result.isEmpty()) {
      throw new IllegalArgumentException("records file is empty");
    }
    return uniqueInputs(result);
  }

  static List<JsonNode> uniqueInputs(List<JsonNode> records) {
    Map<String, JsonNode> unique = new LinkedHashMap<>();
    for (JsonNode record : records) {
      String id = requiredText(record, "id");
      JsonNode previous = unique.putIfAbsent(id, record);
      if (previous != null
          && (!previous.path("messages").equals(record.path("messages"))
              || !previous.path("tools").equals(record.path("tools"))
              || !previous.path("prompt").equals(record.path("prompt")))) {
        throw new IllegalArgumentException("conflicting duplicate input for case ID: " + id);
      }
    }
    return List.copyOf(unique.values());
  }

  private static Map<String, Decision> loadDecisions(ObjectMapper mapper, Path path)
      throws IOException {
    JsonNode root = mapper.readTree(path.toFile());
    JsonNode source = requiredArray(root, "observations");
    Map<String, Decision> result = new LinkedHashMap<>();
    for (JsonNode item : source) {
      String id = requiredText(item, "id");
      Decision previous =
          result.put(
              id,
              new Decision(
                  requiredText(item, "kind"),
                  requiredText(item, "partition"),
                  item.path("callExpected").asBoolean(),
                  item.path("margin").asDouble()));
      if (previous != null) {
        throw new IllegalArgumentException("duplicate decision ID: " + id);
      }
    }
    return Map.copyOf(result);
  }

  private static JsonNode requiredArray(JsonNode parent, String name) {
    JsonNode value = parent.path(name);
    if (!value.isArray()) {
      throw new IllegalArgumentException(name + " must be an array");
    }
    return value;
  }

  private static java.util.Optional<String> optionalText(JsonNode parent, String name) {
    JsonNode value = parent.path(name);
    if (!value.isTextual() || value.asText().isBlank()) {
      return java.util.Optional.empty();
    }
    return java.util.Optional.of(value.asText().strip());
  }

  private static String requiredText(JsonNode parent, String name) {
    return optionalText(parent, name)
        .orElseThrow(() -> new IllegalArgumentException(name + " must be non-blank text"));
  }

  private static String words(String value) {
    return value.strip().replaceAll("[._-]+", " ").replaceAll("\\s+", " ");
  }

  private static String sentence(String value) {
    String stripped = value.strip();
    return stripped.endsWith(".") ? stripped : stripped + ".";
  }

  private static Path requiredFile(Map<String, String> values, String name) {
    Path path = Path.of(required(values, name));
    if (!Files.isRegularFile(path)) {
      throw new IllegalArgumentException(name + " is not a file: " + path);
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

  private static void print(String label, Score score) {
    System.out.printf(
        Locale.ROOT,
        "%s calls=%d/%d no-calls=%d/%d balanced=%.5f%n",
        label,
        score.correctCalls(),
        score.calls(),
        score.correctNoCalls(),
        score.noCalls(),
        score.balancedAccuracy());
  }

  private record Decision(String kind, String partition, boolean callExpected, double margin) {}
}
