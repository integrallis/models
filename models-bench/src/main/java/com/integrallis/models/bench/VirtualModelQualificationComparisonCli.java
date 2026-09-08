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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Compares repeated fresh-process control and hybrid virtual-model qualification arms. */
final class VirtualModelQualificationComparisonCli {
  private static final int PASS = 0;
  private static final int FAIL = 1;
  private static final ObjectMapper JSON =
      new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
  private static final Set<String> OPTIONS =
      Set.of("control-reports", "hybrid-reports", "minimum-improvement", "output");

  private VirtualModelQualificationComparisonCli() {}

  record Evaluation(
      long controlMedianMillis,
      long hybridMedianMillis,
      double improvement,
      boolean correctnessPassed,
      boolean qualified,
      List<String> diagnostics) {
    Evaluation {
      diagnostics = List.copyOf(diagnostics);
    }
  }

  record ComparisonReport(
      int schemaVersion,
      String createdAt,
      double minimumImprovement,
      List<String> controlReports,
      List<String> hybridReports,
      Evaluation evaluation) {
    ComparisonReport {
      controlReports = List.copyOf(controlReports);
      hybridReports = List.copyOf(hybridReports);
    }
  }

  private record RunEvidence(
      Path path,
      String arm,
      String createdAt,
      String runId,
      long processId,
      boolean complete,
      boolean passed,
      long totalMillis,
      JsonNode comparability,
      Set<String> artifactDigests) {}

  static int run(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    List<Path> controlPaths = paths(values, "control-reports");
    List<Path> hybridPaths = paths(values, "hybrid-reports");
    double minimumImprovement = decimal(values, "minimum-improvement", 0.05);
    if (minimumImprovement < 0 || minimumImprovement >= 1) {
      throw new IllegalArgumentException("--minimum-improvement must be in [0, 1)");
    }
    Path output =
        Path.of(values.getOrDefault("output", "build/reports/virtual-model/comparison.json"));

    List<RunEvidence> controls = read(controlPaths, "CONTROL");
    List<RunEvidence> hybrids = read(hybridPaths, "HYBRID");
    List<String> validation = validateComparable(controls, hybrids);
    boolean correct =
        validation.isEmpty()
            && java.util.stream.Stream.concat(controls.stream(), hybrids.stream())
                .allMatch(run -> run.complete() && run.passed());
    Evaluation evaluation =
        evaluate(
            controls.stream().map(RunEvidence::totalMillis).toList(),
            hybrids.stream().map(RunEvidence::totalMillis).toList(),
            correct,
            minimumImprovement);
    if (!validation.isEmpty()) {
      List<String> diagnostics = new ArrayList<>(validation);
      diagnostics.addAll(evaluation.diagnostics());
      evaluation =
          new Evaluation(
              evaluation.controlMedianMillis(),
              evaluation.hybridMedianMillis(),
              evaluation.improvement(),
              false,
              false,
              diagnostics);
    }
    ComparisonReport report =
        new ComparisonReport(
            1,
            Instant.now().toString(),
            minimumImprovement,
            controlPaths.stream()
                .map(path -> path.toAbsolutePath().normalize().toString())
                .toList(),
            hybridPaths.stream().map(path -> path.toAbsolutePath().normalize().toString()).toList(),
            evaluation);
    Path absolute = output.toAbsolutePath().normalize();
    Path parent = absolute.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    JSON.writeValue(absolute.toFile(), report);
    System.out.printf(
        "%s control-median=%d ms hybrid-median=%d ms improvement=%.2f%% report=%s%n",
        evaluation.qualified() ? "PASS" : "FAIL",
        evaluation.controlMedianMillis(),
        evaluation.hybridMedianMillis(),
        evaluation.improvement() * 100,
        absolute);
    evaluation.diagnostics().forEach(diagnostic -> System.out.println("  " + diagnostic));
    return evaluation.qualified() ? PASS : FAIL;
  }

  static Evaluation evaluate(
      List<Long> controlTotals,
      List<Long> hybridTotals,
      boolean correctnessPassed,
      double minimumImprovement) {
    if (controlTotals.size() < 3 || hybridTotals.size() < 3) {
      throw new IllegalArgumentException("at least three fresh processes are required per arm");
    }
    if (controlTotals.size() != hybridTotals.size()) {
      throw new IllegalArgumentException("control and hybrid arms must have the same run count");
    }
    long controlMedian = median(controlTotals);
    long hybridMedian = median(hybridTotals);
    double improvement = (controlMedian - hybridMedian) / (double) controlMedian;
    List<String> diagnostics = new ArrayList<>();
    if (!correctnessPassed) {
      diagnostics.add("one or more runs failed correctness or comparability");
    }
    if (improvement < minimumImprovement) {
      diagnostics.add(
          "median improvement "
              + String.format(java.util.Locale.ROOT, "%.4f", improvement)
              + " is below "
              + String.format(java.util.Locale.ROOT, "%.4f", minimumImprovement));
    }
    return new Evaluation(
        controlMedian,
        hybridMedian,
        improvement,
        correctnessPassed,
        diagnostics.isEmpty(),
        diagnostics);
  }

  private static List<RunEvidence> read(List<Path> paths, String expectedArm) throws IOException {
    List<RunEvidence> reports = new ArrayList<>();
    for (Path path : paths) {
      JsonNode root = JSON.readTree(path.toFile());
      String arm = root.path("arm").asText();
      if (!expectedArm.equals(arm)) {
        throw new IllegalArgumentException(path + " is " + arm + ", expected " + expectedArm);
      }
      Set<String> digests = new HashSet<>();
      root.path("artifacts").forEach(artifact -> digests.add(artifact.path("sha256").asText()));
      JsonNode comparable =
          JSON.valueToTree(
              Map.of(
                  "policyVersion", root.path("policyVersion"),
                  "protocolSha256", root.path("protocolSha256"),
                  "modelsRevision", root.path("modelsRevision"),
                  "maxTokens", root.path("maxTokens"),
                  "environment", root.path("environment"),
                  "jvmArguments", root.path("jvmArguments"),
                  "vectorRuntime", root.path("vectorRuntime"),
                  "vectorOverrides", root.path("vectorOverrides")));
      reports.add(
          new RunEvidence(
              path,
              arm,
              root.path("createdAt").asText(),
              root.path("runId").asText(),
              root.path("processId").asLong(),
              root.path("complete").asBoolean(),
              root.path("passed").asBoolean(),
              root.path("totalMillis").asLong(),
              comparable,
              Set.copyOf(digests)));
    }
    return List.copyOf(reports);
  }

  private static List<String> validateComparable(
      List<RunEvidence> controls, List<RunEvidence> hybrids) {
    List<String> diagnostics = new ArrayList<>();
    if (controls.size() != hybrids.size()) {
      diagnostics.add("control and hybrid report counts differ");
      return diagnostics;
    }
    List<RunEvidence> all =
        java.util.stream.Stream.concat(controls.stream(), hybrids.stream()).toList();
    if (all.isEmpty()) {
      diagnostics.add("no reports were provided");
      return diagnostics;
    }
    JsonNode expectedComparability = all.getFirst().comparability();
    Set<String> expectedDigests =
        Set.of(VirtualModelQualificationCli.CHAT_SHA256, VirtualModelQualificationCli.TOOL_SHA256);
    Set<String> runIds = new HashSet<>();
    Set<Long> processIds = new HashSet<>();
    for (RunEvidence run : all) {
      if (!run.comparability().equals(expectedComparability)) {
        diagnostics.add(run.path() + " does not match the common runtime configuration");
      }
      if (!run.artifactDigests().equals(expectedDigests)) {
        diagnostics.add(run.path() + " does not contain the pinned Qwen pair");
      }
      if (!runIds.add(run.runId())) {
        diagnostics.add("duplicate run id " + run.runId());
      }
      if (run.processId() <= 0 || !processIds.add(run.processId())) {
        diagnostics.add("runs must come from distinct process ids");
      }
      if (!run.complete() || !run.passed()) {
        diagnostics.add(run.path() + " did not pass every protocol turn");
      }
    }
    List<RunEvidence> chronological =
        all.stream().sorted(Comparator.comparing(RunEvidence::createdAt)).toList();
    long transitions =
        java.util.stream.IntStream.range(1, chronological.size())
            .filter(
                index -> !chronological.get(index - 1).arm().equals(chronological.get(index).arm()))
            .count();
    if (transitions < 2) {
      diagnostics.add("run order is not counterbalanced across arms");
    }
    return diagnostics;
  }

  private static long median(List<Long> values) {
    List<Long> ordered = values.stream().sorted().toList();
    int middle = ordered.size() / 2;
    if ((ordered.size() & 1) == 1) {
      return ordered.get(middle);
    }
    return (ordered.get(middle - 1) + ordered.get(middle)) / 2;
  }

  private static List<Path> paths(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    List<Path> paths =
        java.util.Arrays.stream(value.split(","))
            .map(String::trim)
            .filter(item -> !item.isEmpty())
            .map(Path::of)
            .toList();
    for (Path path : paths) {
      if (!Files.isRegularFile(path)) {
        throw new IllegalArgumentException("report does not exist: " + path);
      }
    }
    return paths;
  }

  private static double decimal(Map<String, String> values, String name, double defaultValue) {
    String value = values.get(name);
    if (value == null) {
      return defaultValue;
    }
    try {
      return Double.parseDouble(value);
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be a decimal: " + value, failure);
    }
  }
}
