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
package com.integrallis.models.bench.fusion;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Pilot/arm summaries with ORACLE headroom and the stop rule, and the results-log renderer. */
final class SummaryCommands {
  static final double STOP_RULE_HEADROOM = 0.03;
  private final FusionCli cli;

  SummaryCommands(FusionCli cli) {
    this.cli = cli;
  }

  int summarize(String[] args) throws IOException {
    CliOptions options =
        CliOptions.parse(args, 1, Set.of("reports", "report", "member-labels", "fused-label"));
    Path report = Path.of(options.required("report"));
    List<String> memberLabels = FusionCli.csv(options.get("member-labels", "A,B,C"));
    String fusedLabel = options.get("fused-label", "F-tuned");
    List<Map<String, Object>> rows = new ArrayList<>();
    Map<String, JsonNode> byLabel = new LinkedHashMap<>();
    for (Path path : options.fileList("reports")) {
      JsonNode root = cli.mapper().readTree(path.toFile());
      if (!"arm".equals(root.path("kind").asText())) {
        continue;
      }
      JsonNode config = root.path("config");
      JsonNode summary = root.path("summary");
      String label =
          config.path("arm").path("label").asText(config.path("arm").path("canonical").asText());
      byLabel.put(label, root);
      rows.add(
          FusionCli.map(
              "label", label,
              "spec", config.path("arm").path("canonical").asText(),
              "pilotSubstitution", config.path("arm").path("pilotSubstitution").asText(null),
              "dataset", config.path("dataset").path("name").asText(),
              "thinking", config.path("decoding").path("thinking").asText(),
              "items", summary.path("items").asInt(),
              "accuracy", summary.path("accuracy").asDouble(),
              "tokensPerSecond", summary.path("tokensPerSecond").asDouble(),
              "peakRssBytes", root.path("timings").path("peakRssBytes").asLong(),
              "coreSeconds", summary.path("coreSeconds").asDouble(),
              "coreSecondsPerItem", summary.path("coreSecondsPerItem").asDouble(),
              "reusedCoreSeconds", summary.path("reusedCoreSeconds").asDouble(),
              "truncationRate", summary.path("truncationRate").asDouble(),
              "p95TraceTokens", summary.path("p95TraceTokens").asDouble(),
              "aggregationAccuracy", summary.path("aggregationAccuracy"),
              "g3Flagged", summary.path("g3").path("flagged").asBoolean(false),
              "report", path.toAbsolutePath().toString()));
      cli.out()
          .printf(
              Locale.ROOT,
              "%-14s %-40s acc=%.3f tok/s=%7.2f rss=%6.2fGB core-s/item=%7.1f trunc=%.3f p95=%.0f%n",
              label,
              config.path("arm").path("canonical").asText(),
              summary.path("accuracy").asDouble(),
              summary.path("tokensPerSecond").asDouble(),
              root.path("timings").path("peakRssBytes").asLong() / 1e9,
              summary.path("coreSecondsPerItem").asDouble(),
              summary.path("truncationRate").asDouble(),
              summary.path("p95TraceTokens").asDouble());
    }

    Map<String, Object> oracle = oracle(byLabel, memberLabels);
    Map<String, Object> stopRule = stopRule(byLabel, oracle, fusedLabel);
    cli.out().println("ORACLE " + oracle);
    cli.out().println("STOP-RULE " + stopRule.get("verdict") + " " + stopRule);
    Map<String, Object> summary =
        FusionCli.map("arms", rows.size(), "oracle", oracle, "stopRule", stopRule);
    long now = System.nanoTime();
    try (PeakRss rss = new PeakRss()) {
      cli.writeReport(
          report,
          cli.plainReport(
              "summary",
              FusionCli.emptyConfig(
                  List.of(),
                  null,
                  null,
                  FusionCli.map("memberLabels", memberLabels, "fusedLabel", fusedLabel)),
              now,
              FusionEnvironment.processCpuNanos(),
              rss,
              summary,
              rows,
              null),
          true);
    }
    cli.out().printf("DONE summarize arms=%d report=%s%n", rows.size(), report.toAbsolutePath());
    return FusionCli.PASS;
  }

  static Map<String, Object> oracle(Map<String, JsonNode> byLabel, List<String> memberLabels) {
    Map<String, Map<String, Boolean>> correctness = new LinkedHashMap<>();
    Map<String, Double> accuracy = new LinkedHashMap<>();
    String datasetSha = null;
    for (String label : memberLabels) {
      JsonNode root = byLabel.get(label);
      if (root == null) {
        return FusionCli.map("available", false, "reason", "missing member report " + label);
      }
      String sha =
          root.path("config").path("dataset").path("sha256").asText()
              + "|"
              + root.path("config").path("dataset").path("selectedIdsSha256").asText();
      if (datasetSha != null && !datasetSha.equals(sha)) {
        return FusionCli.map("available", false, "reason", "member reports cover different items");
      }
      datasetSha = sha;
      Map<String, Boolean> items = new HashMap<>();
      root.path("items")
          .forEach(item -> items.put(item.path("id").asText(), item.path("correct").asBoolean()));
      correctness.put(label, items);
      accuracy.put(label, root.path("summary").path("accuracy").asDouble());
    }
    Map<String, Boolean> first = correctness.values().iterator().next();
    int any = 0;
    for (String id : first.keySet()) {
      boolean correct = false;
      for (Map<String, Boolean> member : correctness.values()) {
        correct |= member.getOrDefault(id, false);
      }
      any += correct ? 1 : 0;
    }
    double oracleAccuracy = first.isEmpty() ? 0 : any / (double) first.size();
    String best = null;
    double bestAccuracy = -1;
    for (Map.Entry<String, Double> entry : accuracy.entrySet()) {
      if (entry.getValue() > bestAccuracy) {
        best = entry.getKey();
        bestAccuracy = entry.getValue();
      }
    }
    return FusionCli.map(
        "available",
        true,
        "items",
        first.size(),
        "oracleAccuracy",
        oracleAccuracy,
        "memberAccuracy",
        accuracy,
        "bestMember",
        best,
        "bestMemberAccuracy",
        bestAccuracy,
        "headroom",
        oracleAccuracy - bestAccuracy,
        "bestMemberNote",
        "chosen on these items; the protocol chooses it on the development split at S3");
  }

  static Map<String, Object> stopRule(
      Map<String, JsonNode> byLabel, Map<String, Object> oracle, String fusedLabel) {
    JsonNode fused = byLabel.get(fusedLabel);
    if (!Boolean.TRUE.equals(oracle.get("available")) || fused == null) {
      return FusionCli.map(
          "verdict",
          "UNDETERMINED",
          "reason",
          fused == null ? "missing " + fusedLabel + " report" : oracle.get("reason"));
    }
    double headroom = (double) oracle.get("headroom");
    double fusedAccuracy = fused.path("summary").path("accuracy").asDouble();
    double best = (double) oracle.get("bestMemberAccuracy");
    boolean triggered = headroom < STOP_RULE_HEADROOM && fusedAccuracy <= best;
    return FusionCli.map(
        "verdict", triggered ? "TRIGGERED" : "NOT-TRIGGERED",
        "headroom", headroom,
        "headroomThreshold", STOP_RULE_HEADROOM,
        "fusedLabel", fusedLabel,
        "fusedAccuracy", fusedAccuracy,
        "fusedPilotSubstitution",
            fused.path("config").path("arm").path("pilotSubstitution").asText(null),
        "bestMemberAccuracy", best);
  }

  int resultsLog(String[] args) throws IOException {
    CliOptions options = CliOptions.parse(args, 1, Set.of("in", "report"));
    Path in = options.existingFile("in");
    Path out = Path.of(options.required("report"));
    StringBuilder markdown = new StringBuilder();
    markdown
        .append("<!-- generated from ")
        .append(in.getFileName())
        .append(" by logit-fusion results-log; do not edit -->\n\n");
    markdown.append(
        "| date | commit | host | phase | arm | dataset | evidence | outcome | provenance |\n");
    markdown.append("|---|---|---|---|---|---|---|---|---|\n");
    int rows = 0;
    for (String line : Files.readAllLines(in, StandardCharsets.UTF_8)) {
      if (line.isBlank()) {
        continue;
      }
      JsonNode row = cli.mapper().readTree(line);
      markdown.append('|');
      for (String field :
          List.of(
              "date",
              "modelsCommit",
              "host",
              "phase",
              "arm",
              "dataset",
              "evidencePath",
              "outcome",
              "provenance")) {
        String value =
            row.path(field).isMissingNode() || row.path(field).isNull()
                ? ""
                : row.path(field).asText();
        markdown.append(' ').append(value.replace("|", "\\|")).append(" |");
      }
      markdown.append('\n');
      rows++;
    }
    Files.writeString(out, markdown.toString(), StandardCharsets.UTF_8);
    cli.out().printf("DONE results-log rows=%d report=%s%n", rows, out.toAbsolutePath());
    return FusionCli.PASS;
  }
}
