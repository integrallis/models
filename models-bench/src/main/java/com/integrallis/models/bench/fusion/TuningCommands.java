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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Stage S3: weight grid search, confidence calibration, and the frozen record. */
final class TuningCommands {
  private final FusionCli cli;

  TuningCommands(FusionCli cli) {
    this.cli = cli;
  }

  int tuneWeights(String[] args) throws IOException {
    Set<String> allowed = new HashSet<>(FusionCli.RUN_OPTIONS);
    allowed.addAll(Set.of("arm-members", "rule", "grid-step", "work-dir"));
    allowed.removeAll(
        Set.of("arm", "arm-label", "frozen", "frozen-sha256", "member-reports", "pilot"));
    CliOptions options = CliOptions.parse(args, 1, allowed);
    List<String> members = List.of(options.required("arm-members").split("\\+"));
    FusionRule rule = FusionRule.parse(options.get("rule", "poe"));
    double step = Double.parseDouble(options.get("grid-step", "0.1"));
    int divisions = (int) Math.round(1.0 / step);
    if (Math.abs(divisions * step - 1.0) > 1e-9) {
      throw new IllegalArgumentException("--grid-step must divide 1 exactly");
    }
    Path workDir = Path.of(options.required("work-dir"));
    Files.createDirectories(workDir);
    Path report = Path.of(options.required("report"));
    String revision = FusionCli.modelsRevision(options);
    FusionCli.LoadedData data = cli.loadData(options);

    List<Map<String, Object>> points = new ArrayList<>();
    for (double[] weights : FusionMath.simplexGrid(members.size(), divisions)) {
      String weightText =
          String.join(
              ",",
              Arrays.stream(weights).mapToObj(w -> String.format(Locale.ROOT, "%.4f", w)).toList());
      String spec = "fuse:" + String.join("+", members) + ":" + rule.id() + ":" + weightText;
      Path pointReport =
          workDir.resolve("point-" + rule.id() + "-" + weightText.replace(',', '_') + ".json");
      if (!Files.exists(pointReport)) {
        List<String> runArgs = new ArrayList<>(Arrays.asList(args).subList(1, args.length));
        stripOption(runArgs, "arm-members");
        stripOption(runArgs, "rule");
        stripOption(runArgs, "grid-step");
        stripOption(runArgs, "work-dir");
        stripOption(runArgs, "report");
        runArgs.addAll(
            List.of(
                "--arm",
                spec,
                "--arm-label",
                "tune-" + weightText,
                "--report",
                pointReport.toString()));
        runArgs.addFirst("run");
        int status = cli.execute(runArgs.toArray(String[]::new));
        if (status != FusionCli.PASS) {
          throw new IllegalStateException("grid point failed: " + spec);
        }
      }
      byte[] bytes = Files.readAllBytes(pointReport);
      JsonNode root = cli.mapper().readTree(bytes);
      if (!data.config()
              .sha256()
              .equals(root.path("config").path("dataset").path("sha256").asText())
          || !data.config()
              .selectedIdsSha256()
              .equals(root.path("config").path("dataset").path("selectedIdsSha256").asText())) {
        throw new IllegalStateException(pointReport + " was produced on different items");
      }
      points.add(
          FusionCli.map(
              "weights", Arrays.stream(weights).boxed().toList(),
              "accuracy", root.path("summary").path("accuracy").asDouble(),
              "correct", root.path("summary").path("correct").asInt(),
              "items", root.path("summary").path("items").asInt(),
              "report", pointReport.toAbsolutePath().toString(),
              "reportSha256", Digests.sha256(bytes)));
    }
    Map<String, Object> best = selectBest(points, members.size());
    Map<String, Object> summary =
        FusionCli.map(
            "members",
            members,
            "rule",
            rule.id(),
            "gridStep",
            step,
            "points",
            points.size(),
            "best",
            best,
            "selection",
            "highest dev accuracy; ties -> smallest L1 distance to uniform; then grid order");
    long now = System.nanoTime();
    try (PeakRss rss = new PeakRss()) {
      cli.writeReport(
          report,
          cli.plainReport(
              "tuning",
              FusionCli.emptyConfig(List.of(), data.config(), null, summary),
              now,
              FusionEnvironment.processCpuNanos(),
              rss,
              summary,
              points,
              revision),
          true);
    }
    cli.out()
        .printf(
            "DONE tune-weights %s %s best=%s report=%s%n",
            String.join("+", members), rule.id(), best.get("weights"), report.toAbsolutePath());
    return FusionCli.PASS;
  }

  @SuppressWarnings("unchecked")
  static Map<String, Object> selectBest(List<Map<String, Object>> points, int members) {
    Map<String, Object> best = null;
    double bestAccuracy = -1;
    double bestDistance = Double.POSITIVE_INFINITY;
    for (Map<String, Object> point : points) {
      double accuracy = (double) point.get("accuracy");
      double distance = 0;
      for (double w : (List<Double>) point.get("weights")) {
        distance += Math.abs(w - 1.0 / members);
      }
      if (accuracy > bestAccuracy
          || (accuracy == bestAccuracy && distance < bestDistance - 1e-12)) {
        best = point;
        bestAccuracy = accuracy;
        bestDistance = distance;
      }
    }
    return best;
  }

  private static void stripOption(List<String> args, String name) {
    int index = args.indexOf("--" + name);
    if (index >= 0) {
      args.remove(index + 1);
      args.remove(index);
    }
  }

  int calibrate(String[] args) throws IOException {
    CliOptions options =
        CliOptions.parse(args, 1, Set.of("member-reports", "dataset", "scorer", "report"));
    DatasetKind kind = DatasetKind.parse(options.required("dataset"));
    AnswerExtractor extractor =
        kind == DatasetKind.GENERIC
            ? AnswerExtractor.forScorer(options.required("scorer"))
            : AnswerExtractor.forDataset(kind);
    Path report = Path.of(options.required("report"));
    Map<String, Object> members = new LinkedHashMap<>();
    String datasetSha = null;
    String idsSha = null;
    for (Path path : options.fileList("member-reports")) {
      byte[] bytes = Files.readAllBytes(path);
      JsonNode root = cli.mapper().readTree(bytes);
      JsonNode config = root.path("config");
      if (!"member".equals(config.path("arm").path("kind").asText())) {
        throw new IllegalArgumentException(path + " is not a single-member arm report");
      }
      String sha = config.path("dataset").path("sha256").asText();
      if (datasetSha != null && !datasetSha.equals(sha)) {
        throw new IllegalArgumentException("calibration reports come from different datasets");
      }
      datasetSha = sha;
      idsSha = config.path("dataset").path("selectedIdsSha256").asText();
      List<ConfidenceCalibration.Sample> samples = new ArrayList<>();
      for (JsonNode item : root.path("items")) {
        JsonNode output = item.path("outputs").get(0);
        if (output.path("confidence").isNumber() && !output.path("statedAnswer").isNull()) {
          boolean correct =
              extractor.equivalent(
                  output.path("statedAnswer").asText(), item.path("gold").asText());
          samples.add(
              new ConfidenceCalibration.Sample(output.path("confidence").asDouble(), correct));
        }
      }
      ConfidenceCalibration.Fit fit = ConfidenceCalibration.fit(samples);
      String member = config.path("arm").path("members").get(0).asText();
      members.put(
          member,
          FusionCli.map(
              "temperature", fit.temperature(),
              "eceBefore", fit.eceBefore(),
              "eceAfter", fit.eceAfter(),
              "samples", fit.samples(),
              "thinking", config.path("decoding").path("thinking").asText(),
              "sourceReport", path.toAbsolutePath().toString(),
              "sourceReportSha256", Digests.sha256(bytes)));
      cli.out()
          .printf(
              "G7 %s temperature=%.4f ECE before=%.4f after=%.4f n=%d%n",
              member, fit.temperature(), fit.eceBefore(), fit.eceAfter(), fit.samples());
    }
    Map<String, Object> summary =
        FusionCli.map(
            "method",
                "p = exp(meanAnswerTokenLogProb / temperature); temperature minimises Bernoulli NLL on a log grid [0.05, 20]; ECE with 10 equal-width bins",
            "datasetSha256", datasetSha,
            "selectedIdsSha256", idsSha,
            "members", members);
    long now = System.nanoTime();
    try (PeakRss rss = new PeakRss()) {
      cli.writeReport(
          report,
          cli.plainReport(
              "calibration",
              FusionCli.emptyConfig(List.of(), null, null, Map.of()),
              now,
              FusionEnvironment.processCpuNanos(),
              rss,
              summary,
              List.of(),
              null),
          true);
    }
    cli.out()
        .printf("DONE calibrate members=%d report=%s%n", members.size(), report.toAbsolutePath());
    return FusionCli.PASS;
  }

  int freeze(String[] args) throws IOException {
    CliOptions options =
        CliOptions.parse(
            args,
            1,
            Set.of(
                "tuning",
                "calibration",
                "best-member-reports",
                "dataset",
                "data",
                "data-manifest",
                "scorer",
                "report"));
    Path report = Path.of(options.required("report"));
    if (Files.exists(report)) {
      throw new IllegalArgumentException("refusing to overwrite frozen record " + report);
    }
    FusionCli.LoadedData dev = cli.loadData(options);
    List<Map<String, Object>> weights = new ArrayList<>();
    for (Path path : options.fileList("tuning")) {
      byte[] bytes = Files.readAllBytes(path);
      JsonNode summary = cli.mapper().readTree(bytes).path("summary");
      requireDev(dev, cli.mapper().readTree(bytes).path("config").path("dataset"), path);
      List<String> members = new ArrayList<>();
      summary.path("members").forEach(m -> members.add(m.asText()));
      weights.add(
          FusionCli.map(
              "members", members,
              "rule", summary.path("rule").asText(),
              "weights",
                  cli.mapper().convertValue(summary.path("best").path("weights"), List.class),
              "devAccuracy", summary.path("best").path("accuracy").asDouble(),
              "gridStep", summary.path("gridStep").asDouble(),
              "tuningReport", path.toAbsolutePath().toString(),
              "tuningReportSha256", Digests.sha256(bytes)));
    }
    Map<String, Object> accuracies = new LinkedHashMap<>();
    Map<String, Object> sources = new LinkedHashMap<>();
    String bestMember = null;
    double bestAccuracy = -1;
    for (Path path : options.fileList("best-member-reports")) {
      byte[] bytes = Files.readAllBytes(path);
      JsonNode root = cli.mapper().readTree(bytes);
      requireDev(dev, root.path("config").path("dataset"), path);
      if (!"member".equals(root.path("config").path("arm").path("kind").asText())) {
        throw new IllegalArgumentException(path + " is not a single-member arm report");
      }
      String member = root.path("config").path("arm").path("members").get(0).asText();
      double accuracy = root.path("summary").path("accuracy").asDouble();
      accuracies.put(member, accuracy);
      sources.put(member, Digests.sha256(bytes));
      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
        bestMember = member;
      }
    }
    JsonNode calibration = null;
    String calibrationSha = null;
    if (options.has("calibration")) {
      byte[] bytes = Files.readAllBytes(options.existingFile("calibration"));
      calibration = cli.mapper().readTree(bytes).path("summary");
      calibrationSha = Digests.sha256(bytes);
    }
    Map<String, Object> frozen =
        FusionCli.map(
            "schemaVersion",
            1,
            "kind",
            "frozen",
            "createdAt",
            java.time.Instant.now().toString(),
            "devSplit",
            FusionCli.map(
                "path", dev.config().path(),
                "sha256", dev.config().sha256(),
                "idsSha256", DatasetItem.idsSha256(dev.all()),
                "ids", dev.all().stream().map(DatasetItem::id).toList()),
            "weights",
            weights,
            "bestMember",
            FusionCli.map(
                "name",
                bestMember,
                "devAccuracy",
                accuracies,
                "sourceReportSha256",
                sources,
                "tieRule",
                "first report in --best-member-reports order wins ties"),
            "calibration",
            calibration == null
                ? null
                : FusionCli.map(
                    "members",
                    calibration.path("members"),
                    "method",
                    calibration.path("method").asText(),
                    "calibrationReportSha256",
                    calibrationSha));
    Path parent = report.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    byte[] bytes = cli.mapper().writerWithDefaultPrettyPrinter().writeValueAsBytes(frozen);
    Files.write(report, bytes);
    cli.out()
        .printf(
            "DONE freeze sha256=%s report=%s%n", Digests.sha256(bytes), report.toAbsolutePath());
    return FusionCli.PASS;
  }

  private static void requireDev(FusionCli.LoadedData dev, JsonNode dataset, Path path) {
    if (!dev.config().sha256().equals(dataset.path("sha256").asText())) {
      throw new IllegalArgumentException(
          path + " was not produced on the development split " + dev.config().path());
    }
  }
}
