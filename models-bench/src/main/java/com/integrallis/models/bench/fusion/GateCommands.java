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
import com.integrallis.models.bench.fusion.ReportModel.MemberConfig;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Validity gates G0–G6 as commands. */
final class GateCommands {
  private final FusionCli cli;

  GateCommands(FusionCli cli) {
    this.cli = cli;
  }

  int run(String gate, String[] args) throws IOException {
    return switch (gate) {
      case "g0" -> g0(CliOptions.parse(args, 2, Set.of("members", "report")));
      case "g1" ->
          g1(
              CliOptions.parse(
                  args,
                  2,
                  withDecoding(
                      "members",
                      "backend",
                      "models-revision",
                      "report",
                      "dataset",
                      "data",
                      "data-manifest",
                      "scorer",
                      "limit",
                      "offset",
                      "thinking",
                      "context-length",
                      "member-threads",
                      "prompts")));
      case "g2-dump" ->
          g2Dump(
              CliOptions.parse(
                  args,
                  2,
                  withDecoding(
                      "members",
                      "backend",
                      "models-revision",
                      "out",
                      "dataset",
                      "data",
                      "data-manifest",
                      "scorer",
                      "limit",
                      "offset",
                      "steps",
                      "weights",
                      "rule",
                      "context-length",
                      "member-threads",
                      "prompts",
                      "report")));
      case "g3" -> g3(CliOptions.parse(args, 2, Set.of("reports", "report")));
      case "g4" ->
          g4(
              CliOptions.parse(
                  args, 2, Set.of("dataset", "data", "data-manifest", "scorer", "report")));
      case "g5" -> g5(CliOptions.parse(args, 2, Set.of("reports", "report")));
      case "g6" -> g6(CliOptions.parse(args, 2, Set.of("fixtures", "report")));
      default -> throw new IllegalArgumentException("unknown gate: " + gate);
    };
  }

  private static Set<String> withDecoding(String... names) {
    Set<String> all = new java.util.HashSet<>(Arrays.asList(names));
    all.addAll(Set.of("max-tokens", "temperature", "seed", "top-k", "top-p", "token-log-every"));
    return all;
  }

  private int finish(String gate, boolean pass, Path report, String detail) {
    cli.out()
        .printf(
            "%s %s %s report=%s%n", pass ? "PASS" : "FAIL", gate, detail, report.toAbsolutePath());
    return pass ? FusionCli.PASS : FusionCli.FAIL;
  }

  int g0(CliOptions options) throws IOException {
    Path report = Path.of(options.required("report"));
    long wall = System.nanoTime();
    long cpu = FusionEnvironment.processCpuNanos();
    try (PeakRss rss = new PeakRss()) {
      Map<String, Path> paths = options.members();
      if (paths.size() < 2) {
        throw new IllegalArgumentException("G0 compares at least two members");
      }
      List<MemberConfig> members = new ArrayList<>();
      for (Map.Entry<String, Path> entry : paths.entrySet()) {
        members.add(MemberLoader.describe(entry.getKey(), entry.getValue()));
      }
      MemberConfig reference = members.getFirst();
      List<Map<String, Object>> mismatches = new ArrayList<>();
      List<String> templateDiffers = new ArrayList<>();
      for (MemberConfig member : members.subList(1, members.size())) {
        List<String> differing =
            reference.tokenizerFields().keySet().stream()
                .sorted()
                .filter(
                    key ->
                        !reference
                            .tokenizerFields()
                            .get(key)
                            .equals(member.tokenizerFields().get(key)))
                .toList();
        if (!differing.isEmpty()) {
          mismatches.add(
              FusionCli.map(
                  "member", member.name(), "versus", reference.name(), "fields", differing));
        }
        if (!java.util.Objects.equals(
            reference.chatTemplateSha256(), member.chatTemplateSha256())) {
          templateDiffers.add(member.name());
        }
      }
      boolean pass = mismatches.isEmpty();
      Map<String, Object> summary =
          FusionCli.map(
              "pass",
              pass,
              "comparedFields",
              reference.tokenizerFields().keySet().stream().sorted().toList(),
              "mismatches",
              mismatches,
              "chatTemplateDiffersFrom" + reference.name(),
              templateDiffers,
              "note",
              "chat template hashes are reported, not gated; prompts are rendered by one Java template");
      cli.writeReport(
          report,
          cli.plainReport(
              "gate-g0",
              FusionCli.emptyConfig(members, null, null, Map.of()),
              wall,
              cpu,
              rss,
              summary,
              List.of(),
              null),
          true);
      mismatches.forEach(m -> cli.out().println("MISMATCH " + m));
      return finish(
          "G0", pass, report, "members=" + members.size() + " mismatches=" + mismatches.size());
    }
  }

  int g1(CliOptions options) throws IOException {
    Path report = Path.of(options.required("report"));
    String revision = FusionCli.modelsRevision(options);
    FusionCli.LoadedData data = cli.loadData(options);
    PromptBuilder prompts = cli.prompts(options);
    DecodeSettings settings = FusionCli.settings(options);
    if (!settings.greedy()) {
      throw new IllegalArgumentException("G1 is defined at T = 0");
    }
    boolean thinking = FusionCli.thinking(options);
    Map<String, Path> paths = options.members();
    List<String> names = List.copyOf(paths.keySet());
    long wall = System.nanoTime();
    long cpu = FusionEnvironment.processCpuNanos();
    try (PeakRss rss = new PeakRss();
        MemberLoader loader =
            MemberLoader.load(
                paths,
                names,
                options.get("backend", "pure-java"),
                FusionCli.contextLength(options, settings),
                cli.mapper())) {
      ExecutorService executor =
          Executors.newFixedThreadPool(options.integer("member-threads", names.size()));
      try {
        List<FusionMember> members = names.stream().map(n -> loader.members().get(n)).toList();
        List<Map<String, Object>> rows = new ArrayList<>();
        boolean pass = true;
        for (int i = 0; i < members.size(); i++) {
          double[] unit = new double[members.size()];
          unit[i] = 1;
          for (DatasetItem item : data.selected()) {
            int[] prompt =
                members
                    .getFirst()
                    .tokenizer()
                    .encode(prompts.render(data.kind(), data.extractor().id(), item, thinking));
            DecodeResult alone = FusionDecoder.single(members.get(i), prompt, settings);
            DecodeResult fused =
                FusionDecoder.fuse(members, prompt, unit, FusionRule.POE, settings, executor);
            int[] a = alone.tokens();
            int[] b = fused.tokens();
            int divergence = Arrays.mismatch(a, b);
            boolean identical = divergence < 0 && alone.truncated() == fused.truncated();
            pass &= identical;
            rows.add(
                FusionCli.map(
                    "member",
                    names.get(i),
                    "weights",
                    Arrays.stream(unit).boxed().toList(),
                    "id",
                    item.id(),
                    "identical",
                    identical,
                    "firstDivergence",
                    divergence,
                    "aloneTokens",
                    a.length,
                    "fusedTokens",
                    b.length,
                    "aloneTokenIdsSha256",
                    Digests.sha256(Arrays.toString(a)),
                    "fusedTokenIdsSha256",
                    Digests.sha256(Arrays.toString(b)),
                    "aloneTokenIds",
                    a,
                    "fusedTokenIds",
                    b));
            cli.out()
                .printf(
                    "G1 %s e_%d %s identical=%s tokens=%d/%d%n",
                    names.get(i), i, item.id(), identical, a.length, b.length);
          }
        }
        Map<String, Object> summary =
            FusionCli.map(
                "pass", pass, "comparisons", rows.size(), "rule", "poe", "temperature", 0.0);
        ReportModel.Config config =
            FusionCli.emptyConfig(
                loader.configs(),
                data.config(),
                loader.backend(),
                FusionCli.map(
                    "maxTokens",
                    settings.maxTokens(),
                    "thinking",
                    thinking ? "on" : "off",
                    "promptsSha256",
                    prompts.sha256()));
        cli.writeReport(
            report,
            cli.plainReport("gate-g1", config, wall, cpu, rss, summary, rows, revision),
            false);
        return finish("G1", pass, report, "comparisons=" + rows.size());
      } finally {
        executor.shutdownNow();
      }
    }
  }

  int g2Dump(CliOptions options) throws IOException {
    Path outDir = Path.of(options.required("out"));
    String revision = FusionCli.modelsRevision(options);
    FusionCli.LoadedData data = cli.loadData(options);
    PromptBuilder prompts = cli.prompts(options);
    int steps = options.integer("steps", 50);
    DecodeSettings base = FusionCli.settings(options);
    DecodeSettings settings = new DecodeSettings(0f, 0, 1f, base.seed(), steps, 0);
    Map<String, Path> paths = options.members();
    List<String> names = List.copyOf(paths.keySet());
    double[] weights = new double[names.size()];
    if (options.has("weights")) {
      List<String> parts = FusionCli.csv(options.required("weights"));
      if (parts.size() != names.size()) {
        throw new IllegalArgumentException("--weights needs one value per member");
      }
      for (int i = 0; i < weights.length; i++) {
        weights[i] = Double.parseDouble(parts.get(i));
      }
    } else {
      Arrays.fill(weights, 1.0 / names.size());
    }
    FusionMath.validateWeights(weights);
    FusionRule decodeRule = FusionRule.parse(options.get("rule", "poe"));
    if (Files.exists(outDir.resolve("manifest.json"))) {
      throw new IllegalArgumentException("refusing to overwrite dump in " + outDir);
    }
    Files.createDirectories(outDir);
    long wall = System.nanoTime();
    long cpu = FusionEnvironment.processCpuNanos();
    try (PeakRss rss = new PeakRss();
        MemberLoader loader =
            MemberLoader.load(
                paths,
                names,
                options.get("backend", "pure-java"),
                FusionCli.contextLength(options, settings),
                cli.mapper())) {
      ExecutorService executor =
          Executors.newFixedThreadPool(options.integer("member-threads", names.size()));
      try {
        List<FusionMember> members = names.stream().map(n -> loader.members().get(n)).toList();
        List<int[]> promptTokens = new ArrayList<>();
        for (DatasetItem item : data.selected()) {
          promptTokens.add(
              members
                  .getFirst()
                  .tokenizer()
                  .encode(prompts.render(data.kind(), data.extractor().id(), item, false)));
        }
        Map<String, Object> manifest =
            writeDump(
                members,
                data.selected(),
                promptTokens,
                weights,
                decodeRule,
                settings,
                executor,
                outDir,
                cli.out());
        if (options.has("report")) {
          ReportModel.Config config =
              FusionCli.emptyConfig(
                  loader.configs(),
                  data.config(),
                  loader.backend(),
                  FusionCli.map("steps", steps, "outDir", outDir.toAbsolutePath().toString()));
          cli.writeReport(
              Path.of(options.required("report")),
              cli.plainReport(
                  "gate-g2-dump", config, wall, cpu, rss, manifest, List.of(), revision),
              true);
        }
        cli.out()
            .printf(
                "DONE G2-DUMP items=%d dir=%s%n", data.selected().size(), outDir.toAbsolutePath());
        return FusionCli.PASS;
      } finally {
        executor.shutdownNow();
      }
    }
  }

  /** Writes member logits and Java's fused scores for every rule, plus manifest.json. */
  static Map<String, Object> writeDump(
      List<FusionMember> members,
      List<DatasetItem> items,
      List<int[]> promptTokens,
      double[] weights,
      FusionRule decodeRule,
      DecodeSettings settings,
      ExecutorService executor,
      Path outDir,
      java.io.PrintStream out)
      throws IOException {
    List<String> names = members.stream().map(FusionMember::name).toList();
    List<Map<String, Object>> manifestItems = new ArrayList<>();
    int vocabulary = -1;
    int index = 0;
    for (DatasetItem item : items) {
      String stem = String.format(java.util.Locale.ROOT, "item%03d", index++);
      Map<String, DataOutputStream> memberFiles = new LinkedHashMap<>();
      Map<String, DataOutputStream> fusedFiles = new LinkedHashMap<>();
      Map<String, DataOutputStream> rawFiles = new LinkedHashMap<>();
      Map<String, String> memberNames = new LinkedHashMap<>();
      Map<String, String> fusedNames = new LinkedHashMap<>();
      Map<String, String> rawNames = new LinkedHashMap<>();
      for (String name : names) {
        memberNames.put(name, stem + "-" + name + ".f32");
        memberFiles.put(name, open(outDir.resolve(memberNames.get(name))));
      }
      for (FusionRule rule : FusionRule.values()) {
        fusedNames.put(rule.id(), stem + "-fused-" + rule.id() + ".f32");
        rawNames.put(rule.id(), stem + "-fusedraw-" + rule.id() + ".f32");
        fusedFiles.put(rule.id(), open(outDir.resolve(fusedNames.get(rule.id()))));
        rawFiles.put(rule.id(), open(outDir.resolve(rawNames.get(rule.id()))));
      }
      int[] recorded = {0};
      int[] prompt = promptTokens.get(index - 1);
      FusionDecoder.fuse(
          members,
          prompt,
          weights,
          decodeRule,
          settings,
          executor,
          (step, logits, usedScores) -> {
            try {
              double[][] logProbabilities = new double[logits.length][];
              for (int i = 0; i < logits.length; i++) {
                logProbabilities[i] = FusionMath.logSoftmax(logits[i]);
                writeFloats(memberFiles.get(names.get(i)), logits[i]);
              }
              for (FusionRule rule : FusionRule.values()) {
                double[] scores =
                    rule == decodeRule
                        ? usedScores
                        : FusionMath.combine(rule, logits, logProbabilities, weights);
                writeDoubles(rawFiles.get(rule.id()), scores);
                writeDoubles(fusedFiles.get(rule.id()), FusionMath.logSoftmax(scores));
              }
              recorded[0]++;
            } catch (IOException failure) {
              throw new java.io.UncheckedIOException(failure);
            }
          });
      for (DataOutputStream stream : memberFiles.values()) {
        stream.close();
      }
      for (DataOutputStream stream : fusedFiles.values()) {
        stream.close();
      }
      for (DataOutputStream stream : rawFiles.values()) {
        stream.close();
      }
      vocabulary =
          (int)
              (Files.size(outDir.resolve(memberNames.get(names.getFirst())))
                  / 4
                  / Math.max(1, recorded[0]));
      manifestItems.add(
          FusionCli.map(
              "id",
              item.id(),
              "steps",
              recorded[0],
              "members",
              memberNames,
              "fused",
              fusedNames,
              "fusedRaw",
              rawNames));
      out.printf("G2-DUMP %s steps=%d%n", item.id(), recorded[0]);
    }
    Map<String, Object> manifest =
        FusionCli.map(
            "schemaVersion",
            1,
            "vocab",
            vocabulary,
            "members",
            names,
            "weights",
            Arrays.stream(weights).boxed().toList(),
            "rules",
            Arrays.stream(FusionRule.values()).map(FusionRule::id).toList(),
            "decodeRule",
            decodeRule.id(),
            "dtype",
            "float32-le",
            "items",
            manifestItems);
    Files.write(
        outDir.resolve("manifest.json"),
        new com.fasterxml.jackson.databind.ObjectMapper()
            .writerWithDefaultPrettyPrinter()
            .writeValueAsBytes(manifest));
    return manifest;
  }

  private static DataOutputStream open(Path path) throws IOException {
    OutputStream stream = Files.newOutputStream(path);
    return new DataOutputStream(new BufferedOutputStream(stream, 1 << 20));
  }

  private static void writeFloats(DataOutputStream out, float[] values) throws IOException {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    buffer.asFloatBuffer().put(values);
    out.write(buffer.array());
  }

  private static void writeDoubles(DataOutputStream out, double[] values) throws IOException {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (double value : values) {
      buffer.putFloat((float) value);
    }
    out.write(buffer.array());
  }

  int g3(CliOptions options) throws IOException {
    Path report = Path.of(options.required("report"));
    long wall = System.nanoTime();
    long cpu = FusionEnvironment.processCpuNanos();
    try (PeakRss rss = new PeakRss()) {
      List<Map<String, Object>> rows = new ArrayList<>();
      boolean pass = true;
      for (Path path : options.fileList("reports")) {
        JsonNode root = cli.mapper().readTree(path.toFile());
        JsonNode g3 = root.path("summary").path("g3");
        if (g3.isMissingNode() || g3.isNull()) {
          rows.add(FusionCli.map("report", path.toString(), "fused", false));
          continue;
        }
        boolean flagged = g3.path("flagged").asBoolean();
        pass &= !flagged;
        rows.add(
            FusionCli.map(
                "report", path.toString(),
                "label", root.path("config").path("arm").path("label").asText(null),
                "dataset", root.path("config").path("dataset").path("name").asText(null),
                "fused", true,
                "steps", g3.path("steps").asLong(),
                "agreementRate", g3.path("agreementRate").asDouble(),
                "fusedEqualsNoMemberRate", g3.path("fusedEqualsNoMemberRate").asDouble(),
                "fusedEqualsMemberRate", g3.path("fusedEqualsMemberRate"),
                "flagged", flagged));
        cli.out()
            .printf(
                "G3 %s agreement=%.4f flagged=%s%n",
                path.getFileName(), g3.path("agreementRate").asDouble(), flagged);
      }
      cli.writeReport(
          report,
          cli.plainReport(
              "gate-g3",
              FusionCli.emptyConfig(List.of(), null, null, Map.of()),
              wall,
              cpu,
              rss,
              FusionCli.map(
                  "pass", pass, "rule", "flag when member argmax agreement is exactly 100%"),
              rows,
              null),
          true);
      return finish("G3", pass, report, "reports=" + rows.size());
    }
  }

  int g4(CliOptions options) throws IOException {
    Path report = Path.of(options.required("report"));
    long wall = System.nanoTime();
    long cpu = FusionEnvironment.processCpuNanos();
    try (PeakRss rss = new PeakRss()) {
      FusionCli.LoadedData data = cli.loadData(options);
      G4GoldCheck.Result result =
          G4GoldCheck.evaluate(data.kind().id(), data.extractor(), data.selected());
      cli.writeReport(
          report,
          cli.plainReport(
              "gate-g4",
              FusionCli.emptyConfig(List.of(), data.config(), null, Map.of()),
              wall,
              cpu,
              rss,
              result,
              result.failures(),
              null),
          true);
      return finish(
          "G4",
          result.pass(),
          report,
          data.kind().id() + " " + result.passed() + "/" + result.total());
    }
  }

  int g5(CliOptions options) throws IOException {
    Path report = Path.of(options.required("report"));
    long wall = System.nanoTime();
    long cpu = FusionEnvironment.processCpuNanos();
    try (PeakRss rss = new PeakRss()) {
      List<Map<String, Object>> rows = new ArrayList<>();
      boolean pass = true;
      for (Path path : options.fileList("reports")) {
        JsonNode root = cli.mapper().readTree(path.toFile());
        JsonNode summary = root.path("summary");
        double truncation = summary.path("truncationRate").asDouble();
        boolean flagged = truncation > SummaryBuilder.G5_TRUNCATION_LIMIT;
        pass &= !flagged;
        rows.add(
            FusionCli.map(
                "report", path.toString(),
                "label", root.path("config").path("arm").path("label").asText(null),
                "dataset", root.path("config").path("dataset").path("name").asText(null),
                "maxTokens", root.path("config").path("decoding").path("maxTokens").asInt(),
                "truncationRate", truncation,
                "extractionFailureRate", summary.path("extractionFailureRate").asDouble(),
                "p95TraceTokens", summary.path("p95TraceTokens").asDouble(),
                "rerunWithHigherCap", flagged));
        cli.out()
            .printf(
                "G5 %s truncation=%.4f extraction-failures=%.4f flagged=%s%n",
                path.getFileName(),
                truncation,
                summary.path("extractionFailureRate").asDouble(),
                flagged);
      }
      cli.writeReport(
          report,
          cli.plainReport(
              "gate-g5",
              FusionCli.emptyConfig(List.of(), null, null, Map.of()),
              wall,
              cpu,
              rss,
              FusionCli.map("pass", pass, "truncationLimit", SummaryBuilder.G5_TRUNCATION_LIMIT),
              rows,
              null),
          true);
      return finish("G5", pass, report, "reports=" + rows.size());
    }
  }

  int g6(CliOptions options) throws IOException {
    Path report = Path.of(options.required("report"));
    long wall = System.nanoTime();
    long cpu = FusionEnvironment.processCpuNanos();
    try (PeakRss rss = new PeakRss()) {
      List<G6Fixtures.Result> results = new ArrayList<>();
      boolean pass = true;
      for (DatasetKind dataset : List.of(DatasetKind.GSM8K, DatasetKind.ARC, DatasetKind.MATH500)) {
        List<G6Fixtures.Fixture> fixtures =
            options.has("fixtures")
                ? G6Fixtures.load(Path.of(options.required("fixtures")), dataset)
                : G6Fixtures.loadBundled(dataset);
        G6Fixtures.Result result = G6Fixtures.evaluate(dataset, fixtures);
        pass &= result.pass() && result.total() == 30;
        results.add(result);
        cli.out().printf("G6 %s %d/%d%n", dataset.id(), result.passed(), result.total());
      }
      cli.writeReport(
          report,
          cli.plainReport(
              "gate-g6",
              FusionCli.emptyConfig(
                  List.of(),
                  null,
                  null,
                  FusionCli.map("fixtures", options.get("fixtures", "bundled"))),
              wall,
              cpu,
              rss,
              FusionCli.map("pass", pass),
              results,
              null),
          true);
      return finish("G6", pass, report, "datasets=" + results.size());
    }
  }
}
