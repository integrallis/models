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
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.models.bench.fusion.ReportModel.ArmConfig;
import com.integrallis.models.bench.fusion.ReportModel.BackendConfig;
import com.integrallis.models.bench.fusion.ReportModel.Config;
import com.integrallis.models.bench.fusion.ReportModel.DatasetConfig;
import com.integrallis.models.bench.fusion.ReportModel.DecodingConfig;
import com.integrallis.models.bench.fusion.ReportModel.Item;
import com.integrallis.models.bench.fusion.ReportModel.MemberConfig;
import com.integrallis.models.bench.fusion.ReportModel.Report;
import com.integrallis.models.bench.fusion.ReportModel.Summary;
import com.integrallis.models.bench.fusion.ReportModel.Timings;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * {@code logit-fusion} subcommand: arms, gates, tuning, calibration, freezing and summaries.
 *
 * <p>Every command prints a final {@code PASS}, {@code FAIL} or {@code DONE} line and exits 0 on
 * PASS/DONE, 1 on FAIL and 2 on a usage or configuration error.
 */
public final class FusionCli {
  static final int PASS = 0;
  static final int FAIL = 1;
  static final int USAGE = 2;

  static final Set<String> RUN_OPTIONS =
      Set.of(
          "members",
          "backend",
          "models-revision",
          "report",
          "arm",
          "arm-label",
          "dataset",
          "data",
          "data-manifest",
          "scorer",
          "id-field",
          "question-field",
          "answer-field",
          "choices-field",
          "thinking",
          "max-tokens",
          "temperature",
          "seed",
          "top-k",
          "top-p",
          "offset",
          "limit",
          "member-threads",
          "context-length",
          "token-log-every",
          "frozen",
          "frozen-sha256",
          "member-reports",
          "arc-scoring",
          "rerank-score",
          "prompts",
          "pilot");

  static final String USAGE_TEXT =
      """
      usage: logit-fusion <action> --name value ...
        run            --arm <spec> --members A=a.gguf,... --dataset gsm8k|arc|math500|generic --data f.jsonl
                       --models-revision <sha> --report r.json [--thinking on|off] [--max-tokens N] ...
        gate g0        --members ... --report r.json
        gate g1        --members ... --dataset ... --data ... --models-revision <sha> --report r.json [--limit 20]
        gate g2-dump   --members ... --dataset ... --data ... --models-revision <sha> --out <dir> [--limit 20 --steps 50]
        gate g3|g5     --reports a.json,b.json --report r.json
        gate g4        --dataset ... --data ... --report r.json
        gate g6        --report r.json [--fixtures dir]
        tune-weights   --members ... --arm-members A+B+C --rule poe --dataset gsm8k --data dev.jsonl
                       --models-revision <sha> --work-dir dir --report tuning.json [--grid-step 0.1]
        calibrate      --member-reports a.json,... --dataset gsm8k --report calibration.json
        freeze         --tuning t1.json,... --calibration c.json --best-member-reports a.json,...
                       --dataset gsm8k --data dev.jsonl --report frozen.json
        summarize      --reports a.json,... --report summary.json [--member-labels A,B,C --fused-label F-tuned]
        results-log    --in results-log.jsonl --report RESULTS-LOG.generated.md
      arm specs: member:<m> | fuse:<m1+m2..>:<poe|mixture|article>:<uniform|tuned|w1,w2..>
                 vote|vote-consist|vote-conf:<m1+m2..>:<tieBreak|best> | sc:<m>:<k> | rerank:<m1+m2..>:<weights>
      """;

  private final PrintStream out;
  private final ObjectMapper mapper;

  FusionCli(PrintStream out) {
    this.out = out;
    this.mapper = new ObjectMapper();
  }

  /** Entry point used by {@code InferenceBenchmarkCli}. */
  public static int run(String[] args) {
    return new FusionCli(System.out).execute(args);
  }

  int execute(String[] args) {
    try {
      if (args.length == 0) {
        throw new IllegalArgumentException("an action is required");
      }
      return switch (args[0]) {
        case "run" -> runArm(CliOptions.parse(args, 1, RUN_OPTIONS));
        case "gate" -> {
          if (args.length < 2) {
            throw new IllegalArgumentException("gate requires g0|g1|g2-dump|g3|g4|g5|g6");
          }
          yield new GateCommands(this).run(args[1], args);
        }
        case "tune-weights" -> new TuningCommands(this).tuneWeights(args);
        case "calibrate" -> new TuningCommands(this).calibrate(args);
        case "freeze" -> new TuningCommands(this).freeze(args);
        case "summarize" -> new SummaryCommands(this).summarize(args);
        case "results-log" -> new SummaryCommands(this).resultsLog(args);
        default -> throw new IllegalArgumentException("unknown action: " + args[0]);
      };
    } catch (IllegalArgumentException | IllegalStateException failure) {
      System.err.println("ERROR " + failure.getMessage());
      System.err.print(USAGE_TEXT);
      return USAGE;
    } catch (IOException failure) {
      System.err.println("ERROR " + failure);
      return USAGE;
    }
  }

  PrintStream out() {
    return out;
  }

  ObjectMapper mapper() {
    return mapper;
  }

  /** Decoding settings shared by run, gates and tuning. */
  static DecodeSettings settings(CliOptions options) {
    return new DecodeSettings(
        options.decimal("temperature", 0f),
        options.integer("top-k", 0),
        options.decimal("top-p", 1f),
        options.longValue("seed", 20260917L),
        options.integer("max-tokens", 1024),
        options.integer("token-log-every", 1));
  }

  static boolean thinking(CliOptions options) {
    String value = options.get("thinking", "off");
    if (!"on".equals(value) && !"off".equals(value)) {
      throw new IllegalArgumentException("--thinking must be on or off");
    }
    return "on".equals(value);
  }

  static int contextLength(CliOptions options, DecodeSettings settings) {
    return options.integer("context-length", settings.maxTokens() + 2048);
  }

  record LoadedData(
      DatasetKind kind,
      AnswerExtractor extractor,
      List<DatasetItem> all,
      List<DatasetItem> selected,
      DatasetConfig config) {}

  LoadedData loadData(CliOptions options) throws IOException {
    DatasetKind kind = DatasetKind.parse(options.required("dataset"));
    AnswerExtractor extractor =
        kind == DatasetKind.GENERIC
            ? AnswerExtractor.forScorer(options.required("scorer"))
            : AnswerExtractor.forDataset(kind);
    Path data = options.existingFile("data");
    FieldMapping fields =
        new FieldMapping(
            options.get("id-field", "id"),
            options.get("question-field", "question"),
            options.get("answer-field", "answer"),
            options.get("choices-field", "choices"));
    List<DatasetItem> all = DatasetItem.loadJsonl(data, kind, fields);
    int offset = options.integer("offset", 0);
    int limit = options.integer("limit", 0);
    if (offset < 0 || limit < 0 || offset > all.size()) {
      throw new IllegalArgumentException(
          "--offset/--limit out of range for " + all.size() + " items");
    }
    int end = limit == 0 ? all.size() : Math.min(all.size(), offset + limit);
    List<DatasetItem> selected = all.subList(offset, end);
    String sha = Digests.sha256(data);
    String hfRepo = null;
    String hfRevision = null;
    String manifestPath = null;
    String manifestSha = null;
    Boolean matches = null;
    String devIds = null;
    if (options.has("data-manifest")) {
      Path manifest = options.existingFile("data-manifest");
      byte[] bytes = Files.readAllBytes(manifest);
      JsonNode root = mapper.readTree(bytes);
      manifestPath = manifest.toAbsolutePath().toString();
      manifestSha = Digests.sha256(bytes);
      String fileName = String.valueOf(data.toAbsolutePath().getFileName());
      for (Map.Entry<String, JsonNode> entry : root.path("datasets").properties()) {
        JsonNode dataset = entry.getValue();
        if (fileName.equals(dataset.path("output").asText())) {
          hfRepo = dataset.path("repo").asText(null);
          hfRevision = dataset.path("revision").asText(null);
          matches = sha.equals(dataset.path("outputSha256").asText());
        }
        if ("gsm8k-dev".equals(entry.getKey())) {
          devIds = dataset.path("idsSha256").asText(null);
        }
      }
      if (Boolean.FALSE.equals(matches)) {
        throw new IllegalArgumentException(
            "dataset " + data + " sha256 " + sha + " does not match the data manifest");
      }
    }
    DatasetConfig config =
        new DatasetConfig(
            kind.id(),
            data.toAbsolutePath().toString(),
            sha,
            hfRepo,
            hfRevision,
            manifestPath,
            manifestSha,
            matches,
            devIds,
            offset,
            limit,
            selected.size(),
            DatasetItem.idsSha256(selected));
    return new LoadedData(kind, extractor, all, selected, config);
  }

  PromptBuilder prompts(CliOptions options) throws IOException {
    return options.has("prompts")
        ? PromptBuilder.load(options.existingFile("prompts"))
        : PromptBuilder.bundled();
  }

  static String modelsRevision(CliOptions options) {
    String revision = options.required("models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be a 40-character git sha");
    }
    return revision;
  }

  int runArm(CliOptions options) throws IOException {
    ArmSpec arm = ArmSpec.parse(options.required("arm"));
    String label = options.get("arm-label", null);
    String revision = modelsRevision(options);
    Path reportPath = Path.of(options.required("report"));
    LoadedData data = loadData(options);
    PromptBuilder prompts = prompts(options);
    DecodeSettings settings = settings(options);
    boolean thinking = thinking(options);
    boolean pilot = options.bool("pilot", false);
    String arcScoring = options.get("arc-scoring", "generative");
    if (!Set.of("generative", "loglik").contains(arcScoring)) {
      throw new IllegalArgumentException("--arc-scoring must be generative or loglik");
    }
    boolean loglik = "loglik".equals(arcScoring);
    if (loglik && data.kind() != DatasetKind.ARC) {
      throw new IllegalArgumentException("--arc-scoring loglik applies to --dataset arc only");
    }
    String rerankScore = options.get("rerank-score", "sum");
    if (!Set.of("sum", "mean").contains(rerankScore)) {
      throw new IllegalArgumentException("--rerank-score must be sum or mean");
    }
    FrozenTuning frozen = null;
    if (options.has("frozen")) {
      frozen =
          FrozenTuning.load(options.existingFile("frozen"), options.get("frozen-sha256", null));
      frozen.requireDisjoint(data.selected());
    } else if (options.has("frozen-sha256")) {
      throw new IllegalArgumentException("--frozen-sha256 given without --frozen");
    }
    if (arm.requiresFrozen() && frozen == null && !pilot) {
      throw new IllegalArgumentException(
          "arm "
              + arm.canonical()
              + " requires --frozen and --frozen-sha256 (tuned before any test run)");
    }
    if (thinking && arm.kind() == ArmSpec.Kind.FUSE) {
      out.println("WARNING the protocol runs fused decoding arms with thinking off only");
    }

    ReusedOutputs.Loaded reused =
        ReusedOutputs.load(
            options.fileList("member-reports"),
            mapper,
            data.config().sha256(),
            thinking,
            settings,
            prompts.sha256());
    if (!reused.byMember().isEmpty()
        && !(arm.kind().isVote() || arm.kind() == ArmSpec.Kind.RERANK)) {
      throw new IllegalArgumentException("--member-reports applies to vote and rerank arms only");
    }
    List<String> toLoad = new ArrayList<>();
    for (String name : arm.members()) {
      if (arm.kind() == ArmSpec.Kind.RERANK || !reused.byMember().containsKey(name)) {
        toLoad.add(name);
      }
    }
    int contextLength = contextLength(options, settings);
    Map<String, Path> memberPaths =
        toLoad.isEmpty() && !options.has("members") ? Map.of() : options.members();
    String backendName = options.get("backend", "pure-java");

    long wallStart = System.nanoTime();
    long cpuStart = FusionEnvironment.processCpuNanos();
    try (PeakRss rss = new PeakRss();
        MemberLoader loader =
            MemberLoader.load(memberPaths, toLoad, backendName, contextLength, mapper)) {
      int threads = Math.max(1, options.integer("member-threads", Math.max(1, toLoad.size())));
      ExecutorService executor = Executors.newFixedThreadPool(threads);
      try {
        ArmRunner runner =
            new ArmRunner(
                new ArmRunner.Context(
                    arm,
                    data.kind(),
                    data.extractor(),
                    prompts,
                    settings,
                    thinking,
                    loglik,
                    "mean".equals(rerankScore),
                    frozen,
                    pilot,
                    executor,
                    loader.members(),
                    reused.byMember()));
        ArmRunner.Resolution resolution = runner.resolution();
        out.printf(
            "START arm=%s label=%s dataset=%s items=%d thinking=%s weights=%s substitution=%s%n",
            arm.canonical(),
            label,
            data.kind().id(),
            data.selected().size(),
            thinking ? "on" : "off",
            Arrays.toString(resolution.weights()),
            resolution.pilotSubstitution());
        List<Item> items = runner.runAll(data.selected(), out::println);
        Summary summary = SummaryBuilder.summarize(items);
        List<MemberConfig> memberConfigs = new ArrayList<>(loader.configs());
        for (MemberConfig config : reused.members()) {
          if (memberConfigs.stream().noneMatch(m -> m.name().equals(config.name()))) {
            memberConfigs.add(config);
          }
        }
        List<Long> seeds = new ArrayList<>();
        for (int j = 0; j < arm.samples(); j++) {
          seeds.add(settings.seed() + j);
        }
        ArmConfig armConfig =
            new ArmConfig(
                label,
                options.required("arm"),
                arm.canonical(),
                arm.kind().id(),
                arm.members(),
                arm.rule() == null ? null : arm.rule().id(),
                arm.kind() == ArmSpec.Kind.FUSE || arm.kind() == ArmSpec.Kind.RERANK
                    ? Arrays.stream(resolution.weights()).boxed().toList()
                    : null,
                resolution.weightSource(),
                resolution.tieBreak(),
                arm.samples(),
                frozen == null ? null : frozen.path(),
                frozen == null ? null : frozen.sha256(),
                pilot,
                resolution.pilotSubstitution(),
                data.extractor().id(),
                reused.paths(),
                reused.sha256s());
        DecodingConfig decoding =
            new DecodingConfig(
                settings.temperature(),
                settings.topK(),
                settings.topP(),
                seeds,
                settings.maxTokens(),
                thinking ? "on" : "off",
                PromptBuilder.template(thinking).id(),
                prompts.templateSha256(thinking),
                prompts.source(),
                prompts.sha256(),
                threads,
                contextLength,
                settings.tokenLogEvery(),
                arcScoring,
                rerankScore,
                arm.kind() == ArmSpec.Kind.RERANK ? "off" : null,
                "argmax over fused double scores, lowest token id on ties; Sampler when T > 0");
        Config config =
            new Config(
                armConfig, decoding, memberConfigs, data.config(), loader.backend(), Map.of());
        Report report =
            new Report(
                ReportModel.SCHEMA_VERSION,
                "arm",
                Instant.now().toString(),
                config,
                FusionEnvironment.capture(revision, mapper),
                timings(wallStart, cpuStart, rss, loader.loadMillis()),
                summary,
                items);
        writeReport(reportPath, report, false);
        out.printf(
            "DONE arm %s accuracy=%.4f (%d/%d) truncation=%.4f tokens/s=%.2f core-s=%.1f report=%s%n",
            label == null ? arm.canonical() : label,
            summary.accuracy(),
            summary.correct(),
            summary.items(),
            summary.truncationRate(),
            summary.tokensPerSecond(),
            summary.coreSeconds(),
            reportPath.toAbsolutePath());
        return PASS;
      } finally {
        executor.shutdownNow();
      }
    }
  }

  static Timings timings(long wallStart, long cpuStart, PeakRss rss, long loadMillis) {
    long cpuEnd = FusionEnvironment.processCpuNanos();
    return new Timings(
        (System.nanoTime() - wallStart) / 1_000_000L,
        cpuStart < 0 ? -1 : (cpuEnd - cpuStart) / 1e9,
        rss.peakBytes(),
        rss.method(),
        loadMillis);
  }

  /** Validates against {@link ReportSchema} and writes; refuses to overwrite an existing report. */
  JsonNode writeReport(Path path, Report report, boolean indent) throws IOException {
    ObjectNode tree = mapper.valueToTree(report);
    List<String> violations = ReportSchema.violations(tree);
    if (!violations.isEmpty()) {
      throw new IllegalStateException("report violates schema v1: " + violations);
    }
    if (Files.exists(path)) {
      throw new IllegalArgumentException("refusing to overwrite existing report " + path);
    }
    Path parent = path.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    ObjectMapper writer =
        indent ? mapper.copy().enable(SerializationFeature.INDENT_OUTPUT) : mapper;
    Files.write(path, writer.writeValueAsBytes(tree));
    return tree;
  }

  /** A report for commands that do not run an arm. */
  Report plainReport(
      String kind,
      Config config,
      long wallStart,
      long cpuStart,
      PeakRss rss,
      Object summary,
      List<?> items,
      String revision) {
    return new Report(
        ReportModel.SCHEMA_VERSION,
        kind,
        Instant.now().toString(),
        config,
        FusionEnvironment.capture(revision, mapper),
        timings(wallStart, cpuStart, rss, 0),
        summary,
        items);
  }

  static Config emptyConfig(
      List<MemberConfig> members,
      DatasetConfig dataset,
      BackendConfig backend,
      Map<String, Object> extra) {
    return new Config(null, null, members, dataset, backend, extra);
  }

  static List<String> csv(String value) {
    Set<String> parts = new LinkedHashSet<>();
    for (String part : value.split(",")) {
      if (!part.isBlank()) {
        parts.add(part.strip());
      }
    }
    return List.copyOf(parts);
  }

  static String readString(Path path) throws IOException {
    return Files.readString(path, StandardCharsets.UTF_8);
  }

  static Map<String, Object> map(Object... keyValues) {
    Map<String, Object> result = new LinkedHashMap<>();
    for (int i = 0; i < keyValues.length; i += 2) {
      result.put((String) keyValues[i], keyValues[i + 1]);
    }
    return result;
  }
}
