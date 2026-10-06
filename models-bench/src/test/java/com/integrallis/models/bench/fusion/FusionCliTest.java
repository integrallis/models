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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import com.integrallis.models.bench.fusion.ReportModel.Item;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FusionCliTest {
  private final ByteArrayOutputStream captured = new ByteArrayOutputStream();
  private final FusionCli cli =
      new FusionCli(new PrintStream(captured, true, StandardCharsets.UTF_8));
  private final ObjectMapper mapper = new ObjectMapper();

  private static Path resource(String name) throws Exception {
    return Path.of(
        FusionCliTest.class
            .getResource("/com/integrallis/models/bench/fusion/gold/" + name)
            .toURI());
  }

  @Test
  void optionParsingIsStrict() {
    assertThatThrownBy(() -> CliOptions.parse(new String[] {"--x"}, 0, Set.of("x")))
        .hasMessageContaining("pairs");
    assertThatThrownBy(() -> CliOptions.parse(new String[] {"--y", "1"}, 0, Set.of("x")))
        .hasMessageContaining("unknown option");
    assertThatThrownBy(
            () -> CliOptions.parse(new String[] {"--x", "1", "--x", "2"}, 0, Set.of("x")))
        .hasMessageContaining("duplicate");
    assertThat(cli.execute(new String[] {"no-such-action"})).isEqualTo(FusionCli.USAGE);
    assertThat(cli.execute(new String[] {"run", "--arm", "member:A"})).isEqualTo(FusionCli.USAGE);
  }

  @Test
  void gateG4PassesOnPinnedGoldAnswersThroughTheCommand(@TempDir Path directory) throws Exception {
    Path report = directory.resolve("g4.json");
    int status =
        cli.execute(
            new String[] {
              "gate",
              "g4",
              "--dataset",
              "math500",
              "--data",
              resource("math500-test.jsonl").toString(),
              "--report",
              report.toString()
            });
    assertThat(status).isEqualTo(FusionCli.PASS);
    JsonNode root = mapper.readTree(report.toFile());
    assertThat(root.path("kind").asText()).isEqualTo("gate-g4");
    assertThat(root.path("summary").path("passed").asInt()).isEqualTo(500);
    assertThat(root.path("environment").path("javaVersionOutput").asText()).contains("version");
    assertThat(captured.toString(StandardCharsets.UTF_8)).contains("PASS G4");
  }

  @Test
  void gateG6RunsTheBundledFixtures(@TempDir Path directory) {
    Path report = directory.resolve("g6.json");
    assertThat(cli.execute(new String[] {"gate", "g6", "--report", report.toString()}))
        .isEqualTo(FusionCli.PASS);
    assertThat(captured.toString(StandardCharsets.UTF_8))
        .contains("G6 gsm8k 30/30", "G6 arc 30/30", "G6 math500 30/30");
  }

  @Test
  void tokenizerIdentityNamesTheDifferingFieldAndIgnoresTheChatTemplate() {
    GgufMetadata base = metadata(List.of("a", "b"), List.of("a b"), "template-1", 1);
    GgufMetadata sameVocabOtherTemplate =
        metadata(List.of("a", "b"), List.of("a b"), "template-2", 1);
    GgufMetadata otherMerges = metadata(List.of("a", "b"), List.of("b a"), "template-1", 1);
    GgufMetadata otherEos = metadata(List.of("a", "b"), List.of("a b"), "template-1", 0);
    TokenizerIdentity reference = TokenizerIdentity.of(base);
    assertThat(reference.differingFields(TokenizerIdentity.of(sameVocabOtherTemplate))).isEmpty();
    assertThat(reference.chatTemplateSha256())
        .isNotEqualTo(TokenizerIdentity.of(sameVocabOtherTemplate).chatTemplateSha256());
    assertThat(reference.differingFields(TokenizerIdentity.of(otherMerges)))
        .containsExactly("tokenizer.ggml.merges");
    assertThat(reference.differingFields(TokenizerIdentity.of(otherEos)))
        .containsExactly("tokenizer.ggml.eos_token_id");
    assertThat(reference.tokenizerSha256())
        .isNotEqualTo(TokenizerIdentity.of(otherMerges).tokenizerSha256());
  }

  private static GgufMetadata metadata(
      List<String> tokens, List<String> merges, String template, int eos) {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("tokenizer.ggml.model", new GgufMetadataValue.StringValue("gpt2"));
    entries.put("tokenizer.ggml.tokens", strings(tokens));
    entries.put("tokenizer.ggml.merges", strings(merges));
    entries.put(
        "tokenizer.ggml.token_type",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.INT32,
            List.of(new GgufMetadataValue.Int32Value(1), new GgufMetadataValue.Int32Value(3))));
    entries.put("tokenizer.ggml.eos_token_id", new GgufMetadataValue.Uint32Value(eos));
    entries.put("tokenizer.chat_template", new GgufMetadataValue.StringValue(template));
    return new GgufMetadata(entries);
  }

  private static GgufMetadataValue strings(List<String> values) {
    return new GgufMetadataValue.ArrayValue(
        GgufValueType.STRING,
        values.stream()
            .map(v -> (GgufMetadataValue) new GgufMetadataValue.StringValue(v))
            .toList());
  }

  @Test
  void memberReportsRoundTripIntoVotingAndTheSummaryComputesOracleAndStopRule(
      @TempDir Path directory) throws Exception {
    DatasetItem item = new DatasetItem("toy-1", "12 plus 3", "7", List.of());
    DecodeSettings greedy = new DecodeSettings(0f, 0, 1f, 3L, 24, 0);
    ExecutorService pool = Executors.newFixedThreadPool(3);
    try {
      Map<String, FusionMember> members = new LinkedHashMap<>();
      members.put("A", new ToyMember("A", 0.7, 0));
      members.put("B", new ToyMember("B", 1.3, 5));
      members.put("C", new ToyMember("C", 2.1, -2));
      List<Path> reports = new java.util.ArrayList<>();
      for (String name : List.of("A", "B", "C")) {
        ArmRunner runner =
            new ArmRunner(
                new ArmRunner.Context(
                    ArmSpec.parse("member:" + name),
                    DatasetKind.GSM8K,
                    AnswerExtractor.forDataset(DatasetKind.GSM8K),
                    PromptBuilder.bundled(),
                    greedy,
                    false,
                    false,
                    false,
                    null,
                    false,
                    pool,
                    members,
                    Map.of()));
        List<Item> items = runner.runAll(List.of(item), line -> {});
        Path path = directory.resolve(name + ".json");
        cli.writeReport(
            path,
            ReportFixtures.armReport(
                mapper, items, name, "member:" + name, "member", List.of(name), "sha-of-data"),
            false);
        reports.add(path);
      }
      Path fused = directory.resolve("F-tuned.json");
      ArmRunner fusedRunner =
          new ArmRunner(
              new ArmRunner.Context(
                  ArmSpec.parse("fuse:A+B+C:poe:uniform"),
                  DatasetKind.GSM8K,
                  AnswerExtractor.forDataset(DatasetKind.GSM8K),
                  PromptBuilder.bundled(),
                  greedy,
                  false,
                  false,
                  false,
                  null,
                  false,
                  pool,
                  members,
                  Map.of()));
      cli.writeReport(
          fused,
          ReportFixtures.armReport(
              mapper,
              fusedRunner.runAll(List.of(item), l -> {}),
              "F-tuned",
              "fuse:A+B+C:poe:uniform",
              "fuse",
              List.of("A", "B", "C"),
              "sha-of-data"),
          false);

      ReusedOutputs.Loaded loaded =
          ReusedOutputs.load(
              reports,
              mapper,
              "sha-of-data",
              false,
              new DecodeSettings(0f, 0, 1f, 3L, 24, 0),
              "e".repeat(64));
      assertThat(loaded.byMember()).containsOnlyKeys("A", "B", "C");
      assertThat(loaded.byMember().get("A").get("toy-1").output().reused()).isTrue();
      assertThatThrownBy(
              () ->
                  ReusedOutputs.load(reports, mapper, "other-data", false, greedy, "e".repeat(64)))
          .hasMessageContaining("different dataset");

      Path summary = directory.resolve("summary.json");
      List<String> all =
          List.of(
              reports.get(0).toString(),
              reports.get(1).toString(),
              reports.get(2).toString(),
              fused.toString());
      assertThat(
              cli.execute(
                  new String[] {
                    "summarize", "--reports", String.join(",", all), "--report", summary.toString()
                  }))
          .isEqualTo(FusionCli.PASS);
      JsonNode stop = mapper.readTree(summary.toFile()).path("summary").path("stopRule");
      assertThat(stop.path("verdict").asText()).isIn("TRIGGERED", "NOT-TRIGGERED");

      Path g3 = directory.resolve("g3.json");
      int g3Status =
          cli.execute(
              new String[] {
                "gate", "g3", "--reports", fused.toString(), "--report", g3.toString()
              });
      assertThat(g3Status).isIn(FusionCli.PASS, FusionCli.FAIL);
      Path g5 = directory.resolve("g5.json");
      cli.execute(
          new String[] {
            "gate", "g5", "--reports", String.join(",", all), "--report", g5.toString()
          });
      assertThat(mapper.readTree(g5.toFile()).path("items")).hasSize(4);
    } finally {
      pool.shutdownNow();
    }
  }

  @Test
  void resultsLogRendersEveryJsonLine(@TempDir Path directory) throws Exception {
    Path in = directory.resolve("results-log.jsonl");
    Files.writeString(
        in,
        "{\"date\":\"2026-09-17T00:00:00Z\",\"modelsCommit\":\"abc\",\"host\":\"h\",\"phase\":\"gates\",\"arm\":null,\"dataset\":null,\"evidencePath\":\"/e\",\"outcome\":\"PHASE-DONE\",\"provenance\":\"measured\"}\n");
    Path out = directory.resolve("log.md");
    assertThat(
            cli.execute(
                new String[] {"results-log", "--in", in.toString(), "--report", out.toString()}))
        .isEqualTo(FusionCli.PASS);
    assertThat(Files.readString(out))
        .contains("| 2026-09-17T00:00:00Z | abc | h | gates |  |  | /e | PHASE-DONE | measured |");
  }

  @Test
  void g2DumpWritesMemberLogitsAndFusedScoresThatRecomputeExactly(@TempDir Path directory)
      throws Exception {
    List<FusionMember> members = List.of(new ToyMember("A", 0.7, 0), new ToyMember("B", 1.3, 5));
    DatasetItem item = new DatasetItem("toy-1", "q", "1", List.of());
    ExecutorService pool = Executors.newFixedThreadPool(2);
    try {
      Map<String, Object> manifest =
          GateCommands.writeDump(
              members,
              List.of(item),
              List.of(new int[] {1, 2, 3}),
              new double[] {0.3, 0.7},
              FusionRule.POE,
              new DecodeSettings(0f, 0, 1f, 1L, 5, 0),
              pool,
              directory,
              new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
      assertThat(manifest.get("vocab")).isEqualTo(ToyMember.VOCAB);
      JsonNode root = mapper.readTree(directory.resolve("manifest.json").toFile());
      int steps = root.path("items").get(0).path("steps").asInt();
      assertThat(steps).isBetween(1, 5);
      float[] a =
          read(directory.resolve(root.path("items").get(0).path("members").path("A").asText()));
      float[] b =
          read(directory.resolve(root.path("items").get(0).path("members").path("B").asText()));
      float[] mixture =
          read(directory.resolve(root.path("items").get(0).path("fused").path("mixture").asText()));
      assertThat(a).hasSize(steps * ToyMember.VOCAB);
      float[] rowA = java.util.Arrays.copyOfRange(a, 0, ToyMember.VOCAB);
      float[] rowB = java.util.Arrays.copyOfRange(b, 0, ToyMember.VOCAB);
      double[] expected =
          FusionMath.logSoftmax(
              FusionMath.combine(
                  FusionRule.MIXTURE,
                  new float[][] {rowA, rowB},
                  new double[][] {FusionMath.logSoftmax(rowA), FusionMath.logSoftmax(rowB)},
                  new double[] {0.3, 0.7}));
      for (int t = 0; t < ToyMember.VOCAB; t++) {
        assertThat((double) mixture[t]).isCloseTo(expected[t], within(1e-5));
      }
    } finally {
      pool.shutdownNow();
    }
  }

  private static float[] read(Path path) throws Exception {
    byte[] bytes = Files.readAllBytes(path);
    float[] values = new float[bytes.length / 4];
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
    return values;
  }
}
