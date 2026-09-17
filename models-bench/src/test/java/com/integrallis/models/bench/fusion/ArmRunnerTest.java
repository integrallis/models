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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.models.bench.fusion.ReportModel.Item;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class ArmRunnerTest {
  private final ExecutorService pool = Executors.newFixedThreadPool(3);
  private static final DatasetItem ITEM = new DatasetItem("toy-1", "12 plus 3", "7", List.of());
  private static final DecodeSettings GREEDY = new DecodeSettings(0f, 0, 1f, 3L, 24, 2);

  @AfterEach
  void shutdown() {
    pool.shutdownNow();
  }

  private static Map<String, FusionMember> members() {
    Map<String, FusionMember> members = new LinkedHashMap<>();
    members.put("A", new ToyMember("A", 0.7, 0));
    members.put("B", new ToyMember("B", 1.3, 5));
    members.put("C", new ToyMember("C", 2.1, -2));
    return members;
  }

  private ArmRunner runner(
      String spec,
      DatasetKind dataset,
      DecodeSettings settings,
      boolean loglik,
      boolean pilot,
      Map<String, FusionMember> members,
      Map<String, Map<String, ReusedOutputs.Entry>> reused) {
    return new ArmRunner(
        new ArmRunner.Context(
            ArmSpec.parse(spec),
            dataset,
            AnswerExtractor.forDataset(dataset),
            PromptBuilder.bundled(),
            settings,
            false,
            loglik,
            false,
            null,
            pilot,
            pool,
            members,
            reused));
  }

  @Test
  void memberArmRecordsTextTokensAnswersAndConfidence() {
    Item item =
        runner("member:B", DatasetKind.GSM8K, GREEDY, false, false, members(), Map.of()).run(ITEM);
    assertThat(item.outputs()).hasSize(1);
    ReportModel.Output output = item.outputs().getFirst();
    assertThat(output.tokenIds()).hasSize(output.tokens());
    assertThat(output.text()).isEqualTo(ToyMember.ToyTokenizer.INSTANCE.decode(output.tokenIds()));
    assertThat(item.prediction()).isEqualTo(output.statedAnswer());
    if (output.statedAnswer() != null) {
      assertThat(output.confidence()).isNotNull().isLessThanOrEqualTo(0.0);
    }
    assertThat(item.renderedPromptSha256()).hasSize(64);
    assertThat(item.aggregationCorrect()).containsKey("oracle");
  }

  @Test
  void tunedArmsRefuseToRunWithoutAFrozenRecordUnlessPiloting() {
    assertThatThrownBy(
            () ->
                runner(
                    "fuse:A+B+C:poe:tuned",
                    DatasetKind.GSM8K,
                    GREEDY,
                    false,
                    false,
                    members(),
                    Map.of()))
        .hasMessageContaining("--frozen");
    ArmRunner pilot =
        runner("fuse:A+B+C:poe:tuned", DatasetKind.GSM8K, GREEDY, false, true, members(), Map.of());
    assertThat(pilot.resolution().pilotSubstitution()).contains("tuned weights -> uniform");
    Item item = pilot.run(ITEM);
    assertThat(item.outputs().getFirst().agreement()).isNotNull();
    assertThat(item.outputs().getFirst().tokenLog()).isNotEmpty();
    ArmRunner conf =
        runner("vote-conf:A+B+C:best", DatasetKind.GSM8K, GREEDY, false, true, members(), Map.of());
    assertThat(conf.resolution().tieBreak()).isEqualTo("B");
    assertThat(conf.resolution().pilotSubstitution())
        .contains("tie-break best -> B")
        .contains("temperature 1.0");
  }

  @Test
  void votingReusesMemberOutputsWithoutLoadingModels() {
    Map<String, Map<String, ReusedOutputs.Entry>> reused = new LinkedHashMap<>();
    for (String name : List.of("A", "B", "C")) {
      Item memberItem =
          runner("member:" + name, DatasetKind.GSM8K, GREEDY, false, false, members(), Map.of())
              .run(ITEM);
      reused.put(
          name,
          Map.of(
              ITEM.id(),
              new ReusedOutputs.Entry(
                  markReused(memberItem.outputs().getFirst()),
                  1.5,
                  memberItem.renderedPromptSha256())));
    }
    Item vote =
        runner("vote:A+B+C:B", DatasetKind.GSM8K, GREEDY, false, false, Map.of(), reused).run(ITEM);
    assertThat(vote.outputs()).hasSize(3).allMatch(ReportModel.Output::reused);
    assertThat(vote.reusedCoreSeconds()).isEqualTo(4.5);
    assertThat(vote.generatedTokens()).isZero();
    assertThat(vote.aggregations()).containsKeys("majority", "consist", "conf");
    List<Vote> votes =
        vote.outputs().stream()
            .map(o -> new Vote(o.producer(), o.statedAnswer(), o.consistent(), Double.NaN))
            .toList();
    assertThat(vote.prediction())
        .isEqualTo(
            Voting.majority(
                votes, AnswerExtractor.forDataset(DatasetKind.GSM8K), List.of("B", "A", "C")));
  }

  private static ReportModel.Output markReused(ReportModel.Output o) {
    return new ReportModel.Output(
        o.producer(),
        o.seed(),
        o.text(),
        o.tokenIds(),
        o.tokenLogProbabilities(),
        o.reasoningAnswer(),
        o.statedAnswer(),
        o.confidence(),
        o.consistent(),
        o.thinkPresent(),
        o.thinkTruncated(),
        o.truncated(),
        o.stoppedOnEndOfGeneration(),
        o.tokens(),
        o.wallMillis(),
        o.prefillMillis(),
        o.memberForwardMillis(),
        null,
        List.of(),
        true);
  }

  @Test
  void selfConsistencyRecordsEverySampleWithItsSeed() {
    DecodeSettings sampled = new DecodeSettings(0.7f, 0, 1f, 40L, 16, 0);
    Item item =
        runner("sc:C:3", DatasetKind.GSM8K, sampled, false, false, members(), Map.of()).run(ITEM);
    assertThat(item.outputs()).extracting(ReportModel.Output::seed).containsExactly(40L, 41L, 42L);
  }

  @Test
  void rerankScoresEveryDistinctCandidateWithEveryMember() {
    Item item =
        runner("rerank:A+B+C:uniform", DatasetKind.GSM8K, GREEDY, false, false, members(), Map.of())
            .run(ITEM);
    assertThat(item.candidates())
        .allMatch(c -> c.memberScores().keySet().containsAll(List.of("A", "B", "C")));
    if (!item.candidates().isEmpty()) {
      assertThat(item.candidates())
          .extracting(ReportModel.CandidateScore::answer)
          .contains(item.prediction());
    }
  }

  @Test
  void arcLogLikelihoodScoresEveryOptionUnderTheFusionRule() {
    DatasetItem arc =
        new DatasetItem(
            "arc-toy",
            "pick",
            "2",
            List.of(
                new DatasetItem.Choice("1", "x"),
                new DatasetItem.Choice("2", "y"),
                new DatasetItem.Choice("3", "z")));
    Item item =
        runner(
                "fuse:A+B:mixture:0.5,0.5",
                DatasetKind.ARC,
                GREEDY,
                true,
                false,
                members(),
                Map.of())
            .run(arc);
    assertThat(item.candidates())
        .extracting(ReportModel.CandidateScore::answer)
        .containsExactly("1", "2", "3");
    double best =
        item.candidates().stream()
            .mapToDouble(ReportModel.CandidateScore::fusedScore)
            .max()
            .orElseThrow();
    assertThat(
            item.candidates().stream()
                .filter(c -> c.fusedScore() == best)
                .findFirst()
                .orElseThrow()
                .answer())
        .isEqualTo(item.prediction());
    assertThat(item.outputs()).isEmpty();
  }

  @Test
  void aCompleteArmReportSatisfiesTheSchemaAndAMissingFieldIsCaught() {
    ObjectMapper mapper = new ObjectMapper();
    ArmRunner runner =
        runner(
            "fuse:A+B+C:poe:uniform", DatasetKind.GSM8K, GREEDY, false, false, members(), Map.of());
    List<Item> items = runner.runAll(List.of(ITEM), line -> {});
    ReportModel.Report report = ReportFixtures.armReport(mapper, items);
    ObjectNode tree = mapper.valueToTree(report);
    assertThat(ReportSchema.violations(tree)).isEmpty();

    ObjectNode missingCap = tree.deepCopy();
    ((ObjectNode) missingCap.path("config").path("decoding")).remove("maxTokens");
    assertThat(ReportSchema.violations(missingCap))
        .contains("config/decoding/maxTokens is missing");

    ObjectNode nullJava = tree.deepCopy();
    ((ObjectNode) nullJava.path("environment")).putNull("javaVersionOutput");
    assertThat(ReportSchema.violations(nullJava))
        .contains("environment/javaVersionOutput must not be null");

    ObjectNode noTokens = tree.deepCopy();
    ((ObjectNode) noTokens.path("items").get(0).path("outputs").get(0)).remove("tokenIds");
    assertThat(ReportSchema.violations(noTokens))
        .contains("/items/0/outputs/0/tokenIds is missing");

    ObjectNode noMemberHash = tree.deepCopy();
    ((ObjectNode) noMemberHash.path("config").path("members").get(0)).remove("tokenizerSha256");
    assertThat(ReportSchema.violations(noMemberHash))
        .contains("/config/members/0/tokenizerSha256 is missing");
  }
}
