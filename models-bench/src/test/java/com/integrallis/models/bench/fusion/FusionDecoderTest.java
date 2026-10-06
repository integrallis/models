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

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class FusionDecoderTest {
  private final ExecutorService pool = Executors.newFixedThreadPool(3);
  private static final int[] PROMPT = {1, 4, 2, 10, 7};

  @AfterEach
  void shutdown() {
    pool.shutdownNow();
  }

  private static List<FusionMember> members() {
    return List.of(
        new ToyMember("A", 0.7, 0), new ToyMember("B", 1.3, 5), new ToyMember("C", 2.1, -2));
  }

  private static DecodeSettings greedy(int maxTokens) {
    return new DecodeSettings(0f, 0, 1f, 7L, maxTokens, 1);
  }

  @Test
  void unitWeightVectorReproducesEachMemberAloneTokenForTokenUnderEveryRule() {
    for (FusionRule rule : FusionRule.values()) {
      for (int i = 0; i < 3; i++) {
        List<FusionMember> fused = members();
        double[] weights = new double[3];
        weights[i] = 1;
        DecodeResult viaFusion = FusionDecoder.fuse(fused, PROMPT, weights, rule, greedy(40), pool);
        DecodeResult alone = FusionDecoder.single(members().get(i), PROMPT, greedy(40));
        assertThat(viaFusion.tokens()).as(rule + " e_" + i).containsExactly(alone.tokens());
        assertThat(viaFusion.truncated()).isEqualTo(alone.truncated());
      }
    }
  }

  @Test
  void uniformProductOfExpertsPicksArgmaxOfSummedLogProbabilities() {
    List<FusionMember> fused = members();
    DecodeResult result =
        FusionDecoder.fuse(
            fused,
            PROMPT,
            new double[] {1 / 3.0, 1 / 3.0, 1 / 3.0},
            FusionRule.POE,
            greedy(1),
            pool);
    double[] sum = new double[ToyMember.VOCAB];
    for (FusionMember member : members()) {
      double[] logProbs = FusionMath.logSoftmax(member.prefill(PROMPT, 0));
      for (int t = 0; t < sum.length; t++) {
        sum[t] += logProbs[t] / 3.0;
      }
    }
    assertThat(result.tokens()[0]).isEqualTo(FusionMath.argmax(sum));
  }

  @Test
  void stopsOnEndOfGenerationAndFlagsTruncationAtTheCap() {
    DecodeResult capped = FusionDecoder.single(members().get(0), PROMPT, greedy(2));
    assertThat(capped.tokens()).hasSize(2);
    assertThat(capped.truncated()).isTrue();
    DecodeResult free = FusionDecoder.single(members().get(0), PROMPT, greedy(400));
    assertThat(free.truncated()).isFalse();
    assertThat(free.stoppedOnEndOfGeneration()).isTrue();
    assertThat(free.tokens()).doesNotContain(ToyMember.EOS);
  }

  @Test
  void parallelAndSequentialMemberSteppingAreIdentical() {
    double[] weights = {0.2, 0.5, 0.3};
    DecodeResult parallel =
        FusionDecoder.fuse(members(), PROMPT, weights, FusionRule.MIXTURE, greedy(30), pool);
    ExecutorService single = Executors.newSingleThreadExecutor();
    try {
      DecodeResult sequential =
          FusionDecoder.fuse(members(), PROMPT, weights, FusionRule.MIXTURE, greedy(30), single);
      assertThat(parallel.tokens()).containsExactly(sequential.tokens());
    } finally {
      single.shutdownNow();
    }
  }

  @Test
  void seededSamplingIsReproducible() {
    DecodeSettings sampled = new DecodeSettings(0.7f, 0, 1f, 99L, 30, 1);
    DecodeResult first =
        FusionDecoder.fuse(
            members(), PROMPT, new double[] {0.4, 0.3, 0.3}, FusionRule.POE, sampled, pool);
    DecodeResult second =
        FusionDecoder.fuse(
            members(), PROMPT, new double[] {0.4, 0.3, 0.3}, FusionRule.POE, sampled, pool);
    assertThat(first.tokens()).containsExactly(second.tokens());
  }

  @Test
  void agreementStatisticsCountEveryStepAndTheLogHonoursSampling() {
    DecodeSettings everyThird = new DecodeSettings(0f, 0, 1f, 7L, 30, 3);
    DecodeResult result =
        FusionDecoder.fuse(
            members(),
            PROMPT,
            new double[] {1 / 3.0, 1 / 3.0, 1 / 3.0},
            FusionRule.POE,
            everyThird,
            pool);
    AgreementStats stats = result.agreement();
    assertThat(stats.steps())
        .isEqualTo(result.tokens().length + (result.stoppedOnEndOfGeneration() ? 1 : 0));
    assertThat(stats.fusedEqualsMember()).hasSize(3);
    assertThat(stats.allMembersAgree()).isBetween(0, stats.steps());
    assertThat(result.tokenLog()).allMatch(entry -> entry.step() % 3 == 0);
    assertThat(result.tokenLog().get(0).klFusedToMember()).hasSize(3);
    assertThat(result.tokenLog().get(0).fusedEntropy()).isGreaterThanOrEqualTo(0);
    int fusedMatchesSomeMember = 0;
    for (TokenLogEntry entry : result.tokenLog()) {
      if (!entry.matchesMembers().isEmpty()) {
        fusedMatchesSomeMember++;
      }
    }
    assertThat(fusedMatchesSomeMember).isLessThanOrEqualTo(result.tokenLog().size());
  }

  @Test
  void memberTimingsAndPerTokenLogProbabilitiesAreRecorded() {
    DecodeResult result =
        FusionDecoder.fuse(
            members(), PROMPT, new double[] {0.5, 0.25, 0.25}, FusionRule.POE, greedy(10), pool);
    assertThat(result.memberForwardNanos()).containsOnlyKeys("A", "B", "C");
    assertThat(result.tokenLogProbabilities()).hasSize(result.tokens().length);
    for (double logProb : result.tokenLogProbabilities()) {
      assertThat(logProb).isLessThanOrEqualTo(0);
    }
  }

  @Test
  void selfConsistencyReusesThePromptPrefixAcrossSamples() {
    ToyMember member = new ToyMember("C", 2.1, -2);
    DecodeSettings sampled = new DecodeSettings(0.7f, 0, 1f, 5L, 20, 0);
    List<DecodeResult> samples = FusionDecoder.samples(member, PROMPT, sampled, 4);
    assertThat(samples).hasSize(4);
    assertThat(member.prefillCalls).isEqualTo(1);
    DecodeResult reference =
        FusionDecoder.single(new ToyMember("C", 2.1, -2), PROMPT, sampled.withSeed(5L + 2));
    assertThat(samples.get(2).tokens()).containsExactly(reference.tokens());
  }

  @Test
  void continuationScoringMatchesStepwiseLogProbabilities() {
    ToyMember member = new ToyMember("A", 0.7, 0);
    int[] continuation = {3, 1, 4};
    float[] promptLogits = member.prefill(PROMPT, 0);
    int checkpoint = member.checkpoint();
    ContinuationScore viaVerify =
        FusionDecoder.scoreContinuation(member, promptLogits, checkpoint, continuation, true);
    ContinuationScore viaForward =
        FusionDecoder.scoreContinuation(member, promptLogits, checkpoint, continuation, false);
    assertThat(member.checkpoint()).isEqualTo(checkpoint);
    assertThat(viaVerify.tokenLogProbabilities())
        .containsExactly(viaForward.tokenLogProbabilities());
    ToyMember fresh = new ToyMember("A", 0.7, 0);
    double expected = FusionMath.logSoftmax(fresh.prefill(PROMPT, 0))[3];
    expected += FusionMath.logSoftmax(fresh.forward(3, PROMPT.length))[1];
    expected += FusionMath.logSoftmax(fresh.forward(1, PROMPT.length + 1))[4];
    assertThat(viaForward.sum()).isCloseTo(expected, org.assertj.core.api.Assertions.within(1e-12));
  }

  @Test
  void answerSpanMapsToTheTokensThatProducedIt() {
    int[] tokens = {1, 10, 4, 2, 10, 9};
    String text = ToyMember.ToyTokenizer.INSTANCE.decode(tokens);
    int start = text.indexOf("42");
    int[] span = AnswerTokens.span(ToyMember.ToyTokenizer.INSTANCE, tokens, start, start + 2);
    assertThat(span).containsExactly(2, 4);
  }
}
