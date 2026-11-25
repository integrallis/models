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
package com.integrallis.models.runtime.fusion;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.runtime.InferencePipeline;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FusedInferenceBackendTest {

  private static final int[] PROMPT = {3, 1, 4, 1, 5};
  private static final int STEPS = 40;

  private static ToyBackend lead() {
    return new ToyBackend("lead", 1.3, 0.0);
  }

  private static ToyBackend math() {
    return new ToyBackend("math", 2.1, 0.4);
  }

  private static ToyBackend general() {
    return new ToyBackend("general", 0.7, -0.2);
  }

  /** Greedy decode: prefill the prompt, then argmax until EOS or {@code STEPS}. */
  private static List<Integer> greedy(InferenceBackend backend) {
    backend.reset();
    float[] logits = backend.prefill(PROMPT, 0);
    int position = PROMPT.length;
    List<Integer> out = new ArrayList<>();
    for (int i = 0; i < STEPS; i++) {
      int token = FusionMath.argmax(logits);
      out.add(token);
      if (token == ToyBackend.EOS) {
        break;
      }
      logits = backend.forward(token, position++);
    }
    return out;
  }

  @Test
  void gateThatNeverFiresReproducesTheLeadTokenForTokenAndLeavesAuxiliariesIdle() {
    List<Integer> alone = greedy(lead());
    ToyBackend math = math();
    ToyBackend general = general();
    FusedInferenceBackend fused =
        new FusedInferenceBackend(
            List.of(lead(), math, general),
            new FusionRecipe(
                FusionRule.POE, new double[] {0.55, 0.25, 0.20}, EnsembleGate.never()));
    assertThat(greedy(fused)).isEqualTo(alone);
    assertThat(math.prefillDeltas).isEmpty();
    assertThat(math.forwardCalls).isZero();
    assertThat(general.prefillDeltas).isEmpty();
    assertThat(fused.metrics().consulted()).isZero();
    assertThat(fused.metrics().steps()).isEqualTo(alone.size());
  }

  @Test
  void entropyGateAtInfinityIsTheNeverGateAndAtZeroTheAlwaysGate() {
    assertThat(new EntropyGate(Double.POSITIVE_INFINITY).consult(50.0, 0)).isFalse();
    assertThat(new EntropyGate(0.0).consult(0.0, 0)).isTrue();
    assertThat(new EntropyGate(0.65).consult(0.6499, 3)).isFalse();
    assertThat(new EntropyGate(0.65).consult(0.65, 3)).isTrue();
    assertThatThrownBy(() -> new EntropyGate(-0.1)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void gateThatAlwaysFiresEqualsProductOfExpertsOverEveryMemberAtEveryStep() {
    double[] weights = {0.5, 0.3, 0.2};
    // Reference: every member steps every token; the next token is the argmax of the weighted sum
    // of log-softmax vectors.
    List<ToyBackend> reference = List.of(lead(), math(), general());
    List<Integer> expected = new ArrayList<>();
    float[][] logits = new float[3][];
    for (int i = 0; i < 3; i++) {
      reference.get(i).reset();
      logits[i] = reference.get(i).prefill(PROMPT, 0);
    }
    int position = PROMPT.length;
    for (int step = 0; step < STEPS; step++) {
      double[][] logProbabilities = new double[3][];
      for (int i = 0; i < 3; i++) {
        logProbabilities[i] = FusionMath.logSoftmax(logits[i]);
      }
      int token =
          FusionMath.argmax(FusionMath.combine(FusionRule.POE, logits, logProbabilities, weights));
      expected.add(token);
      if (token == ToyBackend.EOS) {
        break;
      }
      for (int i = 0; i < 3; i++) {
        logits[i] = reference.get(i).forward(token, position);
      }
      position++;
    }

    ToyBackend math = math();
    FusedInferenceBackend fused =
        new FusedInferenceBackend(
            List.of(lead(), math, general()),
            new FusionRecipe(FusionRule.POE, weights, EnsembleGate.always()));
    List<Integer> actual = greedy(fused);
    assertThat(actual).isEqualTo(expected);
    assertThat(fused.metrics().consulted()).isEqualTo(actual.size());
    assertThat(fused.metrics().consultRate()).isEqualTo(1.0);
    // Consulted at every step, each auxiliary catches up exactly one token per step after the
    // prompt, through prefill, never through forward.
    assertThat(math.forwardCalls).isZero();
    assertThat(math.prefillDeltas.getFirst()).isEqualTo(PROMPT);
    assertThat(math.prefillDeltas.subList(1, math.prefillDeltas.size()))
        .allSatisfy(delta -> assertThat(delta).hasSize(1));
  }

  @Test
  void auxiliariesCatchUpOverExactlyTheUnseenDeltaAndSeeTheSameHistoryAsStepping() {
    // Fire at positions 7 and 11 only (the prompt ends at position 4).
    EnsembleGate at = (entropy, position) -> position == 7 || position == 11;
    ToyBackend math = math();
    List<FusionStep> steps = new ArrayList<>();
    FusedInferenceBackend fused =
        new FusedInferenceBackend(
            List.of(lead(), math, general()), FusionRecipe.uniform(3, at), steps::add);
    List<Integer> out = greedy(fused);
    assertThat(out.size()).isGreaterThan(8);

    // First catch-up: the prompt plus the generated tokens up to and including position 7.
    assertThat(math.prefillStarts).startsWith(0);
    assertThat(math.prefillDeltas.getFirst()).hasSize(8);
    // Second catch-up: positions 8..11, four tokens, starting where the first stopped.
    assertThat(math.prefillStarts.get(1)).isEqualTo(8);
    assertThat(math.prefillDeltas.get(1)).hasSize(4);
    assertThat(math.forwardCalls).isZero();

    // What the auxiliary saw equals the lead's history at that point: the prompt plus the emitted
    // tokens, token for token.
    List<Integer> fed = new ArrayList<>();
    for (int token : PROMPT) {
      fed.add(token);
    }
    fed.addAll(out.subList(0, 3));
    assertThat(toList(math.prefillDeltas.getFirst())).isEqualTo(fed);
    // And its logits after the catch-up are those of a fresh member having stepped that history.
    float[] viaCatchUp = ToyBackend.logitsAfter(2.1, 0.4, fed);
    assertThat(viaCatchUp).containsExactly(ToyBackend.logitsAfter(2.1, 0.4, fed), within(0f));

    FusionStep first = steps.stream().filter(FusionStep::consulted).findFirst().orElseThrow();
    assertThat(first.position()).isEqualTo(7);
    assertThat(first.catchUpTokens()).containsExactly(8, 8);
    assertThat(fused.metrics().consulted()).isEqualTo(2);
    assertThat(fused.metrics().catchUpTokens()).isEqualTo(2L * (8 + 4));
    assertThat(fused.metrics().steps()).isEqualTo(out.size());
  }

  @Test
  void rewindRestoresEveryMemberAndContinuationIsIdentical() {
    EnsembleGate at = (entropy, position) -> position % 3 == 0;
    FusedInferenceBackend fused =
        new FusedInferenceBackend(List.of(lead(), math(), general()), FusionRecipe.uniform(3, at));
    fused.reset();
    float[] logits = fused.prefill(PROMPT, 0);
    int position = PROMPT.length;
    List<Integer> firstRun = new ArrayList<>();
    int checkpoint = -1;
    float[] logitsAtCheckpoint = null;
    for (int i = 0; i < 12; i++) {
      int token = FusionMath.argmax(logits);
      firstRun.add(token);
      logits = fused.forward(token, position++);
      if (i == 5) {
        checkpoint = fused.checkpoint();
        logitsAtCheckpoint = logits.clone();
      }
    }
    assertThat(checkpoint).isEqualTo(PROMPT.length + 6);
    fused.rewind(checkpoint);
    assertThat(fused.checkpoint()).isEqualTo(checkpoint);
    // Replay from the checkpoint: the same tokens come out.
    logits = logitsAtCheckpoint;
    position = checkpoint;
    List<Integer> replay = new ArrayList<>(firstRun.subList(0, 6));
    for (int i = 6; i < 12; i++) {
      int token = FusionMath.argmax(logits);
      replay.add(token);
      logits = fused.forward(token, position++);
    }
    assertThat(replay).isEqualTo(firstRun);
    assertThatThrownBy(() -> fused.rewind(999)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void resetClearsHistoryAndMetricsOnEveryMember() {
    ToyBackend math = math();
    FusedInferenceBackend fused =
        new FusedInferenceBackend(
            List.of(lead(), math, general()), FusionRecipe.uniform(3, EnsembleGate.always()));
    greedy(fused);
    assertThat(fused.metrics().steps()).isPositive();
    fused.reset();
    assertThat(fused.metrics()).isEqualTo(FusionMetrics.empty());
    assertThat(fused.checkpoint()).isZero();
    assertThat(math.resets).isGreaterThanOrEqualTo(2);
    // Positions restart at zero: a forward at the old position is refused.
    assertThatThrownBy(() -> fused.forward(1, 5)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void rejectsMismatchedVocabulariesWeightsAndNonRewindableMembers() {
    ToyBackend other =
        new ToyBackend("other", 1.0, 0.0) {
          @Override
          public com.integrallis.models.api.ModelMetadata metadata() {
            return new com.integrallis.models.api.ModelMetadata(
                "toy", "other", 256, 13, 8, 1, 1, 1);
          }
        };
    assertThatThrownBy(
            () ->
                new FusedInferenceBackend(
                    List.of(lead(), other), FusionRecipe.uniform(2, EnsembleGate.always())))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("vocabulary");
    assertThatThrownBy(
            () ->
                new FusedInferenceBackend(
                    List.of(lead(), math(), general()),
                    FusionRecipe.uniform(2, EnsembleGate.always())))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("weights");
    assertThatThrownBy(
            () -> new FusionRecipe(FusionRule.POE, new double[] {0.7, 0.7}, EnsembleGate.always()))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> FusionRecipe.uniform(1, EnsembleGate.always()))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void worksBehindTheInferencePipelineAndReportsItsMembersInItsName() {
    FusedInferenceBackend fused =
        new FusedInferenceBackend(
            List.of(lead(), math(), general()),
            new FusionRecipe(
                FusionRule.POE,
                new double[] {0.55, 0.25, 0.20},
                new EntropyGate(EntropyGate.PUBLISHED_THRESHOLD_NATS)));
    assertThat(fused.name()).isEqualTo("fused(lead+math+general)");
    try (InferencePipeline pipeline = new InferencePipeline(fused)) {
      assertThat(pipeline.rewindable()).isTrue();
      float[] logits = pipeline.prefill(PROMPT, 0);
      assertThat(logits).hasSize(ToyBackend.VOCAB);
      pipeline.forward(FusionMath.argmax(logits), PROMPT.length);
      assertThat(fused.metrics().steps()).isEqualTo(2);
    }
    assertThat(fused.members()).allSatisfy(m -> assertThat(((ToyBackend) m).closed).isTrue());
  }

  @Test
  void singlePassEntropyMatchesTheLogProbabilityDefinition() {
    float[] logits = {1.5f, -0.3f, 2.2f, 0.0f, -4.0f, 2.2f};
    double viaLogProbabilities = FusionMath.entropy(FusionMath.logSoftmax(logits));
    assertThat(FusionMath.entropyOfLogits(logits)).isCloseTo(viaLogProbabilities, within(1e-6));
    assertThat(FusionMath.entropyOfLogits(new float[] {0f, 0f, 0f, 0f}))
        .isCloseTo(Math.log(4), within(1e-9));
    assertThat(FusionMath.entropyOfLogits(new float[] {100f, 0f})).isCloseTo(0.0, within(1e-9));
  }

  private static List<Integer> toList(int[] values) {
    List<Integer> list = new ArrayList<>();
    for (int value : values) {
      list.add(value);
    }
    return list;
  }
}
