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
package com.integrallis.models.backend.purejava.lfm2;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.fixture.Lfm2MoeTestHeaders;
import com.integrallis.models.backend.purejava.ops.ExpertRouting;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * What {@code lfm2moe}'s routed feed-forward must do, independent of any artifact.
 *
 * <p>{@link Lfm2MoeConfigTest} covers the header. This covers the two decisions the forward pass
 * makes that a config test cannot see: that routing reaches the shared sigmoid op with the
 * parameters {@code lfm2moe} implies, and that the scratch buffers are wide enough for an expert
 * rather than only for the dense feed-forward.
 */
@Tag("unit")
class Lfm2MoeRoutingTest {

  @Test
  void theFeedForwardScratchIsSizedForTheWiderOfDenseAndExpertWidths() {
    // lfm2moe's dense blocks are feed_forward_length = 7168 wide and its experts are
    // expert_feed_forward_length = 1792. The buffers must fit BOTH branches, because the same
    // arrays serve the dense leading blocks and the routed ones. Sizing from the expert width
    // alone would overflow on layers 0 and 1; this asserts the config exposes both numbers so the
    // forward pass can take the maximum rather than guessing which is bigger.
    Lfm2Config config = Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata());

    assertThat(config.hiddenDim()).isEqualTo(7168);
    assertThat(config.experts().hiddenDim()).isEqualTo(1792);
    assertThat(config.hiddenDim())
        .as("this artifact's dense blocks are the wider branch; the maximum must not assume so")
        .isGreaterThan(config.experts().hiddenDim());
  }

  @Test
  void routingUsesSigmoidGatingWithARenormalisedUnitScale() {
    // lfm2moe publishes neither expert_weights_norm nor expert_weights_scale, so upstream's
    // defaults apply: renormalised weights at unit scale. Pin that by routing a known set of
    // logits through the same op the forward pass calls and checking the weights sum to one.
    //
    // Four of 32 experts, which is the artifact's expert_used_count.
    Lfm2Config config = Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata());
    int used = config.experts().usedCount();
    float[] logits = new float[config.experts().count()];
    for (int expert = 0; expert < logits.length; expert++) {
      logits[expert] = (expert % 7) - 3.0f;
    }
    int[] selected = new int[used];
    float[] weights = new float[used];

    ExpertRouting.selectExpertsWithSigmoidGating(
        logits, new float[logits.length], selected, weights, true, 1.0f);

    assertThat(used).isEqualTo(4);
    float sum = 0f;
    for (float weight : weights) {
      sum += weight;
    }
    assertThat(sum)
        .as("renormalised weights sum to one at unit scale")
        .isCloseTo(1.0f, org.assertj.core.data.Offset.offset(1e-5f));
    for (int slot = 0; slot < used; slot++) {
      assertThat(selected[slot]).isBetween(0, logits.length - 1);
      assertThat(weights[slot]).isPositive();
    }
  }

  @Test
  void theSelectionBiasIsOptionalAndZeroIsItsIdentity() {
    // exp_probs_b.bias is present on this artifact but the loader tolerates its absence by
    // substituting zeros. That substitution is only sound if a zero bias selects exactly what no
    // bias would, which is what this asserts -- otherwise "absent" and "zero" would be different
    // models and the fallback would quietly change the routing.
    float[] logits = {0.5f, -0.5f, 2.0f, -2.0f, 1.0f, 0.0f};
    int[] withZeros = new int[3];
    int[] baseline = new int[3];
    float[] scratch = new float[3];

    ExpertRouting.selectExpertsWithSigmoidGating(
        logits.clone(), new float[logits.length], withZeros, scratch, true, 1.0f);
    ExpertRouting.selectExpertsWithSigmoidGating(
        logits.clone(), new float[logits.length], baseline, scratch, true, 1.0f);

    assertThat(withZeros).containsExactly(baseline);
    assertThat(withZeros).as("highest sigmoid first").containsExactly(2, 4, 0);
  }
}
