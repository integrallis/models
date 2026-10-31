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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.fixture.Lfm2MoeTestHeaders;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code lfm2moe}'s config, against the real header.
 *
 * <p>The fixture is {@link Lfm2MoeTestHeaders}, read off {@code LFM2.5-8B-A1B-Q4_0.gguf} with the
 * surfaced key count reconciled against {@code kv_count}. Two axes decide this port and neither is
 * visible in a scalar key: which layers attend, and which layers route.
 */
@Tag("unit")
class Lfm2MoeConfigTest {

  private static final int LAYERS = Lfm2MoeTestHeaders.LAYERS;

  @Test
  void readsTheExpertBlockFromTheArtifactsOwnKeys() {
    Lfm2Config config = Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata());

    assertThat(config.isMixtureOfExperts()).isTrue();
    assertThat(config.experts().count()).isEqualTo(32);
    assertThat(config.experts().usedCount()).isEqualTo(4);
    // expert_feed_forward_length, NOT feed_forward_length (7168). Taking the dense width here
    // would allocate the expert matrices four times too wide and load nothing that fits.
    assertThat(config.experts().hiddenDim()).isEqualTo(1792);
    assertThat(config.experts().leadingDenseBlocks()).isEqualTo(2);
  }

  @Test
  void theDenseAndRoutedBlocksSplitAtTheDeclaredLeadingCount() {
    Lfm2Config config = Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata());

    for (int layer = 0; layer < LAYERS; layer++) {
      assertThat(config.usesExperts(layer))
          .describedAs("layer %d", layer)
          .isEqualTo(Lfm2MoeTestHeaders.routed(layer));
    }
    assertThat(config.usesExperts(0)).isFalse();
    assertThat(config.usesExperts(1)).isFalse();
    assertThat(config.usesExperts(2)).isTrue();
    assertThat(config.usesExperts(LAYERS - 1)).isTrue();
  }

  @Test
  void theAttentionSplitIsADifferentAxisFromTheRoutedSplit() {
    // The two per-layer axes are independent and are easy to conflate: 6 of 24 layers attend
    // (head_count_kv [0,0,8,0] x 6) and 22 of 24 route (leading_dense_block_count 2). Layer 2 is
    // both; layer 0 is neither; layer 3 routes and does not attend. A port that derived one from
    // the other would be right on some layers and wrong on others, which is the worst case.
    Lfm2Config config = Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata());

    assertThat(config.attentionLayers()).isEqualTo(6);
    assertThat(Lfm2MoeTestHeaders.attentionLayers()).isEqualTo(6);
    for (int layer = 0; layer < LAYERS; layer++) {
      assertThat(config.usesAttention(layer))
          .describedAs("layer %d attends", layer)
          .isEqualTo(Lfm2MoeTestHeaders.attends(layer));
    }

    assertThat(config.usesAttention(2) && config.usesExperts(2)).as("layer 2 is both").isTrue();
    assertThat(!config.usesAttention(0) && !config.usesExperts(0))
        .as("layer 0 is neither")
        .isTrue();
    assertThat(!config.usesAttention(3) && config.usesExperts(3))
        .as("layer 3 routes but convolves")
        .isTrue();
  }

  @Test
  void refusesAGatingFunctionThisBackendDoesNotImplement() {
    // #283: "do not work around it with a softmax approximation -- a gating function that is close
    // but not the declared one is the kind of silent divergence this queue exists to avoid." So a
    // file declaring anything but sigmoid is refused at load rather than served with the wrong
    // gate.
    assertThatThrownBy(
            () -> Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata("lfm2moe", true, 1)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sigmoid")
        .hasMessageContaining("declares 1");

    // And every other value llama.cpp numbers, so the refusal is not accidentally specific to 1.
    for (int gating : new int[] {0, 3, 4}) {
      assertThatThrownBy(
              () -> Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata("lfm2moe", true, gating)))
          .describedAs("expert_gating_func=%d", gating)
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void aMixtureThatDeclaresNoExpertKeysIsRefused() {
    // The architecture string claims a mixture and the file describes none. Defaulting to dense
    // would load it, run the dense branch on every layer, and silently ignore every expert tensor.
    assertThatThrownBy(() -> Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata("lfm2moe", false)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void theDenseSiblingStillLoadsAndDeclaresNoExperts() {
    // The negative control for the whole change: lfm2 must be unaffected. If accepting a second
    // architecture string had made the expert block mandatory, or had changed the prefix the dense
    // keys are read under, this fails.
    Lfm2Config config = Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata("lfm2", false));

    assertThat(config.isMixtureOfExperts()).isFalse();
    assertThat(config.experts()).isNull();
    for (int layer = 0; layer < LAYERS; layer++) {
      assertThat(config.usesExperts(layer)).describedAs("layer %d", layer).isFalse();
    }
    assertThat(config.attentionLayers()).isEqualTo(6);
    assertThat(config.shortConvCache()).isEqualTo(3);
  }

  @Test
  void anUnknownArchitectureIsStillRefused() {
    assertThatThrownBy(() -> Lfm2Config.fromMetadata(Lfm2MoeTestHeaders.metadata("lfm3", false)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("lfm2 or lfm2moe");
  }
}
