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
package com.integrallis.models.backend.purejava.deepseek2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The DeepSeek-V2 contract, keyed exactly as the published {@code DeepSeek-Coder-V2-Lite-Instruct}
 * header is -- every value below was read out of that file on 2026-09-29.
 *
 * <p>The two derived scalars are what this mostly exists for. YaRN makes the attention softmax
 * scale 1.59 times the plain {@code 1/sqrt(keyLength)} and the rotary magnitude 0.73 of unity, and
 * both are arrived at by an indirect route in the reference. Getting either wrong produces a model
 * that runs.
 */
@Tag("unit")
class Deepseek2ConfigTest {

  @Test
  void parsesThePublishedLiteHeader() {
    Deepseek2Config config = Deepseek2Config.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.embeddingDim()).isEqualTo(2_048);
    assertThat(config.numLayers()).isEqualTo(27);
    assertThat(config.numHeads()).isEqualTo(16);
    assertThat(config.keyLength()).isEqualTo(192);
    assertThat(config.valueLength()).isEqualTo(128);
    assertThat(config.kvLoraRank()).isEqualTo(512);
    assertThat(config.ropeDimension()).isEqualTo(64);
    assertThat(config.vocabSize()).isEqualTo(102_400);
    assertThat(config.hiddenDim()).isEqualTo(10_944);
    assertThat(config.expertHiddenDim()).isEqualTo(1_408);
    assertThat(config.numExperts()).isEqualTo(64);
    assertThat(config.numExpertsUsed()).isEqualTo(6);
    assertThat(config.numSharedExperts()).isEqualTo(2);
    assertThat(config.leadingDenseLayers()).isEqualTo(1);
  }

  /** The derived widths, each of which a tensor shape in the published file has to match. */
  @Test
  void theDerivedWidthsMatchThePublishedTensorShapes() {
    Deepseek2Config config = Deepseek2Config.fromMetadata(new GgufMetadata(entries()));

    // attn_q.weight is [2048, 3072]
    assertThat(config.queryDim()).isEqualTo(3_072);
    // attn_kv_a_mqa.weight is [2048, 576] = 512 latent + 64 rope
    assertThat(config.compressedKeyValueDim()).isEqualTo(576);
    // attn_kv_b.weight is [512, 4096] = 16 heads x (128 no-rope + 128 value)
    assertThat(config.noRopeDimension()).isEqualTo(128);
    assertThat(config.decompressedKeyValueDim()).isEqualTo(4_096);
    // ffn_*_shexp is 2816 wide: the two shared experts fused, 2 x 1408
    assertThat(config.sharedExpertHiddenDim()).isEqualTo(2_816);
    assertThat(config.cachedKeyDim()).isEqualTo(3_072);
    assertThat(config.cachedValueDim()).isEqualTo(2_048);
  }

  /**
   * YaRN's attention scale, computed rather than estimated.
   *
   * <p>{@code mscale = 1 + 0.0707 * ln(40) = 1.260804}, so the scale is {@code 1.260804^2 /
   * sqrt(192) = 0.114721} where the plain scale is {@code 0.072169}. The assertion names both,
   * because the failure mode is silently using the plain one.
   */
  @Test
  void theAttentionScaleCarriesTheYarnMagnitude() {
    Deepseek2Config config = Deepseek2Config.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.attentionScale()).isEqualTo(0.114721f, within(1.0e-6f));
    assertThat(config.attentionScale())
        .describedAs("the plain 1/sqrt(keyLength) would be 1.59x too small")
        .isNotEqualTo(0.072169f);
  }

  /**
   * And the rotary magnitude, which is below one because the reference moves the rest into the
   * scale.
   */
  @Test
  void theRotaryMagnitudeIsTheReciprocalOfTheGenericYarnTerm() {
    Deepseek2Config config = Deepseek2Config.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.ropeAttentionFactor()).isEqualTo(0.730520f, within(1.0e-6f));
    assertThat(config.ropeAttentionFactor())
        .describedAs("the generic YaRN magnitude 1 + 0.1*ln(factor) is its reciprocal")
        .isLessThan(1.0f);
  }

  /**
   * Without YaRN both reduce to the ordinary values, so a model that omits the keys still works.
   */
  @Test
  void aModelWithoutYarnGetsThePlainScales() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.remove("deepseek2.rope.scaling.factor");
    entries.remove("deepseek2.rope.scaling.yarn_log_multiplier");
    Deepseek2Config config = Deepseek2Config.fromMetadata(new GgufMetadata(entries));

    assertThat(config.ropeAttentionFactor()).isEqualTo(1.0f);
    assertThat(config.attentionScale()).isEqualTo((float) (1.0 / Math.sqrt(192)), within(1.0e-7f));
  }

  /** An expert-weight scale of 1.0 means no scaling, the same as the reference's 0.0 default. */
  @Test
  void anExpertWeightScaleOfOneMeansNoScaling() {
    Deepseek2Config config = Deepseek2Config.fromMetadata(new GgufMetadata(entries()));
    assertThat(config.expertWeightsScale()).isEqualTo(1.0f);
    assertThat(config.scalesExpertWeights()).isFalse();

    Map<String, GgufMetadataValue> zeroed = entries();
    zeroed.put("deepseek2.expert_weights_scale", new GgufMetadataValue.Float32Value(0.0f));
    assertThat(Deepseek2Config.fromMetadata(new GgufMetadata(zeroed)).scalesExpertWeights())
        .describedAs("0 and 1 both mean no scaling in the reference")
        .isFalse();

    Map<String, GgufMetadataValue> scaled = entries();
    scaled.put("deepseek2.expert_weights_scale", new GgufMetadataValue.Float32Value(2.5f));
    assertThat(Deepseek2Config.fromMetadata(new GgufMetadata(scaled)).scalesExpertWeights())
        .isTrue();
  }

  /** Layer 0 is dense and the rest are routed, which is what leading_dense_block_count says. */
  @Test
  void theLeadingLayersAreDense() {
    Deepseek2Config config = Deepseek2Config.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.usesMixtureOfExperts(0)).isFalse();
    for (int layer = 1; layer < config.numLayers(); layer++) {
      assertThat(config.usesMixtureOfExperts(layer)).isTrue();
    }
  }

  @Test
  void aRopeDimensionThatLeavesNoNoRopePartIsRefused() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("deepseek2.rope.dimension_count", new GgufMetadataValue.Uint32Value(192));

    assertThatThrownBy(() -> Deepseek2Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("no-rope");
  }

  @Test
  void anotherArchitectureIsRefused() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("llama"));

    assertThatThrownBy(() -> Deepseek2Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("deepseek2");
  }

  private static Map<String, GgufMetadataValue> entries() {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("deepseek2"));
    entries.put("deepseek2.embedding_length", new GgufMetadataValue.Uint32Value(2_048));
    entries.put("deepseek2.block_count", new GgufMetadataValue.Uint32Value(27));
    entries.put("deepseek2.attention.head_count", new GgufMetadataValue.Uint32Value(16));
    entries.put("deepseek2.attention.head_count_kv", new GgufMetadataValue.Uint32Value(16));
    entries.put("deepseek2.attention.key_length", new GgufMetadataValue.Uint32Value(192));
    entries.put("deepseek2.attention.value_length", new GgufMetadataValue.Uint32Value(128));
    entries.put("deepseek2.attention.kv_lora_rank", new GgufMetadataValue.Uint32Value(512));
    entries.put("deepseek2.rope.dimension_count", new GgufMetadataValue.Uint32Value(64));
    entries.put("deepseek2.vocab_size", new GgufMetadataValue.Uint32Value(102_400));
    entries.put("deepseek2.context_length", new GgufMetadataValue.Uint32Value(163_840));
    entries.put("deepseek2.feed_forward_length", new GgufMetadataValue.Uint32Value(10_944));
    entries.put("deepseek2.expert_feed_forward_length", new GgufMetadataValue.Uint32Value(1_408));
    entries.put("deepseek2.expert_count", new GgufMetadataValue.Uint32Value(64));
    entries.put("deepseek2.expert_used_count", new GgufMetadataValue.Uint32Value(6));
    entries.put("deepseek2.expert_shared_count", new GgufMetadataValue.Uint32Value(2));
    entries.put("deepseek2.expert_weights_scale", new GgufMetadataValue.Float32Value(1.0f));
    entries.put("deepseek2.leading_dense_block_count", new GgufMetadataValue.Uint32Value(1));
    entries.put("deepseek2.rope.freq_base", new GgufMetadataValue.Float32Value(10_000.0f));
    entries.put("deepseek2.rope.scaling.factor", new GgufMetadataValue.Float32Value(40.0f));
    entries.put(
        "deepseek2.rope.scaling.original_context_length", new GgufMetadataValue.Uint32Value(4_096));
    entries.put(
        "deepseek2.rope.scaling.yarn_log_multiplier",
        new GgufMetadataValue.Float32Value(0.07069999724626541f));
    entries.put(
        "deepseek2.attention.layer_norm_rms_epsilon",
        new GgufMetadataValue.Float32Value(9.999999974752427e-07f));
    return entries;
  }
}
