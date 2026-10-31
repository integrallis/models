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
package com.integrallis.models.backend.purejava.qwen35;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Pins what EdgeRock-4B's real header says, and what this repository does about it today.
 *
 * <p>Every value here is read from {@code EdgeRock-4B-Q4_K_M.gguf} with {@code
 * experiments/edgerock-decoder/tools/gguf_describe.py}, not from the model card. The point of the
 * test is that the scope recorded in {@code experiments/edgerock-decoder/README.md} cannot drift
 * from the artifact without something failing.
 *
 * <p>It sits in the {@code qwen35} package because it asserts that package's layer schedule against
 * EdgeRock's published mask, and {@code fullAttentionLayers()} is package-private. Widening
 * production API for a test would be the wrong trade.
 */
@Tag("unit")
class EdgeRockScopeTest {

  /** 32 blocks, of which 8 are full attention: the published mask, transcribed. */
  private static final int[] RECURRENT_LAYERS = {
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0
  };

  @Test
  void theBackboneKeysEdgeRockPublishesAreTheOnesQwen35Requires() {
    // Same keys, same meanings, only the prefix differs -- which is why this parses at all. If
    // Qwen35Config ever starts requiring a key EdgeRock does not publish, this fails and the
    // README's "22 of 37 roles are shared" claim stops being true quietly.
    Qwen35Config config = Qwen35Config.fromMetadata(new GgufMetadata(edgeRockShapeUnderQwen35()));

    assertThat(config.numLayers()).isEqualTo(32);
    assertThat(config.embeddingDim()).isEqualTo(2_560);
    assertThat(config.hiddenDim()).isEqualTo(9_216);
    assertThat(config.numHeads()).isEqualTo(16);
    assertThat(config.numKvHeads()).isEqualTo(4);
    assertThat(config.attentionHeadDim()).isEqualTo(256);
    assertThat(config.vocabSize()).isEqualTo(248_320);
    assertThat(config.contextLength()).isEqualTo(262_144);
    assertThat(config.gdnConvKernel()).isEqualTo(4);
    assertThat(config.ropeDimension()).isEqualTo(64);
  }

  @Test
  void theIntervalDerivedScheduleMatchesThePublishedRecurrentMask() {
    // EdgeRock ships both an explicit mask and full_attention_interval=4. They must agree, or the
    // interval-derived schedule our decoder computes would put attention on the wrong layers --
    // which loads fine and produces wrong output, so it has to be checked rather than assumed.
    Qwen35Config config = Qwen35Config.fromMetadata(new GgufMetadata(edgeRockShapeUnderQwen35()));

    Integer[] attentionFromMask =
        java.util.stream.IntStream.range(0, RECURRENT_LAYERS.length)
            .filter(layer -> RECURRENT_LAYERS[layer] == 0)
            .boxed()
            .toArray(Integer[]::new);

    assertThat(config.fullAttentionLayers()).containsExactly(3, 7, 11, 15, 19, 23, 27, 31);
    assertThat(config.fullAttentionLayers()).containsExactly(attentionFromMask);
    // And the count agrees with the artifact: 8 attn_q tensors, 24 ssm_conv1d tensors.
    assertThat(attentionFromMask).hasSize(8);
    assertThat(RECURRENT_LAYERS.length - attentionFromMask.length).isEqualTo(24);
  }

  @Test
  void edgeRockDeclaringItselfStillNeedsItsOwnPrefixedKeys() {
    // This test used to assert that edgerock was refused outright. It is now served, and the
    // assertion was kept alive by its fixture rather than by the behaviour: with every key under
    // qwen35., renaming the architecture to edgerock made them all missing, so the throw came
    // from absence and not from any refusal. What is worth pinning is the prefix rule itself --
    // a file that declares edgerock is read under edgerock., and the real published file is
    // dual-keyed precisely so that both readings resolve.
    Map<String, GgufMetadataValue> entries = edgeRockShapeUnderQwen35();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("edgerock"));

    assertThatThrownBy(() -> Qwen35Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("edgerock.embedding_length");

    // Admission and the memory-anchor requirement are asserted against the transcribed header in
    // Qwen35ConfigEdgeRockTest, which carries the edgerock.* keys the real file publishes.
  }

  /**
   * EdgeRock-4B's measured shape, keyed under {@code qwen35.} so the shared backbone can be
   * asserted before {@code edgerock} is a served architecture.
   */
  private static Map<String, GgufMetadataValue> edgeRockShapeUnderQwen35() {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("qwen35"));
    entries.put("qwen35.block_count", uint(32));
    entries.put("qwen35.context_length", uint(262_144));
    entries.put("qwen35.embedding_length", uint(2_560));
    entries.put("qwen35.feed_forward_length", uint(9_216));
    entries.put("qwen35.attention.head_count", uint(16));
    entries.put("qwen35.attention.head_count_kv", uint(4));
    entries.put("qwen35.attention.key_length", uint(256));
    entries.put("qwen35.attention.value_length", uint(256));
    entries.put("qwen35.attention.layer_norm_rms_epsilon", f32(9.999_999_974_752_427e-7f));
    entries.put("qwen35.rope.freq_base", f32(10_000_000.0f));
    entries.put("qwen35.rope.dimension_count", uint(64));
    entries.put("qwen35.ssm.conv_kernel", uint(4));
    entries.put("qwen35.ssm.state_size", uint(128));
    entries.put("qwen35.ssm.group_count", uint(16));
    entries.put("qwen35.ssm.time_step_rank", uint(32));
    entries.put("qwen35.ssm.inner_size", uint(4_096));
    entries.put("qwen35.full_attention_interval", uint(4));
    entries.put("qwen35.vocab_size", uint(248_320));
    return entries;
  }

  private static GgufMetadataValue.Uint32Value uint(int value) {
    return new GgufMetadataValue.Uint32Value(value);
  }

  private static GgufMetadataValue.Float32Value f32(float value) {
    return new GgufMetadataValue.Float32Value(value);
  }
}
