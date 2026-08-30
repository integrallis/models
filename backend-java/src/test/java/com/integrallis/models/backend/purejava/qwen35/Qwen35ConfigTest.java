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

@Tag("unit")
class Qwen35ConfigTest {

  @Test
  void parsesThePinnedDense08BShape() {
    Qwen35Config config = Qwen35Config.fromMetadata(metadata());

    assertThat(config.embeddingDim()).isEqualTo(1_024);
    assertThat(config.numLayers()).isEqualTo(24);
    assertThat(config.numHeads()).isEqualTo(8);
    assertThat(config.numKvHeads()).isEqualTo(2);
    assertThat(config.attentionHeadDim()).isEqualTo(256);
    assertThat(config.hiddenDim()).isEqualTo(3_584);
    assertThat(config.vocabSize()).isEqualTo(248_320);
    assertThat(config.contextLength()).isEqualTo(262_144);
    assertThat(config.gdnKeyHeads()).isEqualTo(16);
    assertThat(config.gdnValueHeads()).isEqualTo(16);
    assertThat(config.gdnHeadDim()).isEqualTo(128);
    assertThat(config.gdnKeyDim()).isEqualTo(2_048);
    assertThat(config.gdnValueDim()).isEqualTo(2_048);
    assertThat(config.gdnConvDim()).isEqualTo(6_144);
    assertThat(config.gdnConvKernel()).isEqualTo(4);
    assertThat(config.ropeDimension()).isEqualTo(64);
    assertThat(config.fullAttentionLayers()).containsExactly(3, 7, 11, 15, 19, 23);
  }

  @Test
  void rejectsIncompatibleGdnHeadWidths() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("qwen35.ssm.inner_size", uint(4_096));

    assertThatThrownBy(() -> Qwen35Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Gated DeltaNet key and value head dimensions");
  }

  @Test
  void rejectsAnArchitectureThatOnlyLooksSimilar() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("qwen3"));

    assertThatThrownBy(() -> Qwen35Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("general.architecture=qwen35");
  }

  private static GgufMetadata metadata() {
    return new GgufMetadata(entries());
  }

  private static Map<String, GgufMetadataValue> entries() {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("qwen35"));
    entries.put("qwen35.block_count", uint(24));
    entries.put("qwen35.context_length", uint(262_144));
    entries.put("qwen35.embedding_length", uint(1_024));
    entries.put("qwen35.feed_forward_length", uint(3_584));
    entries.put("qwen35.attention.head_count", uint(8));
    entries.put("qwen35.attention.head_count_kv", uint(2));
    entries.put("qwen35.attention.key_length", uint(256));
    entries.put("qwen35.attention.value_length", uint(256));
    entries.put("qwen35.attention.layer_norm_rms_epsilon", f32(1.0e-6f));
    entries.put("qwen35.rope.freq_base", f32(10_000_000.0f));
    entries.put("qwen35.rope.dimension_count", uint(64));
    entries.put("qwen35.ssm.conv_kernel", uint(4));
    entries.put("qwen35.ssm.state_size", uint(128));
    entries.put("qwen35.ssm.group_count", uint(16));
    entries.put("qwen35.ssm.time_step_rank", uint(16));
    entries.put("qwen35.ssm.inner_size", uint(2_048));
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
