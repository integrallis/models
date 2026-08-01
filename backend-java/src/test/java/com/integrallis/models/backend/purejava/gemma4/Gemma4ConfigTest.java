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
package com.integrallis.models.backend.purejava.gemma4;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Gemma4ConfigTest {

  @Test
  void parsesThePinnedGemma426BA4BMetadata() {
    Gemma4Config config = Gemma4Config.fromMetadata(metadata());

    assertThat(config.embeddingDim()).isEqualTo(2_816);
    assertThat(config.numLayers()).isEqualTo(30);
    assertThat(config.numHeads()).isEqualTo(16);
    assertThat(config.vocabSize()).isEqualTo(262_144);
    assertThat(config.contextLength()).isEqualTo(262_144);
    assertThat(config.sharedHiddenDim()).isEqualTo(2_112);
    assertThat(config.expertHiddenDim()).isEqualTo(704);
    assertThat(config.numExperts()).isEqualTo(128);
    assertThat(config.numExpertsUsed()).isEqualTo(8);
    assertThat(config.slidingWindow()).isEqualTo(1_024);
    assertThat(config.rmsNormEps()).isEqualTo(1.0e-6f);
    assertThat(config.finalLogitSoftcap()).isEqualTo(30.0f);
    assertThat(config.embeddingScale()).isEqualTo((float) Math.sqrt(2_816));
    assertThat(config.attentionScale()).isEqualTo(1.0f);

    assertThat(config.usesSlidingWindow(0)).isTrue();
    assertThat(config.numKvHeads(0)).isEqualTo(8);
    assertThat(config.headDim(0)).isEqualTo(256);
    assertThat(config.keyDim(0)).isEqualTo(2_048);
    assertThat(config.valueDim(0)).isEqualTo(2_048);
    assertThat(config.ropeDimension(0)).isEqualTo(256);
    assertThat(config.ropeTheta(0)).isEqualTo(10_000.0f);

    assertThat(config.usesSlidingWindow(5)).isFalse();
    assertThat(config.numKvHeads(5)).isEqualTo(2);
    assertThat(config.headDim(5)).isEqualTo(512);
    assertThat(config.keyDim(5)).isEqualTo(1_024);
    assertThat(config.valueDim(5)).isEqualTo(1_024);
    assertThat(config.ropeDimension(5)).isEqualTo(512);
    assertThat(config.ropeTheta(5)).isEqualTo(1_000_000.0f);

    assertThat(config.fullAttentionLayers()).containsExactly(5, 11, 17, 23, 29);
  }

  @Test
  void rejectsPerLayerArraysThatDoNotMatchTheBlockCount() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma4.attention.head_count_kv", intArray(List.of(8, 8)));

    assertThatThrownBy(() -> Gemma4Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("gemma4.attention.head_count_kv")
        .hasMessageContaining("30");
  }

  @Test
  void rejectsGemma4VariantsThatNeedUnsupportedSharedKvLayers() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma4.attention.shared_kv_layers", new GgufMetadataValue.Uint32Value(1));

    assertThatThrownBy(() -> Gemma4Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("shared_kv_layers=1");
  }

  @Test
  void rejectsGemma4VariantsThatNeedPerLayerEmbeddings() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma4.embedding_length_per_layer_input", new GgufMetadataValue.Uint32Value(256));

    assertThatThrownBy(() -> Gemma4Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("embedding_length_per_layer_input=256");
  }

  private static GgufMetadata metadata() {
    return new GgufMetadata(entries());
  }

  private static Map<String, GgufMetadataValue> entries() {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma4"));
    entries.put("gemma4.embedding_length", new GgufMetadataValue.Uint32Value(2_816));
    entries.put("gemma4.block_count", new GgufMetadataValue.Uint32Value(30));
    entries.put("gemma4.attention.head_count", new GgufMetadataValue.Uint32Value(16));
    entries.put("gemma4.attention.head_count_kv", intArray(headCounts()));
    entries.put("gemma4.attention.key_length", new GgufMetadataValue.Uint32Value(512));
    entries.put("gemma4.attention.key_length_swa", new GgufMetadataValue.Uint32Value(256));
    entries.put("gemma4.attention.value_length", new GgufMetadataValue.Uint32Value(512));
    entries.put("gemma4.attention.value_length_swa", new GgufMetadataValue.Uint32Value(256));
    entries.put(
        "gemma4.attention.layer_norm_rms_epsilon", new GgufMetadataValue.Float32Value(1.0e-6f));
    entries.put("gemma4.attention.shared_kv_layers", new GgufMetadataValue.Uint32Value(0));
    entries.put("gemma4.attention.sliding_window", new GgufMetadataValue.Uint32Value(1_024));
    entries.put("gemma4.attention.sliding_window_pattern", boolArray(slidingWindowPattern()));
    entries.put("gemma4.context_length", new GgufMetadataValue.Uint32Value(262_144));
    entries.put("gemma4.embedding_length_per_layer_input", new GgufMetadataValue.Uint32Value(0));
    entries.put("gemma4.expert_count", new GgufMetadataValue.Uint32Value(128));
    entries.put("gemma4.expert_feed_forward_length", new GgufMetadataValue.Uint32Value(704));
    entries.put("gemma4.expert_used_count", new GgufMetadataValue.Uint32Value(8));
    entries.put("gemma4.feed_forward_length", new GgufMetadataValue.Uint32Value(2_112));
    entries.put("gemma4.final_logit_softcapping", new GgufMetadataValue.Float32Value(30.0f));
    entries.put("gemma4.rope.dimension_count", new GgufMetadataValue.Uint32Value(512));
    entries.put("gemma4.rope.dimension_count_swa", new GgufMetadataValue.Uint32Value(256));
    entries.put("gemma4.rope.freq_base", new GgufMetadataValue.Float32Value(1_000_000.0f));
    entries.put("gemma4.rope.freq_base_swa", new GgufMetadataValue.Float32Value(10_000.0f));
    entries.put("tokenizer.ggml.tokens", stringArray(262_144));
    return entries;
  }

  private static List<Integer> headCounts() {
    List<Integer> values = new ArrayList<>(30);
    for (int layer = 0; layer < 30; layer++) {
      values.add(isFullAttention(layer) ? 2 : 8);
    }
    return values;
  }

  private static List<Boolean> slidingWindowPattern() {
    List<Boolean> values = new ArrayList<>(30);
    for (int layer = 0; layer < 30; layer++) {
      values.add(!isFullAttention(layer));
    }
    return values;
  }

  private static boolean isFullAttention(int layer) {
    return (layer + 1) % 6 == 0;
  }

  private static GgufMetadataValue.ArrayValue intArray(List<Integer> values) {
    return new GgufMetadataValue.ArrayValue(
        GgufValueType.INT32,
        values.stream()
            .map(GgufMetadataValue.Int32Value::new)
            .map(GgufMetadataValue.class::cast)
            .toList());
  }

  private static GgufMetadataValue.ArrayValue boolArray(List<Boolean> values) {
    return new GgufMetadataValue.ArrayValue(
        GgufValueType.BOOL,
        values.stream()
            .map(GgufMetadataValue.BoolValue::new)
            .map(GgufMetadataValue.class::cast)
            .toList());
  }

  private static GgufMetadataValue.ArrayValue stringArray(int size) {
    List<GgufMetadataValue> values = new ArrayList<>(size);
    for (int token = 0; token < size; token++) {
      values.add(new GgufMetadataValue.StringValue("token-" + token));
    }
    return new GgufMetadataValue.ArrayValue(GgufValueType.STRING, values);
  }
}
