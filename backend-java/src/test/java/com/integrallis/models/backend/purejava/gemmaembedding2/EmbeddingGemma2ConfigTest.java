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
package com.integrallis.models.backend.purejava.gemmaembedding2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class EmbeddingGemma2ConfigTest {
  @Test
  void readsTheHeaderShapesWithoutConfusingHiddenAndOutputWidths() {
    var config = EmbeddingGemma2Config.fromMetadata(new GgufMetadata(entries()));
    assertThat(config.embeddingDim()).isEqualTo(512);
    assertThat(config.outputDim()).isEqualTo(768);
    assertThat(config.headDim(0)).isEqualTo(256);
    assertThat(config.headDim(5)).isEqualTo(512);
    assertThat(config.kvHeads().get(0)).isEqualTo(2);
    assertThat(config.kvHeads().get(5)).isEqualTo(1);
  }

  @Test
  void rejectsCausalPoolingSharingAndMismatchedHeadContracts() {
    for (String key :
        List.of(
            "attention.causal",
            "pooling_type",
            "attention.shared_kv_layers",
            "attention.value_length")) {
      var entries = entries();
      entries.put(
          "gemma-embedding2." + key,
          key.equals("attention.causal")
              ? new GgufMetadataValue.BoolValue(true)
              : new GgufMetadataValue.Uint32Value(2));
      assertThatThrownBy(() -> EmbeddingGemma2Config.fromMetadata(new GgufMetadata(entries)))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void rejectsIncompleteOrInvalidShapesAndNonfiniteEpsilon() {
    for (String key : List.of("embedding_length", "block_count", "attention.key_length_swa")) {
      var entries = entries();
      entries.remove("gemma-embedding2." + key);
      assertThatThrownBy(() -> EmbeddingGemma2Config.fromMetadata(new GgufMetadata(entries)))
          .isInstanceOf(IllegalArgumentException.class);
    }
    var entries = entries();
    entries.put(
        "gemma-embedding2.attention.layer_norm_rms_epsilon",
        new GgufMetadataValue.Float32Value(Float.NaN));
    assertThatThrownBy(() -> EmbeddingGemma2Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static Map<String, GgufMetadataValue> entries() {
    var entries = new HashMap<String, GgufMetadataValue>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma-embedding2"));
    entries.put("gemma-embedding2.block_count", new GgufMetadataValue.Uint32Value(24));
    entries.put("gemma-embedding2.context_length", new GgufMetadataValue.Uint32Value(262144));
    entries.put("gemma-embedding2.embedding_length", new GgufMetadataValue.Uint32Value(512));
    entries.put("gemma-embedding2.feed_forward_length", new GgufMetadataValue.Uint32Value(2048));
    entries.put("gemma-embedding2.attention.head_count", new GgufMetadataValue.Uint32Value(4));
    entries.put(
        "gemma-embedding2.attention.head_count_kv",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.INT32,
            List.of(
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(1),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(1),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(1),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(2),
                new GgufMetadataValue.Int32Value(1))));
    entries.put("gemma-embedding2.rope.freq_base", new GgufMetadataValue.Float32Value(1000000.0f));
    entries.put(
        "gemma-embedding2.rope.freq_base_swa", new GgufMetadataValue.Float32Value(10000.0f));
    entries.put(
        "gemma-embedding2.attention.layer_norm_rms_epsilon",
        new GgufMetadataValue.Float32Value(9.999999974752427e-07f));
    entries.put("gemma-embedding2.attention.key_length", new GgufMetadataValue.Uint32Value(512));
    entries.put("gemma-embedding2.attention.value_length", new GgufMetadataValue.Uint32Value(512));
    entries.put(
        "gemma-embedding2.attention.sliding_window", new GgufMetadataValue.Uint32Value(1024));
    entries.put(
        "gemma-embedding2.attention.shared_kv_layers", new GgufMetadataValue.Uint32Value(0));
    entries.put(
        "gemma-embedding2.embedding_length_per_layer_input",
        new GgufMetadataValue.Uint32Value(512));
    entries.put(
        "gemma-embedding2.attention.sliding_window_pattern",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.BOOL,
            List.of(
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(false),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(false),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(false),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(true),
                new GgufMetadataValue.BoolValue(false))));
    entries.put(
        "gemma-embedding2.attention.key_length_swa", new GgufMetadataValue.Uint32Value(256));
    entries.put(
        "gemma-embedding2.attention.value_length_swa", new GgufMetadataValue.Uint32Value(256));
    entries.put("gemma-embedding2.rope.dimension_count", new GgufMetadataValue.Uint32Value(512));
    entries.put(
        "gemma-embedding2.rope.dimension_count_swa", new GgufMetadataValue.Uint32Value(256));
    entries.put("gemma-embedding2.embedding_length_out", new GgufMetadataValue.Uint32Value(768));
    entries.put("gemma-embedding2.attention.causal", new GgufMetadataValue.BoolValue(false));
    entries.put("gemma-embedding2.pooling_type", new GgufMetadataValue.Uint32Value(1));
    // Token content belongs to the real fixture gate; this metadata test needs only its length.
    entries.put(
        "tokenizer.ggml.tokens",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING, List.of(new GgufMetadataValue.StringValue("token"))));
    return entries;
  }
}
