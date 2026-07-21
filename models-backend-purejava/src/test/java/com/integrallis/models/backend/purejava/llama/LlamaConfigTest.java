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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LlamaConfigTest {

  @Test
  void doesNotExposeEqualWidthKvCompatibilityAccessor() {
    assertThatThrownBy(() -> LlamaConfig.class.getDeclaredMethod("kvDim"))
        .isInstanceOf(NoSuchMethodException.class);
  }

  private GgufMetadata createLlamaMetadata(
      int embeddingDim, int blockCount, int headCount, int headCountKv) {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("llama.embedding_length", new GgufMetadataValue.Uint32Value(embeddingDim));
    entries.put("llama.block_count", new GgufMetadataValue.Uint32Value(blockCount));
    entries.put("llama.attention.head_count", new GgufMetadataValue.Uint32Value(headCount));
    entries.put("llama.attention.head_count_kv", new GgufMetadataValue.Uint32Value(headCountKv));
    return new GgufMetadata(entries);
  }

  @Nested
  class FromMetadata {

    @Test
    void extractsFromMetadata() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("llama.embedding_length", new GgufMetadataValue.Uint32Value(2048));
      entries.put("llama.block_count", new GgufMetadataValue.Uint32Value(22));
      entries.put("llama.attention.head_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("llama.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));
      entries.put("llama.vocab_size", new GgufMetadataValue.Uint32Value(32000));
      entries.put("llama.context_length", new GgufMetadataValue.Uint32Value(4096));
      entries.put("llama.feed_forward_length", new GgufMetadataValue.Uint32Value(5504));
      entries.put("llama.rope.freq_base", new GgufMetadataValue.Float32Value(500000.0f));
      GgufMetadata metadata = new GgufMetadata(entries);

      LlamaConfig config = LlamaConfig.fromMetadata(metadata);

      assertThat(config.embeddingDim()).isEqualTo(2048);
      assertThat(config.numLayers()).isEqualTo(22);
      assertThat(config.numHeads()).isEqualTo(32);
      assertThat(config.numKvHeads()).isEqualTo(8);
      assertThat(config.vocabSize()).isEqualTo(32000);
      assertThat(config.contextLength()).isEqualTo(4096);
      assertThat(config.hiddenDim()).isEqualTo(5504);
      assertThat(config.ropeTheta()).isEqualTo(500000.0f);
      assertThat(config.headDim()).isEqualTo(64); // 2048/32
      assertThat(config.keyDim()).isEqualTo(512); // 64*8
      assertThat(config.valueDim()).isEqualTo(512); // 64*8
    }

    @Test
    void missingEmbeddingLengthThrows() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("llama.block_count", new GgufMetadataValue.Uint32Value(22));
      entries.put("llama.attention.head_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("llama.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));

      assertThatThrownBy(() -> LlamaConfig.fromMetadata(new GgufMetadata(entries)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("llama.embedding_length");
    }

    @Test
    void usesDefaults() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("llama.embedding_length", new GgufMetadataValue.Uint32Value(512));
      entries.put("llama.block_count", new GgufMetadataValue.Uint32Value(4));
      entries.put("llama.attention.head_count", new GgufMetadataValue.Uint32Value(8));
      entries.put("llama.attention.head_count_kv", new GgufMetadataValue.Uint32Value(2));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.vocabSize()).isEqualTo(32000);
      assertThat(config.contextLength()).isEqualTo(2048);
      assertThat(config.ropeTheta()).isEqualTo(10000.0f);
      assertThat(config.ropeFrequencyScale()).isEqualTo(1.0f);
      assertThat(config.rmsNormEps()).isEqualTo(1e-5f);
    }

    @Test
    void readsLegacyLinearRopeScale() {
      Map<String, GgufMetadataValue> entries = requiredLlamaEntries();
      entries.put("llama.rope.scale_linear", new GgufMetadataValue.Float32Value(4.0f));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.ropeFrequencyScale()).isEqualTo(0.25f);
    }

    @Test
    void readsModernLinearRopeScalingFactor() {
      Map<String, GgufMetadataValue> entries = requiredLlamaEntries();
      entries.put("llama.rope.scaling.type", new GgufMetadataValue.StringValue("linear"));
      entries.put("llama.rope.scaling.factor", new GgufMetadataValue.Float32Value(8.0f));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.ropeFrequencyScale()).isEqualTo(0.125f);
    }

    @Test
    void readsExplicitQwenHeadWidthsAndRopeLayout() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("qwen3"));
      entries.put("qwen3.embedding_length", new GgufMetadataValue.Uint32Value(1024));
      entries.put("qwen3.block_count", new GgufMetadataValue.Uint32Value(28));
      entries.put("qwen3.attention.head_count", new GgufMetadataValue.Uint32Value(16));
      entries.put("qwen3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));
      entries.put("qwen3.attention.key_length", new GgufMetadataValue.Uint32Value(128));
      entries.put("qwen3.attention.value_length", new GgufMetadataValue.Uint32Value(128));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.keyLength()).isEqualTo(128);
      assertThat(config.valueLength()).isEqualTo(128);
      assertThat(config.queryDim()).isEqualTo(2048);
      assertThat(config.keyDim()).isEqualTo(1024);
      assertThat(config.valueDim()).isEqualTo(1024);
      assertThat(config.attentionOutputDim()).isEqualTo(2048);
      assertThat(config.usesNeoxRope()).isTrue();
    }

    @Test
    void selectsSmolLm3LayersThatSkipRope() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("smollm3"));
      entries.put("smollm3.embedding_length", new GgufMetadataValue.Uint32Value(2048));
      entries.put("smollm3.block_count", new GgufMetadataValue.Uint32Value(36));
      entries.put("smollm3.attention.head_count", new GgufMetadataValue.Uint32Value(16));
      entries.put("smollm3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(4));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.usesRope(0)).isTrue();
      assertThat(config.usesRope(1)).isTrue();
      assertThat(config.usesRope(2)).isTrue();
      assertThat(config.usesRope(3)).isFalse();
      assertThat(config.usesRope(7)).isFalse();
      assertThat(config.usesRope(35)).isFalse();
    }

    @Test
    void derivesVocabSizeFromTokenizerTokensWhenArchitectureKeyIsMissing() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("qwen3"));
      entries.put("qwen3.embedding_length", new GgufMetadataValue.Uint32Value(1024));
      entries.put("qwen3.block_count", new GgufMetadataValue.Uint32Value(28));
      entries.put("qwen3.attention.head_count", new GgufMetadataValue.Uint32Value(16));
      entries.put("qwen3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));
      entries.put(
          "tokenizer.ggml.tokens",
          new GgufMetadataValue.ArrayValue(
              GgufValueType.STRING,
              List.of(
                  new GgufMetadataValue.StringValue("a"),
                  new GgufMetadataValue.StringValue("b"),
                  new GgufMetadataValue.StringValue("c"))));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.vocabSize()).isEqualTo(3);
    }

    private static Map<String, GgufMetadataValue> requiredLlamaEntries() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("llama.embedding_length", new GgufMetadataValue.Uint32Value(512));
      entries.put("llama.block_count", new GgufMetadataValue.Uint32Value(4));
      entries.put("llama.attention.head_count", new GgufMetadataValue.Uint32Value(8));
      entries.put("llama.attention.head_count_kv", new GgufMetadataValue.Uint32Value(2));
      return entries;
    }
  }
}
