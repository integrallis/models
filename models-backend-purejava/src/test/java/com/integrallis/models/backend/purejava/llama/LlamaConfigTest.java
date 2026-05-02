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
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LlamaConfigTest {

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
  static class FromMetadata {

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
      assertThat(config.kvDim()).isEqualTo(512); // 64*8
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
      assertThat(config.rmsNormEps()).isEqualTo(1e-5f);
    }
  }
}
