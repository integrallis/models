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
package com.integrallis.models.backend.purejava.bert;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BertConfigTest {

  @Test
  void loadsTheMiniLmGgufContract() {
    BertConfig config = BertConfig.fromMetadata(miniLmMetadata());

    assertThat(config.embeddingDim()).isEqualTo(384);
    assertThat(config.numLayers()).isEqualTo(6);
    assertThat(config.numHeads()).isEqualTo(12);
    assertThat(config.headDim()).isEqualTo(32);
    assertThat(config.vocabSize()).isEqualTo(30_522);
    assertThat(config.contextLength()).isEqualTo(512);
    assertThat(config.hiddenDim()).isEqualTo(1_536);
    assertThat(config.layerNormEps()).isEqualTo(1.0e-12f);
    assertThat(config.pooling()).isEqualTo(BertConfig.Pooling.MEAN);
  }

  @Test
  void rejectsCausalMetadataInsteadOfSilentlyChangingTheModel() {
    Map<String, GgufMetadataValue> entries = new HashMap<>(miniLmMetadata().entries());
    entries.put("bert.attention.causal", new GgufMetadataValue.BoolValue(true));

    assertThatThrownBy(() -> BertConfig.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("bidirectional");
  }

  @Test
  void rejectsAnUnknownPoolingContract() {
    Map<String, GgufMetadataValue> entries = new HashMap<>(miniLmMetadata().entries());
    entries.put("bert.pooling_type", new GgufMetadataValue.Uint32Value(99));

    assertThatThrownBy(() -> BertConfig.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("pooling");
  }

  private static GgufMetadata miniLmMetadata() {
    Map<String, GgufMetadataValue> entries = new HashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("bert"));
    entries.put("bert.embedding_length", new GgufMetadataValue.Uint32Value(384));
    entries.put("bert.block_count", new GgufMetadataValue.Uint32Value(6));
    entries.put("bert.attention.head_count", new GgufMetadataValue.Uint32Value(12));
    entries.put("bert.context_length", new GgufMetadataValue.Uint32Value(512));
    entries.put("bert.feed_forward_length", new GgufMetadataValue.Uint32Value(1_536));
    entries.put("bert.attention.layer_norm_epsilon", new GgufMetadataValue.Float32Value(1e-12f));
    entries.put("bert.attention.causal", new GgufMetadataValue.BoolValue(false));
    entries.put("bert.pooling_type", new GgufMetadataValue.Uint32Value(1));
    List<GgufMetadataValue> tokens = new ArrayList<>(30_522);
    for (int token = 0; token < 30_522; token++) {
      tokens.add(new GgufMetadataValue.StringValue("token-" + token));
    }
    entries.put(
        "tokenizer.ggml.tokens", new GgufMetadataValue.ArrayValue(GgufValueType.STRING, tokens));
    return new GgufMetadata(entries);
  }
}
