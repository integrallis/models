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

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The LFM2 contract.
 *
 * <p>The fixture mirrors LFM2.5-1.2B as published: sixteen layers of which six attend, at the exact
 * indices the file declares rather than at a computed interval.
 */
@Tag("unit")
class Lfm2ConfigTest {

  private static final List<Integer> ATTENTION_LAYERS = List.of(2, 5, 8, 10, 12, 14);

  @Test
  void readsTheHybridLayerPlanFromTheFile() {
    Lfm2Config config = Lfm2Config.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.numLayers()).isEqualTo(16);
    assertThat(config.embeddingDim()).isEqualTo(2_048);
    assertThat(config.numHeads()).isEqualTo(32);
    assertThat(config.headDim())
        .describedAs("no key_length is published, so it derives from the width and head count")
        .isEqualTo(64);
    assertThat(config.hiddenDim()).isEqualTo(8_192);
    assertThat(config.vocabSize()).isEqualTo(65_536);
    assertThat(config.shortConvCache()).isEqualTo(3);

    // Six of sixteen attend, and a zero kv head count is what marks a convolutional layer.
    assertThat(config.attentionLayers()).isEqualTo(6);
    for (int layer = 0; layer < 16; layer++) {
      assertThat(config.usesAttention(layer))
          .describedAs("layer %s", layer)
          .isEqualTo(ATTENTION_LAYERS.contains(layer));
    }
    assertThat(config.numKvHeads(2)).isEqualTo(8);
    assertThatThrownBy(() -> config.numKvHeads(0))
        .describedAs("a convolutional layer has no kv heads to report")
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("convolutional");
  }

  @Test
  void reportsTheRecurrentStateWidthPerConvolutionalLayer() {
    Lfm2Config config = Lfm2Config.fromMetadata(new GgufMetadata(entries()));

    // (l_cache - 1) past values per channel: the convolution needs a shift register, not a
    // key-value cache, which is the reason this architecture cannot reuse the Llama decoder.
    assertThat(config.shortConvStateSize()).isEqualTo(2 * 2_048);
    assertThat(config.queryDim()).isEqualTo(32 * 64);
    assertThat(config.keyDim(2)).isEqualTo(8 * 64);
  }

  @Test
  void refusesAScalarKvHeadCountBecauseItCannotSayWhichLayersAreConvolutional() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("lfm2.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));

    assertThatThrownBy(() -> Lfm2Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("per layer");
  }

  @Test
  void refusesAModelWithNoAttentionAtAll() {
    Map<String, GgufMetadataValue> entries = entries();
    List<Integer> allConvolutional = new ArrayList<>();
    for (int layer = 0; layer < 16; layer++) {
      allConvolutional.add(0);
    }
    entries.put("lfm2.attention.head_count_kv", intArray(allConvolutional));

    // Accepting one would mean never allocating a key-value cache and never noticing.
    assertThatThrownBy(() -> Lfm2Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("at least one attention layer");
  }

  @Test
  void refusesAConvolutionWidthBelowTwo() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("lfm2.shortconv.l_cache", new GgufMetadataValue.Uint32Value(1));

    assertThatThrownBy(() -> Lfm2Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("l_cache must be at least 2");
  }

  @Test
  void refusesAPerLayerListThatDoesNotMatchTheBlockCount() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("lfm2.attention.head_count_kv", intArray(List.of(0, 8)));

    assertThatThrownBy(() -> Lfm2Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("one entry per layer");
  }

  @Test
  void treatsAFileWithNoCausalityKeyAsADecoder() {
    Lfm2Config config = Lfm2Config.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.causalAttention())
        .describedAs(
            "llama_hparams initialises causal_attn to true and reads the key as optional, so an"
                + " absent key must not turn a generative model into an encoder")
        .isTrue();
    assertThat(config.encodesWholeSequence()).isFalse();
    assertThat(config.pooling()).isEqualTo(Lfm2Config.Pooling.NONE);
  }

  @Test
  void readsTheBidirectionalEmbeddingContract() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("lfm2.attention.causal", new GgufMetadataValue.BoolValue(false));
    entries.put("lfm2.pooling_type", new GgufMetadataValue.Uint32Value(2));

    Lfm2Config config = Lfm2Config.fromMetadata(new GgufMetadata(entries));

    // The two keys LFM2.5-Embedding-350M publishes, read from that artifact.
    assertThat(config.causalAttention()).isFalse();
    assertThat(config.encodesWholeSequence()).isTrue();
    assertThat(config.pooling()).isEqualTo(Lfm2Config.Pooling.CLS);
  }

  @Test
  void refusesABidirectionalFileThatDeclaresNoPooling() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("lfm2.attention.causal", new GgufMetadataValue.BoolValue(false));

    assertThatThrownBy(() -> Lfm2Config.fromMetadata(new GgufMetadata(entries)))
        .describedAs("such a file can be served neither generatively nor as an embedder")
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("declares no pooling");
  }

  @Test
  void refusesAPoolingCodeItCannotName() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("lfm2.pooling_type", new GgufMetadataValue.Uint32Value(9));

    assertThatThrownBy(() -> Lfm2Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("unsupported LFM2 pooling type: 9");
  }

  private static Map<String, GgufMetadataValue> entries() {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("lfm2"));
    entries.put("lfm2.embedding_length", new GgufMetadataValue.Uint32Value(2_048));
    entries.put("lfm2.block_count", new GgufMetadataValue.Uint32Value(16));
    entries.put("lfm2.attention.head_count", new GgufMetadataValue.Uint32Value(32));
    entries.put("lfm2.attention.head_count_kv", intArray(headCounts()));
    entries.put(
        "lfm2.attention.layer_norm_rms_epsilon", new GgufMetadataValue.Float32Value(1.0e-5f));
    entries.put("lfm2.context_length", new GgufMetadataValue.Uint32Value(128_000));
    entries.put("lfm2.feed_forward_length", new GgufMetadataValue.Uint32Value(8_192));
    entries.put("lfm2.rope.freq_base", new GgufMetadataValue.Float32Value(1_000_000.0f));
    entries.put("lfm2.shortconv.l_cache", new GgufMetadataValue.Uint32Value(3));
    entries.put("lfm2.vocab_size", new GgufMetadataValue.Uint32Value(65_536));
    return entries;
  }

  private static List<Integer> headCounts() {
    List<Integer> values = new ArrayList<>();
    for (int layer = 0; layer < 16; layer++) {
      values.add(ATTENTION_LAYERS.contains(layer) ? 8 : 0);
    }
    return values;
  }

  private static GgufMetadataValue.ArrayValue intArray(List<Integer> values) {
    return new GgufMetadataValue.ArrayValue(
        GgufValueType.INT32,
        values.stream()
            .map(GgufMetadataValue.Int32Value::new)
            .map(GgufMetadataValue.class::cast)
            .toList());
  }
}
