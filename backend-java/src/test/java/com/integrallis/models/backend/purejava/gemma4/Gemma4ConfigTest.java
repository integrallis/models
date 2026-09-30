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
    assertThat(config.sharedHiddenDim(0)).isEqualTo(2_112);
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
  void sharedKeyValueLayersAreParsedAndMappedToTheSourceTheReferenceUses() {
    // These two keys used to be refused outright. The E-series declares both, so refusing them
    // rejected the whole variant before any decoder ran.
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma4.attention.shared_kv_layers", new GgufMetadataValue.Uint32Value(12));
    Gemma4Config config = Gemma4Config.fromMetadata(new GgufMetadata(entries));

    // llama.cpp: n_layer_kv_from_start = n_layer - shared_kv_layers.
    assertThat(config.kvOwningLayers()).isEqualTo(18);
    assertThat(config.ownsKvCache(0)).isTrue();
    assertThat(config.ownsKvCache(17)).isTrue();
    assertThat(config.ownsKvCache(18)).isFalse();
    assertThat(config.ownsKvCache(29)).isFalse();
    // An owning layer is its own source.
    assertThat(config.kvSourceLayer(5)).isEqualTo(5);
    // A sharing layer reads kvOwningLayers - (sliding ? 2 : 1): the last owning layer of its own
    // attention type. In this fixture 16 is sliding and 17 is full.
    assertThat(config.usesSlidingWindow(18)).isTrue();
    assertThat(config.kvSourceLayer(18)).isEqualTo(16);
    assertThat(config.usesSlidingWindow(23)).isFalse();
    assertThat(config.kvSourceLayer(23)).isEqualTo(17);
    // Which is the point of the -2/-1: the shapes must match, or the cache would be misread.
    assertThat(config.headDim(18)).isEqualTo(config.headDim(16));
    assertThat(config.headDim(23)).isEqualTo(config.headDim(17));
    assertThat(config.numKvHeads(18)).isEqualTo(config.numKvHeads(16));
  }

  @Test
  void aSharedKeyValueMappingOntoAMismatchedLayerIsRefused() {
    // 30 - 11 = 19 owning layers, so a sliding sharing layer would read layer 17, which is a FULL
    // layer with a 512 head dimension against its own 256. Reading that cache would produce
    // plausible numbers from misaligned bytes, so it is refused.
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma4.attention.shared_kv_layers", new GgufMetadataValue.Uint32Value(11));
    Gemma4Config config = Gemma4Config.fromMetadata(new GgufMetadata(entries));

    assertThatThrownBy(() -> config.kvSourceLayer(19))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("attention shapes differ");
  }

  @Test
  void perLayerInputEmbeddingsAreParsedRatherThanRefused() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma4.embedding_length_per_layer_input", new GgufMetadataValue.Uint32Value(256));

    Gemma4Config config = Gemma4Config.fromMetadata(new GgufMetadata(entries));

    assertThat(config.perLayerEmbeddingDim()).isEqualTo(256);
    assertThat(config.usesPerLayerEmbeddings()).isTrue();
    // And a variant without them is unchanged: absent means no second embedding table.
    assertThat(Gemma4Config.fromMetadata(metadata()).usesPerLayerEmbeddings()).isFalse();
    assertThat(Gemma4Config.fromMetadata(metadata()).kvOwningLayers())
        .describedAs("shared_kv_layers=0 means every layer owns its cache")
        .isEqualTo(30);
  }

  @Test
  void aScalarHeadCountKvIsBroadcastOverEveryLayer() {
    // The E-series writes this as a single uint32; the dense and routed variants write an array.
    // Demanding the array form is what rejected the E-series first.
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma4.attention.head_count_kv", new GgufMetadataValue.Uint32Value(1));

    Gemma4Config config = Gemma4Config.fromMetadata(new GgufMetadata(entries));

    assertThat(config.kvHeadsByLayer()).hasSize(30).containsOnly(1);
    // The array form still wins when present, so per-layer values are not flattened.
    assertThat(Gemma4Config.fromMetadata(metadata()).kvHeadsByLayer()).isEqualTo(headCounts());
    assertThat(new java.util.HashSet<>(headCounts()))
        .describedAs("the fixture's array genuinely varies, so a broadcast would have flattened it")
        .hasSizeGreaterThan(1);
  }

  @Test
  void aPerLayerFeedForwardLengthArrayIsAccepted() {
    // The E-series publishes feed_forward_length per layer; the others publish one value.
    Map<String, GgufMetadataValue> entries = entries();
    List<Integer> widths = new ArrayList<>();
    for (int layer = 0; layer < 30; layer++) {
      widths.add(6_144);
    }
    entries.put("gemma4.feed_forward_length", intArray(widths));

    assertThat(Gemma4Config.fromMetadata(new GgufMetadata(entries)).sharedHiddenDim(0))
        .isEqualTo(6_144);
  }

  /**
   * A feed-forward width that <b>differs</b> between layers, which is what an E-series MatFormer
   * publishes and what the uniform test above cannot distinguish from a broadcast scalar.
   *
   * <p>Gemma 4 E2B was refused at load with "blk.15.ffn_gate.weight shape must be [1536, 6144],
   * found [1536, 12288]": the config read element 0 of the array and called it the model's width.
   * The test above passed throughout, because every element of its array is the same number.
   */
  @Test
  void aFeedForwardLengthThatVariesPerLayerIsKeptPerLayer() {
    Map<String, GgufMetadataValue> entries = entries();
    List<Integer> widths = new ArrayList<>();
    for (int layer = 0; layer < 30; layer++) {
      widths.add(layer == 15 ? 12_288 : 6_144);
    }
    entries.put("gemma4.feed_forward_length", intArray(widths));

    Gemma4Config config = Gemma4Config.fromMetadata(new GgufMetadata(entries));

    assertThat(config.sharedHiddenDim(0)).isEqualTo(6_144);
    assertThat(config.sharedHiddenDim(15))
        .describedAs("the wide layer must keep its own width")
        .isEqualTo(12_288);
    assertThat(config.sharedHiddenDim(29)).isEqualTo(6_144);
    // Buffers serving every layer need the widest, which is not layer 0's width.
    assertThat(config.maxSharedHiddenDim()).isEqualTo(12_288);
    assertThat(config.sharedHiddenDimByLayer()).hasSize(30);
  }

  /**
   * A scalar feed_forward_length still broadcasts, and then every layer agrees with the maximum.
   */
  @Test
  void aScalarFeedForwardLengthBroadcastsAcrossEveryLayer() {
    Gemma4Config config = Gemma4Config.fromMetadata(metadata());

    assertThat(config.sharedHiddenDimByLayer()).hasSize(config.numLayers());
    assertThat(config.maxSharedHiddenDim()).isEqualTo(config.sharedHiddenDim(0));
    for (int layer = 0; layer < config.numLayers(); layer++) {
      assertThat(config.sharedHiddenDim(layer)).isEqualTo(config.sharedHiddenDim(0));
    }
  }

  @Test
  void aDenseGemma4HasNoExpertMetadataAndIsNotRejected() {
    // Gemma 4 12B and 31B publish no expert_* keys at all, because they are dense: their GGUFs
    // carry
    // zero tensors matching *_exps and a plain blk.N.ffn_{gate,up,down}. Requiring those keys
    // rejected every dense Gemma 4 with "Missing gemma4.expert_feed_forward_length" before it could
    // be loaded.
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>(entries());
    entries.remove("gemma4.expert_count");
    entries.remove("gemma4.expert_feed_forward_length");
    entries.remove("gemma4.expert_used_count");

    Gemma4Config config = Gemma4Config.fromMetadata(new GgufMetadata(entries));

    assertThat(config.isDense()).isTrue();
    assertThat(config.numExperts()).isZero();
    assertThat(config.numExpertsUsed()).isZero();
    assertThat(config.expertHiddenDim()).isZero();
    assertThat(config.sharedHiddenDim(0))
        .describedAs("the dense feed-forward width comes from feed_forward_length")
        .isPositive();
  }

  @Test
  void aMixtureOfExpertsGemma4IsStillDetectedAsSuch() {
    Gemma4Config config = Gemma4Config.fromMetadata(metadata());

    assertThat(config.isDense()).isFalse();
    assertThat(config.numExperts()).isPositive();
  }

  @Test
  void halfSpecifiedExpertMetadataIsRejectedRatherThanTreatedAsDense() {
    // Experts declared but no width, or a width with no experts, is a malformed model. Treating it
    // as dense would silently skip the routed half of every layer and produce wrong output.
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>(entries());
    entries.remove("gemma4.expert_feed_forward_length");

    assertThatThrownBy(() -> Gemma4Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class);
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
