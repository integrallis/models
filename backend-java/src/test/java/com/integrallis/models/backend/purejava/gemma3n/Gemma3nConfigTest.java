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
package com.integrallis.models.backend.purejava.gemma3n;

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
 * The Gemma 3n contract, keyed exactly as the published {@code gemma-3n-E2B-it} header is.
 *
 * <p>Every value in {@link #entries()} was read out of that file on 2026-09-29, including the two
 * that are easy to get wrong by assumption: {@code shared_kv_layers} is a <b>float</b>, and {@code
 * activation_sparsity_scale} is {@code -inf} -- not zero -- on the layers without sparsity.
 */
@Tag("unit")
class Gemma3nConfigTest {

  private static final int LAYERS = 30;

  private static final int VOCABULARY = 64;

  /** The published sparsity multiplier: the 95th-percentile z-score. */
  private static final float SPARSITY = 1.6448533535003662f;

  @Test
  void parsesThePublishedE2bHeader() {
    Gemma3nConfig config = Gemma3nConfig.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.architecture()).isEqualTo("gemma3n");
    assertThat(config.embeddingDim()).isEqualTo(2_048);
    assertThat(config.numLayers()).isEqualTo(LAYERS);
    assertThat(config.numHeads()).isEqualTo(8);
    assertThat(config.numKvHeads()).isEqualTo(2);
    assertThat(config.headDim()).isEqualTo(256);
    assertThat(config.contextLength()).isEqualTo(32_768);
    // No vocab_size key in the published header, so this comes from the token count.
    assertThat(config.vocabSize()).isEqualTo(VOCABULARY);
    assertThat(config.hiddenDim()).isEqualTo(8_192);
    assertThat(config.altupInputs()).isEqualTo(4);
    assertThat(config.altupActiveIndex()).isZero();
    assertThat(config.perLayerEmbeddingDim()).isEqualTo(256);
    assertThat(config.perLayerTotalDim()).isEqualTo(7_680);
    assertThat(config.slidingWindow()).isEqualTo(512);
    assertThat(config.queryDim()).isEqualTo(2_048);
    assertThat(config.keyDim()).isEqualTo(512);
  }

  /**
   * {@code shared_kv_layers} is published as a float.
   *
   * <p>Gemma 4 reads its own equivalent with {@code getUint32}. Doing that here finds nothing, and
   * an {@code orElse(0)} fallback would make every layer appear to own a cache -- which loads
   * cleanly and then reads keys and values nobody wrote.
   */
  @Test
  void theSharedKeyValueLayerCountIsReadFromAFloat() {
    Gemma3nConfig config = Gemma3nConfig.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.sharedKvLayers()).isEqualTo(10);
    assertThat(config.kvOwningLayers()).isEqualTo(20);
    assertThat(config.ownsKvCache(19)).isTrue();
    assertThat(config.ownsKvCache(20)).isFalse();
  }

  @Test
  void anIntegerSharedKeyValueLayerCountIsRefusedRatherThanSilentlyIgnored() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma3n.attention.shared_kv_layers", new GgufMetadataValue.Uint32Value(10));

    // Not a float, so the float read finds nothing. Refusing names the key; defaulting would not.
    assertThatThrownBy(() -> Gemma3nConfig.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("shared_kv_layers");
  }

  @Test
  void aFractionalSharedKeyValueLayerCountIsRefused() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma3n.attention.shared_kv_layers", new GgufMetadataValue.Float32Value(10.5f));

    assertThatThrownBy(() -> Gemma3nConfig.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("whole number");
  }

  /**
   * The layers without sparsity publish {@code -inf}, and that must mean "none" rather than a
   * cutoff.
   *
   * <p>{@code mean + deviation * -inf} is {@code -inf}, and {@code relu(x - -inf)} saturates every
   * activation to {@code +inf}. Treating the entry as a number to use produces an all-infinite
   * feed-forward, which is not subtly wrong.
   */
  @Test
  void aNonFiniteSparsityScaleMeansNoSparsityRatherThanAnInfiniteCutoff() {
    Gemma3nConfig config = Gemma3nConfig.fromMetadata(new GgufMetadata(entries()));

    for (int layer = 0; layer < 10; layer++) {
      assertThat(config.usesActivationSparsity(layer))
          .describedAs("layer %s carries the published multiplier", layer)
          .isTrue();
      assertThat(config.activationSparsityScale(layer)).isEqualTo(SPARSITY);
    }
    for (int layer = 10; layer < LAYERS; layer++) {
      assertThat(config.usesActivationSparsity(layer))
          .describedAs("layer %s publishes -inf, which is not a cutoff", layer)
          .isFalse();
    }
    // And asking for the scale where there is none is a mistake, not a silent infinity.
    assertThatThrownBy(() -> config.activationSparsityScale(29))
        .isInstanceOf(IllegalArgumentException.class);
  }

  /**
   * And a POSITIVE infinity is equally not a scale.
   *
   * <p>{@code -inf > 0} is already false, so a bare positivity test handles the published file by
   * accident. It does not handle {@code +inf}, which would give a cutoff of {@code +inf} and zero
   * every activation -- so the finiteness check earns its place here rather than in the published
   * case.
   */
  @Test
  void aPositiveInfinitySparsityScaleIsAlsoRejected() {
    Map<String, GgufMetadataValue> entries = entries();
    List<Float> scales = new ArrayList<>(sparsityPattern());
    scales.set(3, Float.POSITIVE_INFINITY);
    scales.set(4, Float.NaN);
    entries.put("gemma3n.activation_sparsity_scale", floatArray(scales));

    Gemma3nConfig config = Gemma3nConfig.fromMetadata(new GgufMetadata(entries));

    assertThat(config.usesActivationSparsity(3)).isFalse();
    assertThat(config.usesActivationSparsity(4)).isFalse();
    assertThat(config.usesActivationSparsity(2))
        .describedAs("the finite neighbours are unaffected")
        .isTrue();
  }

  /** The fixture must genuinely publish -inf, or the test above passes against a zero instead. */
  @Test
  void theFixtureCarriesTheNegativeInfinityThePublishedFileCarries() {
    Gemma3nConfig config = Gemma3nConfig.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.sparsityScaleByLayer().get(29)).isEqualTo(Float.NEGATIVE_INFINITY);
    assertThat(config.sparsityScaleByLayer().get(29)).isNotEqualTo(0.0f);
  }

  /**
   * The sliding pattern is taken from the file, and the shared-key-value mapping depends on it.
   *
   * <p>E2B's pattern is {@code SSSS.} repeated: every fifth layer is full attention. Layer 20 is
   * the first sharing layer and is sliding, so it reads {@code 20 - 2 = 18}; layer 24 is full
   * attention, so it reads {@code 20 - 1 = 19}.
   */
  @Test
  void aSharingLayerReadsACacheOfItsOwnSpan() {
    Gemma3nConfig config = Gemma3nConfig.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.usesSlidingWindow(4)).isFalse();
    assertThat(config.usesSlidingWindow(20)).isTrue();
    assertThat(config.usesSlidingWindow(24)).isFalse();

    assertThat(config.kvSourceLayer(20)).isEqualTo(18);
    assertThat(config.kvSourceLayer(24)).isEqualTo(19);
    assertThat(config.usesSlidingWindow(18)).isTrue();
    assertThat(config.usesSlidingWindow(19)).isFalse();
    // An owning layer is its own source.
    assertThat(config.kvSourceLayer(5)).isEqualTo(5);
  }

  /**
   * A sliding pattern that makes the -2/-1 mapping point at a cache of the wrong span is refused.
   *
   * <p>Layer 18 made full-attention while layer 20 stays sliding: layer 20 computes its source as
   * {@code 20 - 2 = 18}, whose span now differs, so the positions it would read were never written
   * for it. That is not something to serve with plausible output -- the assertion is a refusal.
   */
  @Test
  void aSharingLayerWhoseComputedSourceHasTheWrongSpanIsRefused() {
    Map<String, GgufMetadataValue> entries = entries();
    List<Boolean> pattern = new ArrayList<>(slidingPattern());
    pattern.set(18, false);
    assertThat(pattern.get(20)).describedAs("layer 20 must stay sliding").isTrue();
    entries.put("gemma3n.attention.sliding_window_pattern", boolArray(pattern));
    Gemma3nConfig config = Gemma3nConfig.fromMetadata(new GgufMetadata(entries));

    assertThatThrownBy(() -> config.kvSourceLayer(20))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("sliding-window span");
  }

  @Test
  void aWrongLengthPerLayerArrayIsRefused() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("gemma3n.activation_sparsity_scale", floatArray(List.of(1.0f, 2.0f)));

    assertThatThrownBy(() -> Gemma3nConfig.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sparsityScaleByLayer");
  }

  @Test
  void anotherArchitectureIsRefused() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma4"));

    assertThatThrownBy(() -> Gemma3nConfig.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("gemma3n");
  }

  /** The attention scale is 1.0 for this architecture, which no key states. */
  @Test
  void theAttentionScaleIsOne() {
    assertThat(Gemma3nConfig.ATTENTION_SCALE).isEqualTo(1.0f);
  }

  /** Exactly the keys the published E2B header carries, with its values. */
  private static Map<String, GgufMetadataValue> entries() {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma3n"));
    entries.put("gemma3n.block_count", new GgufMetadataValue.Uint32Value(LAYERS));
    entries.put("gemma3n.embedding_length", new GgufMetadataValue.Uint32Value(2_048));
    entries.put("gemma3n.attention.head_count", new GgufMetadataValue.Uint32Value(8));
    entries.put("gemma3n.attention.head_count_kv", new GgufMetadataValue.Uint32Value(2));
    entries.put("gemma3n.attention.key_length", new GgufMetadataValue.Uint32Value(256));
    entries.put("gemma3n.attention.value_length", new GgufMetadataValue.Uint32Value(256));
    entries.put("gemma3n.context_length", new GgufMetadataValue.Uint32Value(32_768));
    entries.put("gemma3n.feed_forward_length", new GgufMetadataValue.Uint32Value(8_192));
    entries.put("gemma3n.embedding_length_per_layer_input", new GgufMetadataValue.Uint32Value(256));
    entries.put("gemma3n.altup.num_inputs", new GgufMetadataValue.Uint32Value(4));
    entries.put("gemma3n.altup.active_idx", new GgufMetadataValue.Uint32Value(0));
    entries.put("gemma3n.attention.sliding_window", new GgufMetadataValue.Uint32Value(512));
    entries.put("gemma3n.rope.freq_base", new GgufMetadataValue.Float32Value(1_000_000.0f));
    entries.put(
        "gemma3n.attention.layer_norm_rms_epsilon", new GgufMetadataValue.Float32Value(1.0e-6f));
    // A float, as published.
    entries.put("gemma3n.attention.shared_kv_layers", new GgufMetadataValue.Float32Value(10.0f));
    entries.put("gemma3n.attention.sliding_window_pattern", boolArray(slidingPattern()));
    entries.put("gemma3n.activation_sparsity_scale", floatArray(sparsityPattern()));
    // The published file carries 262144 tokens and NO vocab_size key. A small array here
    // exercises the same fallback without materialising a quarter of a million entries.
    entries.put("tokenizer.ggml.tokens", stringArray(VOCABULARY));
    return entries;
  }

  /** {@code SSSS.} repeated: every fifth layer is full attention. */
  private static List<Boolean> slidingPattern() {
    List<Boolean> pattern = new ArrayList<>(LAYERS);
    for (int layer = 0; layer < LAYERS; layer++) {
      pattern.add((layer + 1) % 5 != 0);
    }
    return pattern;
  }

  /** The published multiplier on the first ten layers, negative infinity on the rest. */
  private static List<Float> sparsityPattern() {
    List<Float> scales = new ArrayList<>(LAYERS);
    for (int layer = 0; layer < LAYERS; layer++) {
      scales.add(layer < 10 ? SPARSITY : Float.NEGATIVE_INFINITY);
    }
    return scales;
  }

  private static GgufMetadataValue.ArrayValue boolArray(List<Boolean> values) {
    return new GgufMetadataValue.ArrayValue(
        GgufValueType.BOOL,
        values.stream()
            .map(GgufMetadataValue.BoolValue::new)
            .map(GgufMetadataValue.class::cast)
            .toList());
  }

  private static GgufMetadataValue.ArrayValue floatArray(List<Float> values) {
    return new GgufMetadataValue.ArrayValue(
        GgufValueType.FLOAT32,
        values.stream()
            .map(GgufMetadataValue.Float32Value::new)
            .map(GgufMetadataValue.class::cast)
            .toList());
  }

  private static GgufMetadataValue.ArrayValue stringArray(int size) {
    List<GgufMetadataValue> values = new ArrayList<>(size);
    for (int index = 0; index < size; index++) {
      values.add(new GgufMetadataValue.StringValue("t" + index));
    }
    return new GgufMetadataValue.ArrayValue(GgufValueType.STRING, values);
  }
}
