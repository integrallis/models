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
    void ignoresScalingFactorWhenScalingIsDisabled() {
      Map<String, GgufMetadataValue> entries = requiredLlamaEntries();
      entries.put("llama.rope.scaling.type", new GgufMetadataValue.StringValue("none"));
      entries.put("llama.rope.scaling.factor", new GgufMetadataValue.Float32Value(8.0f));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.ropeFrequencyScale()).isEqualTo(1.0f);
      assertThat(config.ropeScaling().type()).isEqualTo(LlamaConfig.RopeScalingType.LINEAR);
      assertThat(config.ropeScaling().factor()).isEqualTo(1.0f);
    }

    @Test
    void readsThePinnedQwenYarnRopeScalingContract() {
      Map<String, GgufMetadataValue> entries = requiredLlamaEntries();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("qwen2"));
      entries.put("qwen2.embedding_length", new GgufMetadataValue.Uint32Value(896));
      entries.put("qwen2.block_count", new GgufMetadataValue.Uint32Value(24));
      entries.put("qwen2.attention.head_count", new GgufMetadataValue.Uint32Value(14));
      entries.put("qwen2.attention.head_count_kv", new GgufMetadataValue.Uint32Value(2));
      entries.put("qwen2.context_length", new GgufMetadataValue.Uint32Value(32768));
      entries.put("qwen2.rope.freq_base", new GgufMetadataValue.Float32Value(1_000_000.0f));
      entries.put("qwen2.rope.scaling.type", new GgufMetadataValue.StringValue("yarn"));
      entries.put("qwen2.rope.scaling.factor", new GgufMetadataValue.Float32Value(4.0f));
      entries.put(
          "qwen2.rope.scaling.original_context_length", new GgufMetadataValue.Uint32Value(32768));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.ropeFrequencyScale()).isEqualTo(0.25f);
      assertThat(config.ropeScaling().type()).isEqualTo(LlamaConfig.RopeScalingType.YARN);
      assertThat(config.ropeScaling().factor()).isEqualTo(4.0f);
      assertThat(config.ropeScaling().originalContext()).isEqualTo(32768);
      assertThat(config.ropeScaling().betaFast()).isEqualTo(32.0f);
      assertThat(config.ropeScaling().betaSlow()).isEqualTo(1.0f);
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
    void extractsHunyuanDenseSemantics() {
      // Hunyuan-MT-7B as published: Qwen 3-shaped -- grouped-query attention with per-head Q/K
      // norms
      // of exactly head_dim, SiLU feed-forward, a very large rope base, and scaling explicitly
      // declared "none". Every key it carries is one LlamaConfig already reads.
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("hunyuan-dense"));
      entries.put("hunyuan-dense.embedding_length", new GgufMetadataValue.Uint32Value(4096));
      entries.put("hunyuan-dense.block_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("hunyuan-dense.attention.head_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("hunyuan-dense.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));
      entries.put("hunyuan-dense.attention.key_length", new GgufMetadataValue.Uint32Value(128));
      entries.put("hunyuan-dense.attention.value_length", new GgufMetadataValue.Uint32Value(128));
      entries.put("hunyuan-dense.feed_forward_length", new GgufMetadataValue.Uint32Value(14336));
      entries.put(
          "hunyuan-dense.rope.freq_base", new GgufMetadataValue.Float32Value(1.200508032e9f));
      entries.put("hunyuan-dense.rope.scaling.type", new GgufMetadataValue.StringValue("none"));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.architecture()).isEqualTo(DecoderArchitecture.HUNYUAN_DENSE);
      assertThat(config.numHeads()).isEqualTo(32);
      assertThat(config.numKvHeads()).isEqualTo(8);
      assertThat(config.keyLength()).isEqualTo(128);
      assertThat(config.ropeTheta(0)).isEqualTo(1.200508032e9f);
      // Not a Gemma: SiLU feed-forward, no post norms, no embedding scaling.
      assertThat(config.usesGeluFfn()).isFalse();
      assertThat(config.usesPostAttentionNorm()).isFalse();
      assertThat(config.usesPostFfnNorm()).isFalse();
      assertThat(config.embeddingScale()).isEqualTo(1.0f);
      assertThat(config.attnLogitSoftcap()).isZero();
      // Declared "none" scaling must not become a scaling factor.
      assertThat(config.ropeFrequencyScale()).isEqualTo(1.0f);
      // The one inference in this architecture, pinned so a later change is deliberate: the layout
      // is
      // NeoX on the strength of the Qwen 3 shape, pending the perplexity comparison described on
      // usesNeoxRope().
      assertThat(config.usesNeoxRope()).isTrue();
    }

    @Test
    void extractsGemma2SemanticsIncludingTheAttentionSoftcapAndAlternatingWindow() {
      // gemma-2-2b-it as published. Two things here are easy to get wrong and silent when wrong:
      // the attention-logit softcap (Gemma 2 only; Gemma 3 dropped it), and the sliding-window
      // pattern, which Gemma 2 does NOT publish -- it alternates every other layer, so the generic
      // default of 6 would make five layers in six local instead of one in two.
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma2"));
      entries.put("gemma2.embedding_length", new GgufMetadataValue.Uint32Value(2304));
      entries.put("gemma2.block_count", new GgufMetadataValue.Uint32Value(26));
      entries.put("gemma2.attention.head_count", new GgufMetadataValue.Uint32Value(8));
      entries.put("gemma2.attention.head_count_kv", new GgufMetadataValue.Uint32Value(4));
      entries.put("gemma2.attention.key_length", new GgufMetadataValue.Uint32Value(256));
      entries.put("gemma2.attention.value_length", new GgufMetadataValue.Uint32Value(256));
      entries.put("gemma2.attention.sliding_window", new GgufMetadataValue.Uint32Value(4096));
      entries.put("gemma2.attn_logit_softcapping", new GgufMetadataValue.Float32Value(50.0f));
      entries.put("gemma2.final_logit_softcapping", new GgufMetadataValue.Float32Value(30.0f));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.architecture()).isEqualTo(DecoderArchitecture.GEMMA2);
      assertThat(config.attnLogitSoftcap()).isEqualTo(50.0f);
      assertThat(config.finalLogitSoftcap()).isEqualTo(30.0f);
      // Gemma family traits:
      assertThat(config.usesGeluFfn()).isTrue();
      assertThat(config.usesNeoxRope()).isTrue();
      assertThat(config.embeddingScale()).isEqualTo((float) Math.sqrt(2304));
      // Gemma 2 HAS the post norms (unlike Gemma 1):
      assertThat(config.usesPostAttentionNorm()).isTrue();
      assertThat(config.usesPostFfnNorm()).isTrue();
      // Alternating local/global: even layers local, odd layers global.
      assertThat(config.slidingWindowPattern()).isEqualTo(2);
      assertThat(config.usesSlidingWindow(0)).isTrue();
      assertThat(config.usesSlidingWindow(1)).isFalse();
      assertThat(config.usesSlidingWindow(2)).isTrue();
      assertThat(config.usesSlidingWindow(3)).isFalse();
    }

    @Test
    void gemma3KeepsItsOwnWindowPatternAndHasNoAttentionSoftcap() {
      // Guards both halves of the gemma2 work against leaking into Gemma 3.
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma3"));
      entries.put("gemma3.embedding_length", new GgufMetadataValue.Uint32Value(1152));
      entries.put("gemma3.block_count", new GgufMetadataValue.Uint32Value(26));
      entries.put("gemma3.attention.head_count", new GgufMetadataValue.Uint32Value(4));
      entries.put("gemma3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(1));
      entries.put("gemma3.attention.sliding_window", new GgufMetadataValue.Uint32Value(512));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.attnLogitSoftcap()).isZero();
      assertThat(config.slidingWindowPattern()).isEqualTo(6);
      assertThat(config.usesSlidingWindow(4)).isTrue();
      assertThat(config.usesSlidingWindow(5)).isFalse();
    }

    @Test
    void extractsPhi3SemanticsForTheFullRotaryVariants() {
      // phi-3.5-mini and phi-3-mini-4k: 32 heads, head_dim 96, rope dim 96 -- full rotary.
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("phi3"));
      entries.put("phi3.embedding_length", new GgufMetadataValue.Uint32Value(3072));
      entries.put("phi3.block_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("phi3.attention.head_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("phi3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(32));
      entries.put("phi3.feed_forward_length", new GgufMetadataValue.Uint32Value(8192));
      entries.put("phi3.rope.dimension_count", new GgufMetadataValue.Uint32Value(96));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.architecture()).isEqualTo(DecoderArchitecture.PHI3);
      assertThat(config.usesNeoxRope()).isTrue();
      assertThat(config.usesGeluFfn()).isFalse();
      assertThat(config.usesPostAttentionNorm()).isFalse();
      assertThat(config.usesPostFfnNorm()).isFalse();
      assertThat(config.embeddingScale()).isEqualTo(1.0f);
    }

    @Test
    void honoursAPartialRotaryWidthInsteadOfRotatingTheWholeHead() {
      // Phi-4-mini rotates 96 of its 128 head dimensions. This used to be REFUSED, because the
      // rotary tables were built at the head width and would have rotated dimensions the model
      // leaves alone. They are now built at the declared width.
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("phi3"));
      entries.put("phi3.embedding_length", new GgufMetadataValue.Uint32Value(3072));
      entries.put("phi3.block_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("phi3.attention.head_count", new GgufMetadataValue.Uint32Value(24));
      entries.put("phi3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));
      entries.put("phi3.attention.key_length", new GgufMetadataValue.Uint32Value(128));
      entries.put("phi3.attention.value_length", new GgufMetadataValue.Uint32Value(128));
      entries.put("phi3.feed_forward_length", new GgufMetadataValue.Uint32Value(8192));
      entries.put("phi3.rope.dimension_count", new GgufMetadataValue.Uint32Value(96));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.ropeDimensions()).isEqualTo(96);
      assertThat(config.headDim()).isEqualTo(128);
      assertThat(config.usesPartialRotary()).isTrue();
    }

    @Test
    void aFullRotaryModelReportsTheHeadWidthAndNoPartialRotary() {
      // Absent means full rotary, and phi-3-mini is genuinely full (96 = 96), so the two must not
      // be
      // conflated: this is what keeps every currently-served model on the same arithmetic.
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("phi3"));
      entries.put("phi3.embedding_length", new GgufMetadataValue.Uint32Value(3072));
      entries.put("phi3.block_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("phi3.attention.head_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("phi3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(32));
      entries.put("phi3.attention.key_length", new GgufMetadataValue.Uint32Value(96));
      entries.put("phi3.attention.value_length", new GgufMetadataValue.Uint32Value(96));
      entries.put("phi3.feed_forward_length", new GgufMetadataValue.Uint32Value(8192));
      entries.put("phi3.rope.dimension_count", new GgufMetadataValue.Uint32Value(96));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.ropeDimensions()).isEqualTo(96);
      assertThat(config.usesPartialRotary()).isFalse();

      // And with the key absent it defaults to the head width rather than to anything derived.
      Map<String, GgufMetadataValue> absent = new LinkedHashMap<>(entries);
      absent.remove("phi3.rope.dimension_count");
      assertThat(LlamaConfig.fromMetadata(new GgufMetadata(absent)).ropeDimensions()).isEqualTo(96);
    }

    @Test
    void aRotaryWidthWiderThanTheHeadOrOddIsRefused() {
      Map<String, GgufMetadataValue> base = new LinkedHashMap<>();
      base.put("general.architecture", new GgufMetadataValue.StringValue("phi3"));
      base.put("phi3.embedding_length", new GgufMetadataValue.Uint32Value(3072));
      base.put("phi3.block_count", new GgufMetadataValue.Uint32Value(32));
      base.put("phi3.attention.head_count", new GgufMetadataValue.Uint32Value(24));
      base.put("phi3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));
      base.put("phi3.attention.key_length", new GgufMetadataValue.Uint32Value(128));
      base.put("phi3.attention.value_length", new GgufMetadataValue.Uint32Value(128));
      base.put("phi3.feed_forward_length", new GgufMetadataValue.Uint32Value(8192));

      for (int width : new int[] {130, 97, 0}) {
        Map<String, GgufMetadataValue> entries = new LinkedHashMap<>(base);
        entries.put("phi3.rope.dimension_count", new GgufMetadataValue.Uint32Value(width));
        assertThatThrownBy(() -> LlamaConfig.fromMetadata(new GgufMetadata(entries)))
            .describedAs("rope width %s", width)
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("rope.dimension_count");
      }
    }

    @Test
    void extractsGemma1SemanticsWithoutTheGemma2Norms() {
      // codegemma-7b-it as published: exactly the Llama tensor set, and the only keys it carries
      // are
      // ones LlamaConfig already reads. What it must NOT do is inherit the post-attention and
      // post-feed-forward norms, which arrived with Gemma 2 -- requiring them would reject every
      // Gemma 1 model for tensors it was never built with.
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma"));
      entries.put("gemma.embedding_length", new GgufMetadataValue.Uint32Value(3072));
      entries.put("gemma.block_count", new GgufMetadataValue.Uint32Value(28));
      entries.put("gemma.attention.head_count", new GgufMetadataValue.Uint32Value(16));
      entries.put("gemma.attention.head_count_kv", new GgufMetadataValue.Uint32Value(16));
      entries.put("gemma.feed_forward_length", new GgufMetadataValue.Uint32Value(24576));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.architecture()).isEqualTo(DecoderArchitecture.GEMMA);
      // Shared with the rest of the family:
      assertThat(config.embeddingScale()).isEqualTo((float) Math.sqrt(3072));
      assertThat(config.usesGeluFfn()).isTrue();
      assertThat(config.usesNeoxRope()).isTrue();
      // NOT shared -- these are Gemma 2+ only:
      assertThat(config.usesPostAttentionNorm()).isFalse();
      assertThat(config.usesPostFfnNorm()).isFalse();
      // No sliding_window key at all, so every layer is full attention.
      assertThat(config.slidingWindow()).isZero();
      assertThat(config.usesSlidingWindow(0)).isFalse();
      assertThat(config.usesSlidingWindow(5)).isFalse();
      // Multi-head, not grouped: 16/16.
      assertThat(config.numHeads()).isEqualTo(16);
      assertThat(config.numKvHeads()).isEqualTo(16);
      assertThat(config.usesGraniteScaling()).isFalse();
      assertThat(config.finalLogitSoftcap()).isZero();
    }

    @Test
    void gemma3StillRequiresItsPostNormsAfterGemma1WasAdded() {
      // Guards the split: adding Gemma 1 must not turn the Gemma 2+ norms off for Gemma 3.
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma3"));
      entries.put("gemma3.embedding_length", new GgufMetadataValue.Uint32Value(1152));
      entries.put("gemma3.block_count", new GgufMetadataValue.Uint32Value(26));
      entries.put("gemma3.attention.head_count", new GgufMetadataValue.Uint32Value(4));
      entries.put("gemma3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(1));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.usesPostAttentionNorm()).isTrue();
      assertThat(config.usesPostFfnNorm()).isTrue();
    }

    @Test
    void extractsGemma3DecoderSemantics() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("gemma3"));
      entries.put("gemma3.embedding_length", new GgufMetadataValue.Uint32Value(1152));
      entries.put("gemma3.block_count", new GgufMetadataValue.Uint32Value(26));
      entries.put("gemma3.attention.head_count", new GgufMetadataValue.Uint32Value(4));
      entries.put("gemma3.attention.head_count_kv", new GgufMetadataValue.Uint32Value(1));
      entries.put("gemma3.attention.key_length", new GgufMetadataValue.Uint32Value(256));
      entries.put("gemma3.attention.value_length", new GgufMetadataValue.Uint32Value(256));
      entries.put("gemma3.attention.sliding_window", new GgufMetadataValue.Uint32Value(512));
      entries.put("gemma3.rope.freq_base", new GgufMetadataValue.Float32Value(1_000_000.0f));
      entries.put("gemma3.final_logit_softcapping", new GgufMetadataValue.Float32Value(30.0f));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.architecture()).isEqualTo(DecoderArchitecture.GEMMA3);
      assertThat(config.embeddingScale()).isEqualTo((float) Math.sqrt(1152));
      assertThat(config.usesGeluFfn()).isTrue();
      assertThat(config.usesPostAttentionNorm()).isTrue();
      assertThat(config.usesPostFfnNorm()).isTrue();
      assertThat(config.usesNeoxRope()).isTrue();
      assertThat(config.slidingWindow()).isEqualTo(512);
      assertThat(config.usesSlidingWindow(0)).isTrue();
      assertThat(config.usesSlidingWindow(4)).isTrue();
      assertThat(config.usesSlidingWindow(5)).isFalse();
      assertThat(config.ropeTheta(0)).isEqualTo(10_000.0f);
      assertThat(config.ropeTheta(5)).isEqualTo(1_000_000.0f);
      assertThat(config.attentionStartPosition(0, 600)).isEqualTo(89);
      assertThat(config.attentionStartPosition(5, 600)).isZero();
      assertThat(config.finalLogitSoftcap()).isEqualTo(30.0f);
    }

    @Test
    void extractsGraniteScalingSemanticsWithoutTreatingItAsLlama() {
      Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
      entries.put("general.architecture", new GgufMetadataValue.StringValue("granite"));
      entries.put("granite.embedding_length", new GgufMetadataValue.Uint32Value(4096));
      entries.put("granite.block_count", new GgufMetadataValue.Uint32Value(40));
      entries.put("granite.attention.head_count", new GgufMetadataValue.Uint32Value(32));
      entries.put("granite.attention.head_count_kv", new GgufMetadataValue.Uint32Value(8));
      entries.put("granite.feed_forward_length", new GgufMetadataValue.Uint32Value(12800));
      entries.put("granite.rope.freq_base", new GgufMetadataValue.Float32Value(10_000_000.0f));
      entries.put("granite.embedding_scale", new GgufMetadataValue.Float32Value(12.0f));
      entries.put("granite.attention.scale", new GgufMetadataValue.Float32Value(0.0078125f));
      entries.put("granite.residual_scale", new GgufMetadataValue.Float32Value(0.22f));
      entries.put("granite.logit_scale", new GgufMetadataValue.Float32Value(16.0f));

      LlamaConfig config = LlamaConfig.fromMetadata(new GgufMetadata(entries));

      assertThat(config.architecture()).isEqualTo(DecoderArchitecture.GRANITE);
      assertThat(config.embeddingScale()).isEqualTo(12.0f);
      assertThat(config.attentionScale()).isEqualTo(0.0078125f);
      assertThat(config.residualScale()).isEqualTo(0.22f);
      assertThat(config.logitScale()).isEqualTo(16.0f);
      assertThat(config.usesNeoxRope()).isFalse();
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
