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
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * EdgeRock read as the fourth name of the Qwen3.5 hybrid contract.
 *
 * <p>Every value below is transcribed from the published header of {@code
 * hkust-aaron/EdgeRock-4B-GGUF} / {@code EdgeRock-4B-Q4_K_M.gguf}, read on 2026-10-30 from a
 * range-fetched prefix with {@code experiments/edgerock-decoder/tools/gguf_describe.py}. They are
 * not representative numbers: a fixture that agrees with itself rather than with the artifact is
 * the thing these tests exist to avoid.
 */
@Tag("unit")
class Qwen35ConfigEdgeRockTest {

  @Test
  void parsesThePublishedEdgeRockFourBillionShape() {
    Qwen35Config config = Qwen35Config.fromMetadata(new GgufMetadata(entries()));

    assertThat(config.architecture()).isEqualTo("edgerock");
    assertThat(config.embeddingDim()).isEqualTo(2_560);
    assertThat(config.numLayers()).isEqualTo(32);
    assertThat(config.numHeads()).isEqualTo(16);
    assertThat(config.numKvHeads()).isEqualTo(4);
    assertThat(config.attentionHeadDim()).isEqualTo(256);
    assertThat(config.hiddenDim()).isEqualTo(9_216);
    assertThat(config.ropeDimension()).isEqualTo(64);
    assertThat(config.gdnKeyHeads()).isEqualTo(16);
    assertThat(config.gdnValueHeads()).isEqualTo(32);
    assertThat(config.gdnHeadDim()).isEqualTo(128);
    assertThat(config.gdnInnerDim()).isEqualTo(4_096);
    // 2 * 16 * 128 + 4096: the width of the attn_qkv in-projection, which the file stores as
    // [2560, 8192].
    assertThat(config.gdnConvDim()).isEqualTo(8_192);
    assertThat(config.usesMixtureOfExperts()).isFalse();
  }

  @Test
  void theEightMemoryLayersAreTheOnesTheFileCarriesMemoryTensorsFor() {
    Qwen35Config config = Qwen35Config.fromMetadata(new GgufMetadata(entries()));

    // The header's own full_attention_interval of 4 over 32 blocks, and independently the exact
    // layer set that carries blk.N.memory.* tensors and attn_q/attn_k/attn_v in the published file.
    assertThat(config.fullAttentionLayers()).containsExactly(3, 7, 11, 15, 19, 23, 27, 31);
    assertThat(config.usesMemoryAnchors()).isTrue();
  }

  @Test
  void theWindowIsReadFromTheHeaderAndOnlyEdgeRockPublishesOne() {
    assertThat(Qwen35Config.fromMetadata(new GgufMetadata(entries())).attentionWindow())
        .isEqualTo(1_024);
  }

  @Test
  void aModelThatPublishesNoWindowAttendsToEverything() {
    // 0 is "unwindowed", which is what the three older architectures mean by publishing no key at
    // all. Reading absence as a zero-width window would make every one of them attend to nothing.
    Map<String, GgufMetadataValue> entries = entries();
    entries.remove("edgerock.attention.window");

    assertThat(Qwen35Config.fromMetadata(new GgufMetadata(entries)).attentionWindow()).isZero();
  }

  @Test
  void aNegativeWindowIsRefused() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("edgerock.attention.window", new GgufMetadataValue.Int32Value(-1));

    assertThatThrownBy(() -> Qwen35Config.fromMetadata(new GgufMetadata(entries)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void theVocabularyComesFromTheTokenizerBecauseTheFilePublishesNoSize() {
    // The real header carries neither edgerock.vocab_size nor qwen35.vocab_size, so the tokenizer
    // array is the only source. Asserting it here keeps the fallback a tested path rather than one
    // a fixture quietly routes around by inventing the key.
    Map<String, GgufMetadataValue> entries = entries();
    assertThat(entries).doesNotContainKey("edgerock.vocab_size");

    assertThat(Qwen35Config.fromMetadata(new GgufMetadata(entries)).vocabSize()).isEqualTo(248_320);
  }

  @Test
  void theThreeOlderArchitecturesDoNotGainMemoryAnchors() {
    Map<String, GgufMetadataValue> entries = entries();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("qwen35"));
    // Re-key the hybrid half under the prefix qwen35 reads, which the real EdgeRock file also
    // publishes in full.
    for (Map.Entry<String, GgufMetadataValue> entry : new LinkedHashMap<>(entries).entrySet()) {
      if (entry.getKey().startsWith("edgerock.")) {
        entries.put("qwen35." + entry.getKey().substring("edgerock.".length()), entry.getValue());
      }
    }
    entries.put("qwen35.vocab_size", new GgufMetadataValue.Uint32Value(248_320));

    Qwen35Config config = Qwen35Config.fromMetadata(new GgufMetadata(entries));

    assertThat(config.usesMemoryAnchors()).isFalse();
    // And the dual-keyed file genuinely describes the same decoder under either prefix.
    assertThat(config.numLayers()).isEqualTo(32);
    assertThat(config.embeddingDim()).isEqualTo(2_560);
  }

  /** The published EdgeRock-4B header, transcribed. */
  private static Map<String, GgufMetadataValue> entries() {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("edgerock"));
    entries.put("edgerock.block_count", uint(32));
    entries.put("edgerock.context_length", uint(262_144));
    entries.put("edgerock.embedding_length", uint(2_560));
    entries.put("edgerock.feed_forward_length", uint(9_216));
    entries.put("edgerock.attention.head_count", uint(16));
    entries.put("edgerock.attention.head_count_kv", uint(4));
    entries.put("edgerock.attention.key_length", uint(256));
    entries.put("edgerock.attention.value_length", uint(256));
    entries.put("edgerock.attention.layer_norm_rms_epsilon", f32(9.999999974752427e-07f));
    entries.put("edgerock.rope.freq_base", f32(10_000_000.0f));
    entries.put("edgerock.rope.dimension_count", uint(64));
    entries.put("edgerock.ssm.conv_kernel", uint(4));
    entries.put("edgerock.ssm.state_size", uint(128));
    entries.put("edgerock.ssm.group_count", uint(16));
    entries.put("edgerock.ssm.time_step_rank", uint(32));
    entries.put("edgerock.ssm.inner_size", uint(4_096));
    entries.put("edgerock.full_attention_interval", uint(4));
    entries.put("edgerock.attention.window", uint(1_024));
    entries.put("edgerock.memory.capacity", uint(1_024));
    entries.put("edgerock.memory.commit_chunk", uint(256));
    entries.put("edgerock.memory.admission_rank", uint(256));
    entries.put("edgerock.memory.writer_intermediate", uint(2_560));
    entries.put("tokenizer.ggml.tokens", tokens(248_320));
    return entries;
  }

  private static GgufMetadataValue tokens(int count) {
    List<GgufMetadataValue> elements =
        new ArrayList<>(Collections.nCopies(count, new GgufMetadataValue.StringValue("t")));
    return new GgufMetadataValue.ArrayValue(GgufValueType.STRING, elements);
  }

  private static GgufMetadataValue.Uint32Value uint(int value) {
    return new GgufMetadataValue.Uint32Value(value);
  }

  private static GgufMetadataValue.Float32Value f32(float value) {
    return new GgufMetadataValue.Float32Value(value);
  }
}
