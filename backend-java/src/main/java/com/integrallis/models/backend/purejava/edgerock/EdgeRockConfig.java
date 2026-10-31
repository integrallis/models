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
package com.integrallis.models.backend.purejava.edgerock;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.List;
import java.util.Objects;

/**
 * EdgeRock's shape, read from {@code edgerock.*} metadata.
 *
 * <p>The backbone half of this duplicates what {@code Qwen35Config} already parses, deliberately.
 * EdgeRock is the Qwen3.5 gated-delta-net backbone -- its reference implementation subclasses
 * {@code Qwen3_5GatedDeltaNet} -- and admitting {@code edgerock} to that class's architecture list
 * would be a two-character change. It is not made, because {@code Qwen35ForwardPass} is a 2600-line
 * final class on the hot path of three qualified architectures with no per-layer seam, so a shared
 * config would invite dispatching EdgeRock through a forward pass that silently ignores its memory
 * layers. A separate config keeps the two apart until an EdgeRock forward pass exists.
 *
 * @param blockCount total layers; 32 for EdgeRock-4B
 * @param fullAttentionInterval every nth layer is full attention, the rest recurrent
 * @param attentionWindow sliding-window span on the attention layers
 * @param memoryCapacity memory slots held per memory layer
 * @param memoryCommitChunk tokens staged before the bank is recomputed
 * @param memoryAdmissionRank bottleneck width of the admission scorer
 * @param memoryWriterIntermediate hidden width of the residual SwiGLU writer
 * @param ropeDimensionSections mRoPE section widths; {@code [11, 11, 10, 0]} on EdgeRock-4B, empty
 *     on a text-only export. Held as a list because a record that exposes an array hands callers a
 *     mutable view of its own state.
 */
public record EdgeRockConfig(
    int blockCount,
    int embeddingLength,
    int feedForwardLength,
    int headCount,
    int headCountKv,
    int headDim,
    int vocabSize,
    int contextLength,
    float rmsNormEpsilon,
    float ropeFreqBase,
    int ropeDimensionCount,
    List<Integer> ropeDimensionSections,
    int ssmConvKernel,
    int ssmStateSize,
    int ssmGroupCount,
    int ssmTimeStepRank,
    int ssmInnerSize,
    int fullAttentionInterval,
    int attentionWindow,
    int memoryCapacity,
    int memoryCommitChunk,
    int memoryAdmissionRank,
    int memoryWriterIntermediate) {

  public static final String ARCHITECTURE = "edgerock";

  public EdgeRockConfig {
    positive("blockCount", blockCount);
    positive("embeddingLength", embeddingLength);
    positive("headCount", headCount);
    positive("headCountKv", headCountKv);
    positive("headDim", headDim);
    positive("fullAttentionInterval", fullAttentionInterval);
    positive("attentionWindow", attentionWindow);
    positive("memoryCapacity", memoryCapacity);
    positive("memoryCommitChunk", memoryCommitChunk);
    positive("memoryAdmissionRank", memoryAdmissionRank);
    positive("memoryWriterIntermediate", memoryWriterIntermediate);
    ropeDimensionSections =
        List.copyOf(Objects.requireNonNull(ropeDimensionSections, "ropeDimensionSections"));
    if (headCount % headCountKv != 0) {
      throw new IllegalArgumentException(
          "headCount " + headCount + " must be a multiple of headCountKv " + headCountKv);
    }
    // The bank is recomputed from (committed ++ pending), so a chunk wider than the bank would
    // stage more candidates than can ever be kept.
    if (memoryCommitChunk > memoryCapacity) {
      throw new IllegalArgumentException(
          "memoryCommitChunk " + memoryCommitChunk + " exceeds memoryCapacity " + memoryCapacity);
    }
  }

  public static EdgeRockConfig fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!ARCHITECTURE.equals(architecture)) {
      throw new IllegalArgumentException(
          "Expected general.architecture " + ARCHITECTURE + ", found " + architecture);
    }
    String p = ARCHITECTURE + ".";
    return new EdgeRockConfig(
        requiredInt(metadata, p + "block_count"),
        requiredInt(metadata, p + "embedding_length"),
        requiredInt(metadata, p + "feed_forward_length"),
        requiredInt(metadata, p + "attention.head_count"),
        requiredInt(metadata, p + "attention.head_count_kv"),
        requiredInt(metadata, p + "attention.key_length"),
        metadata
            .getUint32(p + "vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing EdgeRock vocabulary size")),
        requiredInt(metadata, p + "context_length"),
        requiredFloat(metadata, p + "attention.layer_norm_rms_epsilon"),
        requiredFloat(metadata, p + "rope.freq_base"),
        requiredInt(metadata, p + "rope.dimension_count"),
        // Absent on a text-only export; an empty section list means plain RoPE.
        metadata.getInt32Array(p + "rope.dimension_sections").orElse(List.of()),
        requiredInt(metadata, p + "ssm.conv_kernel"),
        requiredInt(metadata, p + "ssm.state_size"),
        requiredInt(metadata, p + "ssm.group_count"),
        requiredInt(metadata, p + "ssm.time_step_rank"),
        requiredInt(metadata, p + "ssm.inner_size"),
        requiredInt(metadata, p + "full_attention_interval"),
        requiredInt(metadata, p + "attention.window"),
        requiredInt(metadata, p + "memory.capacity"),
        requiredInt(metadata, p + "memory.commit_chunk"),
        requiredInt(metadata, p + "memory.admission_rank"),
        requiredInt(metadata, p + "memory.writer_intermediate"));
  }

  /**
   * The layers carrying full attention, and therefore the memory module.
   *
   * <p>Derived from {@code full_attention_interval} rather than read from {@code
   * attention.recurrent_layers}. The two agree on EdgeRock-4B and {@code EdgeRockScopeTest} checks
   * that they do, because a disagreement would place attention on the wrong layers, load without
   * complaint, and produce wrong output.
   */
  public int[] fullAttentionLayers() {
    int count = blockCount / fullAttentionInterval;
    int[] layers = new int[count];
    for (int i = 0; i < count; i++) {
      layers[i] = (i + 1) * fullAttentionInterval - 1;
    }
    return layers;
  }

  /** Whether this layer is a full-attention layer, and so carries a memory bank. */
  public boolean isFullAttentionLayer(int layer) {
    return (layer + 1) % fullAttentionInterval == 0;
  }

  /** Query heads per key/value group, the GQA fan-out the memory read also uses. */
  public int headsPerGroup() {
    return headCount / headCountKv;
  }

  private static int requiredInt(GgufMetadata metadata, String key) {
    return metadata
        .getUint32(key)
        .orElseThrow(() -> new IllegalArgumentException("Missing EdgeRock metadata " + key));
  }

  private static float requiredFloat(GgufMetadata metadata, String key) {
    return metadata
        .getFloat32(key)
        .orElseThrow(() -> new IllegalArgumentException("Missing EdgeRock metadata " + key));
  }

  private static void positive(String name, int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive: " + value);
    }
  }
}
