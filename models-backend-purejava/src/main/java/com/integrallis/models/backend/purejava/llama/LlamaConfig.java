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

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;

/** Configuration for a Llama-family model, extracted from GGUF metadata. */
public record LlamaConfig(
    int embeddingDim,
    int numLayers,
    int numHeads,
    int numKvHeads,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    float ropeTheta,
    float rmsNormEps) {

  public LlamaConfig {
    if (embeddingDim <= 0) throw new IllegalArgumentException("embeddingDim must be > 0");
    if (numLayers <= 0) throw new IllegalArgumentException("numLayers must be > 0");
    if (numHeads <= 0) throw new IllegalArgumentException("numHeads must be > 0");
    if (numKvHeads <= 0) throw new IllegalArgumentException("numKvHeads must be > 0");
    if (vocabSize <= 0) throw new IllegalArgumentException("vocabSize must be > 0");
  }

  /** Head dimension = embedding_dim / num_heads. */
  public int headDim() {
    return embeddingDim / numHeads;
  }

  /** KV dimension = head_dim * num_kv_heads. */
  public int kvDim() {
    return headDim() * numKvHeads;
  }

  /**
   * Extracts a LlamaConfig from GGUF metadata. Supports Llama-family architectures including models
   * that declare themselves as "qwen2", "qwen3", or "llama" — they all use the same structural
   * layout.
   */
  public static LlamaConfig fromMetadata(GgufMetadata metadata) {
    // Determine the architecture prefix from general.architecture (e.g., "llama", "qwen2",
    // "qwen3")
    String arch = metadata.getString("general.architecture").orElse("llama");

    int embeddingDim =
        getArchKey(metadata, arch, "embedding_length")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing " + arch + ".embedding_length"));
    int numLayers =
        getArchKey(metadata, arch, "block_count")
            .orElseThrow(() -> new IllegalArgumentException("Missing " + arch + ".block_count"));
    int numHeads =
        getArchKey(metadata, arch, "attention.head_count")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing " + arch + ".attention.head_count"));
    int numKvHeads =
        getArchKey(metadata, arch, "attention.head_count_kv")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing " + arch + ".attention.head_count_kv"));
    int vocabSize = getArchKey(metadata, arch, "vocab_size").orElse(32000);
    int contextLength = getArchKey(metadata, arch, "context_length").orElse(2048);
    int hiddenDim = getArchKey(metadata, arch, "feed_forward_length").orElse(embeddingDim * 4);
    float ropeTheta = getArchFloatKey(metadata, arch, "rope.freq_base").orElse(10000.0f);
    float rmsNormEps =
        getArchFloatKey(metadata, arch, "attention.layer_norm_rms_epsilon").orElse(1e-5f);

    return new LlamaConfig(
        embeddingDim,
        numLayers,
        numHeads,
        numKvHeads,
        vocabSize,
        contextLength,
        hiddenDim,
        ropeTheta,
        rmsNormEps);
  }

  /**
   * Looks up an integer metadata key with architecture prefix, falling back to "llama." prefix if
   * the arch-specific key is not found.
   */
  private static java.util.Optional<Integer> getArchKey(
      GgufMetadata metadata, String arch, String key) {
    return metadata.getUint32(arch + "." + key).or(() -> metadata.getUint32("llama." + key));
  }

  /**
   * Looks up a float metadata key with architecture prefix, falling back to "llama." prefix if the
   * arch-specific key is not found.
   */
  private static java.util.Optional<Float> getArchFloatKey(
      GgufMetadata metadata, String arch, String key) {
    return metadata.getFloat32(arch + "." + key).or(() -> metadata.getFloat32("llama." + key));
  }
}
