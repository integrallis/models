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

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.Objects;

/** Structural contract for a bidirectional BERT GGUF encoder. */
public record BertConfig(
    int embeddingDim,
    int numLayers,
    int numHeads,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    float layerNormEps,
    Pooling pooling) {

  /** Sequence-reduction modes defined by GGUF's BERT pooling metadata. */
  public enum Pooling {
    MEAN(1),
    CLS(2);

    private final int code;

    Pooling(int code) {
      this.code = code;
    }

    static Pooling fromCode(int code) {
      for (Pooling value : values()) {
        if (value.code == code) {
          return value;
        }
      }
      throw new IllegalArgumentException("unsupported BERT pooling type: " + code);
    }
  }

  public BertConfig {
    requirePositive(embeddingDim, "embeddingDim");
    requirePositive(numLayers, "numLayers");
    requirePositive(numHeads, "numHeads");
    requirePositive(vocabSize, "vocabSize");
    requirePositive(contextLength, "contextLength");
    requirePositive(hiddenDim, "hiddenDim");
    if (embeddingDim % numHeads != 0) {
      throw new IllegalArgumentException(
          "embeddingDim must be divisible by numHeads: " + embeddingDim + " % " + numHeads);
    }
    if (!(layerNormEps > 0.0f) || !Float.isFinite(layerNormEps)) {
      throw new IllegalArgumentException("layerNormEps must be finite and > 0: " + layerNormEps);
    }
    Objects.requireNonNull(pooling, "pooling");
  }

  /** Width of one attention head. */
  public int headDim() {
    return embeddingDim / numHeads;
  }

  /** Reads and validates the standard BERT GGUF metadata contract. */
  public static BertConfig fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!"bert".equals(architecture)) {
      throw new IllegalArgumentException(
          "BERT configuration requires architecture bert: " + architecture);
    }
    if (metadata.getBool("bert.attention.causal").orElse(false)) {
      throw new IllegalArgumentException(
          "BERT encoder attention must be bidirectional, but the artifact declares causal attention");
    }
    return new BertConfig(
        requiredUint32(metadata, "bert.embedding_length"),
        requiredUint32(metadata, "bert.block_count"),
        requiredUint32(metadata, "bert.attention.head_count"),
        metadata
            .getArraySize("tokenizer.ggml.tokens")
            .orElseThrow(
                () -> new IllegalArgumentException("missing tokenizer.ggml.tokens vocabulary")),
        requiredUint32(metadata, "bert.context_length"),
        requiredUint32(metadata, "bert.feed_forward_length"),
        metadata
            .getFloat32("bert.attention.layer_norm_epsilon")
            .orElseThrow(
                () -> new IllegalArgumentException("missing bert.attention.layer_norm_epsilon")),
        Pooling.fromCode(requiredUint32(metadata, "bert.pooling_type")));
  }

  private static int requiredUint32(GgufMetadata metadata, String key) {
    return metadata
        .getUint32(key)
        .orElseThrow(() -> new IllegalArgumentException("missing " + key));
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0: " + value);
    }
  }
}
