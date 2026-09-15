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
    String architecture,
    EncoderStyle encoderStyle,
    int embeddingDim,
    int numLayers,
    int numHeads,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    float layerNormEps,
    float ropeTheta,
    Pooling pooling) {

  /** Encoder layouts that deliberately share the BERT whole-sequence execution contract. */
  public enum EncoderStyle {
    BERT,
    NOMIC_BERT
  }

  /** Sequence-reduction modes defined by GGUF's BERT pooling metadata. */
  public enum Pooling {
    MEAN(1),
    CLS(2),
    RANK(4);

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
    Objects.requireNonNull(architecture, "architecture");
    Objects.requireNonNull(encoderStyle, "encoderStyle");
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
    if (encoderStyle == EncoderStyle.NOMIC_BERT
        && (!(ropeTheta > 0.0f) || !Float.isFinite(ropeTheta))) {
      throw new IllegalArgumentException("ropeTheta must be finite and > 0: " + ropeTheta);
    }
  }

  /** Width of one attention head. */
  public int headDim() {
    return embeddingDim / numHeads;
  }

  /** Reads and validates the BERT or Nomic-BERT GGUF metadata contract. */
  public static BertConfig fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    EncoderStyle style =
        switch (architecture) {
          case "bert" -> EncoderStyle.BERT;
          case "nomic-bert" -> EncoderStyle.NOMIC_BERT;
          default -> null;
        };
    if (style == null) {
      throw new IllegalArgumentException(
          "BERT configuration requires architecture bert or nomic-bert: " + architecture);
    }
    String prefix = architecture + ".";
    if (metadata.getBool(prefix + "attention.causal").orElse(false)) {
      throw new IllegalArgumentException(
          "BERT encoder attention must be bidirectional, but the artifact declares causal attention");
    }
    return new BertConfig(
        architecture,
        style,
        requiredUint32(metadata, prefix + "embedding_length"),
        requiredUint32(metadata, prefix + "block_count"),
        requiredUint32(metadata, prefix + "attention.head_count"),
        metadata
            .getArraySize("tokenizer.ggml.tokens")
            .orElseThrow(
                () -> new IllegalArgumentException("missing tokenizer.ggml.tokens vocabulary")),
        requiredUint32(metadata, prefix + "context_length"),
        requiredUint32(metadata, prefix + "feed_forward_length"),
        metadata
            .getFloat32(prefix + "attention.layer_norm_epsilon")
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "missing " + prefix + "attention.layer_norm_epsilon")),
        style == EncoderStyle.NOMIC_BERT
            ? metadata
                .getFloat32(prefix + "rope.freq_base")
                .orElseThrow(
                    () -> new IllegalArgumentException("missing " + prefix + "rope.freq_base"))
            : 1.0f,
        Pooling.fromCode(requiredUint32(metadata, prefix + "pooling_type")));
  }

  public boolean usesRotaryPositions() {
    return encoderStyle == EncoderStyle.NOMIC_BERT;
  }

  public boolean usesGatedFeedForward() {
    return encoderStyle == EncoderStyle.NOMIC_BERT;
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
