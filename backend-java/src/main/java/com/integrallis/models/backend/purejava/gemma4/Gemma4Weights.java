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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;

/** Resident tensors and streamed-expert layout for a text-only Gemma 4 decoder. */
final class Gemma4Weights {

  private static final Set<GgufTensorType> MATRIX_TYPES =
      Set.of(
          GgufTensorType.F32,
          GgufTensorType.Q4_0,
          GgufTensorType.Q5_0,
          GgufTensorType.Q8_0,
          GgufTensorType.Q4_K,
          GgufTensorType.Q5_K,
          GgufTensorType.Q6_K);

  record Matrix(MemorySegment data, GgufTensorType type, int rows, int columns) {
    Matrix {
      Objects.requireNonNull(data, "data");
      Objects.requireNonNull(type, "type");
      if (rows <= 0 || columns <= 0) {
        throw new IllegalArgumentException("matrix dimensions must be positive");
      }
    }
  }

  record LayerWeights(
      float[] attentionNorm,
      Matrix queryProjection,
      Matrix keyProjection,
      Matrix valueProjection,
      Matrix attentionOutputProjection,
      float[] queryNorm,
      float[] keyNorm,
      float[] attentionPostNorm,
      float[] sharedFfnNorm,
      Matrix sharedGateProjection,
      Matrix sharedUpProjection,
      Matrix sharedDownProjection,
      float[] routedFfnNorm,
      float[] sharedFfnPostNorm,
      float[] routedFfnPostNorm,
      float[] combinedFfnPostNorm,
      float[] routerScale,
      Matrix routerProjection,
      float[] expertScales,
      float layerOutputScale) {}

  private final MemorySegment tokenEmbedding;
  private final GgufTensorType tokenEmbeddingType;
  private final int embeddingDim;
  private final int vocabSize;
  private final float[] outputNorm;
  private final float[] ropeFrequencyFactors;
  private final LayerWeights[] layers;
  private final Gemma4TensorLayout expertLayout;

  private Gemma4Weights(
      MemorySegment tokenEmbedding,
      GgufTensorType tokenEmbeddingType,
      int embeddingDim,
      int vocabSize,
      float[] outputNorm,
      float[] ropeFrequencyFactors,
      LayerWeights[] layers,
      Gemma4TensorLayout expertLayout) {
    this.tokenEmbedding = tokenEmbedding;
    this.tokenEmbeddingType = tokenEmbeddingType;
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
    this.outputNorm = outputNorm;
    this.ropeFrequencyFactors = ropeFrequencyFactors;
    this.layers = layers;
    this.expertLayout = expertLayout;
  }

  static Gemma4Weights fromGgufFile(GgufFile file, Gemma4Config config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    GgufTensorData tokenEmbedding =
        matrixTensor(file, "token_embd.weight", config.vocabSize(), config.embeddingDim());
    float[] outputNorm = vector(file, "output_norm.weight", config.embeddingDim());
    float[] ropeFrequencyFactors =
        vector(file, "rope_freqs.weight", config.fullRopeDimension() / 2);

    LayerWeights[] layers = new LayerWeights[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "blk." + layer + ".";
      Matrix valueProjection =
          optionalMatrix(
              file, prefix + "attn_v.weight", config.valueDim(layer), config.embeddingDim());
      if (config.usesSlidingWindow(layer) && valueProjection == null) {
        throw new IllegalArgumentException(
            "Tensor not found: " + prefix + "attn_v.weight for sliding attention");
      }
      layers[layer] =
          new LayerWeights(
              vector(file, prefix + "attn_norm.weight", config.embeddingDim()),
              matrix(file, prefix + "attn_q.weight", config.queryDim(layer), config.embeddingDim()),
              matrix(file, prefix + "attn_k.weight", config.keyDim(layer), config.embeddingDim()),
              valueProjection,
              matrix(
                  file,
                  prefix + "attn_output.weight",
                  config.embeddingDim(),
                  config.attentionOutputDim(layer)),
              vector(file, prefix + "attn_q_norm.weight", config.headDim(layer)),
              vector(file, prefix + "attn_k_norm.weight", config.headDim(layer)),
              vector(file, prefix + "post_attention_norm.weight", config.embeddingDim()),
              vector(file, prefix + "ffn_norm.weight", config.embeddingDim()),
              matrix(
                  file,
                  prefix + "ffn_gate.weight",
                  config.sharedHiddenDim(),
                  config.embeddingDim()),
              matrix(
                  file, prefix + "ffn_up.weight", config.sharedHiddenDim(), config.embeddingDim()),
              matrix(
                  file,
                  prefix + "ffn_down.weight",
                  config.embeddingDim(),
                  config.sharedHiddenDim()),
              vector(file, prefix + "pre_ffw_norm_2.weight", config.embeddingDim()),
              vector(file, prefix + "post_ffw_norm_1.weight", config.embeddingDim()),
              vector(file, prefix + "post_ffw_norm_2.weight", config.embeddingDim()),
              vector(file, prefix + "post_ffw_norm.weight", config.embeddingDim()),
              vector(file, prefix + "ffn_gate_inp.scale", config.embeddingDim()),
              matrix(
                  file, prefix + "ffn_gate_inp.weight", config.numExperts(), config.embeddingDim()),
              vector(file, prefix + "ffn_down_exps.scale", config.numExperts()),
              scalar(file, prefix + "layer_output_scale.weight"));
    }

    return new Gemma4Weights(
        tokenEmbedding.dataSegment(),
        tokenEmbedding.type(),
        config.embeddingDim(),
        config.vocabSize(),
        outputNorm,
        ropeFrequencyFactors,
        layers,
        Gemma4TensorLayout.fromGgufFile(file, config));
  }

  void embedToken(int token, float[] output) {
    if (token < 0 || token >= vocabSize) {
      throw new IllegalArgumentException("token out of range: " + token);
    }
    GgufTensorValues.dequantizeRow(tokenEmbedding, tokenEmbeddingType, token, embeddingDim, output);
  }

  MemorySegment tokenEmbedding() {
    return tokenEmbedding;
  }

  GgufTensorType tokenEmbeddingType() {
    return tokenEmbeddingType;
  }

  float[] outputNorm() {
    return outputNorm;
  }

  float[] ropeFrequencyFactors() {
    return ropeFrequencyFactors;
  }

  LayerWeights layer(int layer) {
    if (layer < 0 || layer >= layers.length) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    return layers[layer];
  }

  Gemma4TensorLayout expertLayout() {
    return expertLayout;
  }

  private static Matrix matrix(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = matrixTensor(file, name, rows, columns);
    return new Matrix(tensor.dataSegment(), tensor.type(), rows, columns);
  }

  private static Matrix optionalMatrix(GgufFile file, String name, int rows, int columns) {
    try {
      return matrix(file, name, rows, columns);
    } catch (IllegalArgumentException exception) {
      if (exception.getMessage() != null
          && exception.getMessage().startsWith("Tensor not found:")) {
        return null;
      }
      throw exception;
    }
  }

  private static GgufTensorData matrixTensor(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, columns, rows);
    if (!MATRIX_TYPES.contains(tensor.type())) {
      throw new IllegalArgumentException(name + " has unsupported matrix type " + tensor.type());
    }
    return tensor;
  }

  private static float[] vector(GgufFile file, String name, int length) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, length);
    if (tensor.type() != GgufTensorType.F32) {
      throw new IllegalArgumentException(name + " must be F32, found " + tensor.type());
    }
    return GgufTensorValues.toFloatArray(tensor);
  }

  private static float scalar(GgufFile file, String name) {
    return vector(file, name, 1)[0];
  }

  private static void requireShape(GgufTensorData tensor, long... expected) {
    long[] actual = tensor.shape();
    if (!Arrays.equals(expected, actual)) {
      throw new IllegalArgumentException(
          tensor.name()
              + " shape must be "
              + Arrays.toString(expected)
              + ", found "
              + Arrays.toString(actual));
    }
  }
}
