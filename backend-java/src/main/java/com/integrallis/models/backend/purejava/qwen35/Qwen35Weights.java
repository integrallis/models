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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;

/** Mapped dense Qwen3.5 tensors, kept in their GGUF encodings. */
final class Qwen35Weights {

  private static final Thread ACCESS_PROBE = Thread.ofPlatform().unstarted(() -> {});

  private static final Set<GgufTensorType> MATRIX_TYPES =
      Set.of(
          GgufTensorType.F32,
          GgufTensorType.BF16,
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

  record FullAttention(
      Matrix queryGate,
      Matrix key,
      Matrix value,
      Matrix output,
      float[] queryNorm,
      float[] keyNorm) {}

  record GatedDeltaNet(
      Matrix queryKeyValue,
      Matrix outputGate,
      Matrix beta,
      Matrix alpha,
      float[] convolution,
      float[] timeStepBias,
      float[] decay,
      float[] outputNorm,
      Matrix output) {}

  record Layer(
      float[] attentionNorm,
      float[] postAttentionNorm,
      FullAttention fullAttention,
      GatedDeltaNet gatedDeltaNet,
      Matrix ffnGate,
      Matrix ffnUp,
      Matrix ffnDown) {}

  private final MemorySegment tokenEmbedding;
  private final GgufTensorType tokenEmbeddingType;
  private final int embeddingDim;
  private final int vocabSize;
  private final float[] outputNorm;
  private final Layer[] layers;

  private Qwen35Weights(
      MemorySegment tokenEmbedding,
      GgufTensorType tokenEmbeddingType,
      int embeddingDim,
      int vocabSize,
      float[] outputNorm,
      Layer[] layers) {
    this.tokenEmbedding = tokenEmbedding;
    this.tokenEmbeddingType = tokenEmbeddingType;
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
    this.outputNorm = outputNorm;
    this.layers = layers;
  }

  static Qwen35Weights fromGgufFile(GgufFile file, Qwen35Config config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    GgufTensorData embedding =
        matrixTensor(file, "token_embd.weight", config.vocabSize(), config.embeddingDim());
    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "blk." + layer + ".";
      FullAttention attention = null;
      GatedDeltaNet gatedDeltaNet = null;
      if (config.usesFullAttention(layer)) {
        attention =
            new FullAttention(
                matrix(
                    file,
                    prefix + "attn_q.weight",
                    2 * config.attentionQueryDim(),
                    config.embeddingDim()),
                matrix(
                    file,
                    prefix + "attn_k.weight",
                    config.attentionKeyDim(),
                    config.embeddingDim()),
                matrix(
                    file,
                    prefix + "attn_v.weight",
                    config.attentionKeyDim(),
                    config.embeddingDim()),
                matrix(
                    file,
                    prefix + "attn_output.weight",
                    config.embeddingDim(),
                    config.attentionQueryDim()),
                vector(file, prefix + "attn_q_norm.weight", config.attentionHeadDim()),
                vector(file, prefix + "attn_k_norm.weight", config.attentionHeadDim()));
      } else {
        gatedDeltaNet =
            new GatedDeltaNet(
                matrix(
                    file, prefix + "attn_qkv.weight", config.gdnConvDim(), config.embeddingDim()),
                matrix(
                    file, prefix + "attn_gate.weight", config.gdnValueDim(), config.embeddingDim()),
                matrix(
                    file,
                    prefix + "ssm_beta.weight",
                    config.gdnValueHeads(),
                    config.embeddingDim()),
                matrix(
                    file,
                    prefix + "ssm_alpha.weight",
                    config.gdnValueHeads(),
                    config.embeddingDim()),
                vector(
                    file,
                    prefix + "ssm_conv1d.weight",
                    config.gdnConvDim() * config.gdnConvKernel()),
                vector(file, prefix + "ssm_dt.bias", config.gdnValueHeads()),
                vector(file, prefix + "ssm_a", config.gdnValueHeads()),
                vector(file, prefix + "ssm_norm.weight", config.gdnHeadDim()),
                matrix(
                    file, prefix + "ssm_out.weight", config.embeddingDim(), config.gdnValueDim()));
      }
      layers[layer] =
          new Layer(
              vector(file, prefix + "attn_norm.weight", config.embeddingDim()),
              vector(file, prefix + "post_attention_norm.weight", config.embeddingDim()),
              attention,
              gatedDeltaNet,
              matrix(file, prefix + "ffn_gate.weight", config.hiddenDim(), config.embeddingDim()),
              matrix(file, prefix + "ffn_up.weight", config.hiddenDim(), config.embeddingDim()),
              matrix(file, prefix + "ffn_down.weight", config.embeddingDim(), config.hiddenDim()));
    }
    return new Qwen35Weights(
        embedding.dataSegment(),
        embedding.type(),
        config.embeddingDim(),
        config.vocabSize(),
        vector(file, "output_norm.weight", config.embeddingDim()),
        layers);
  }

  void embedToken(int token, float[] output) {
    if (token < 0 || token >= vocabSize) {
      throw new IllegalArgumentException("token out of range: " + token);
    }
    GgufTensorValues.dequantizeRow(tokenEmbedding, tokenEmbeddingType, token, embeddingDim, output);
  }

  Matrix output() {
    return new Matrix(tokenEmbedding, tokenEmbeddingType, vocabSize, embeddingDim);
  }

  float[] outputNorm() {
    return outputNorm;
  }

  Layer layer(int layer) {
    return layers[layer];
  }

  boolean usesMatrixType(GgufTensorType type) {
    if (tokenEmbeddingType == type) {
      return true;
    }
    for (Layer layer : layers) {
      if (layer.ffnGate().type() == type
          || layer.ffnUp().type() == type
          || layer.ffnDown().type() == type) {
        return true;
      }
      if (layer.fullAttention() != null
          && (layer.fullAttention().queryGate().type() == type
              || layer.fullAttention().key().type() == type
              || layer.fullAttention().value().type() == type
              || layer.fullAttention().output().type() == type)) {
        return true;
      }
      if (layer.gatedDeltaNet() != null
          && (layer.gatedDeltaNet().queryKeyValue().type() == type
              || layer.gatedDeltaNet().outputGate().type() == type
              || layer.gatedDeltaNet().beta().type() == type
              || layer.gatedDeltaNet().alpha().type() == type
              || layer.gatedDeltaNet().output().type() == type)) {
        return true;
      }
    }
    return false;
  }

  boolean hasThreadShareableProjectionWeights() {
    if (!tokenEmbedding.isAccessibleBy(ACCESS_PROBE)) {
      return false;
    }
    for (Layer layer : layers) {
      if (!shareable(layer.ffnGate(), layer.ffnUp(), layer.ffnDown())) {
        return false;
      }
      if (layer.fullAttention() != null
          && !shareable(
              layer.fullAttention().queryGate(),
              layer.fullAttention().key(),
              layer.fullAttention().value(),
              layer.fullAttention().output())) {
        return false;
      }
      if (layer.gatedDeltaNet() != null
          && !shareable(
              layer.gatedDeltaNet().queryKeyValue(),
              layer.gatedDeltaNet().outputGate(),
              layer.gatedDeltaNet().beta(),
              layer.gatedDeltaNet().alpha(),
              layer.gatedDeltaNet().output())) {
        return false;
      }
    }
    return true;
  }

  private static boolean shareable(Matrix... matrices) {
    for (Matrix matrix : matrices) {
      if (!matrix.data().isAccessibleBy(ACCESS_PROBE)) {
        return false;
      }
    }
    return true;
  }

  private static Matrix matrix(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = matrixTensor(file, name, rows, columns);
    return new Matrix(tensor.dataSegment(), tensor.type(), rows, columns);
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
    if (tensor.info().elementCount() != length) {
      throw new IllegalArgumentException(
          name + " must contain " + length + " values, found " + tensor.info().elementCount());
    }
    if (tensor.type() != GgufTensorType.F32) {
      throw new IllegalArgumentException(name + " must be F32, found " + tensor.type());
    }
    return GgufTensorValues.toFloatArray(tensor);
  }

  private static void requireShape(GgufTensorData tensor, long... expected) {
    if (!Arrays.equals(expected, tensor.shape())) {
      throw new IllegalArgumentException(
          tensor.name()
              + " shape must be "
              + Arrays.toString(expected)
              + ", found "
              + Arrays.toString(tensor.shape()));
    }
  }
}
