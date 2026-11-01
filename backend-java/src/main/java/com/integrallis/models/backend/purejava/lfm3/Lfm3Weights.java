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
package com.integrallis.models.backend.purejava.lfm3;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;

/** Mapped LFM3 tensors; convolution and attention groups are exclusive per layer. */
final class Lfm3Weights {

  private static final Set<GgufTensorType> MATRIX_TYPES =
      Set.of(
          GgufTensorType.F32,
          GgufTensorType.F16,
          GgufTensorType.BF16,
          GgufTensorType.Q4_0,
          GgufTensorType.Q5_0,
          GgufTensorType.Q8_0,
          GgufTensorType.Q4_K,
          GgufTensorType.Q5_K,
          GgufTensorType.Q6_K);

  record Matrix(MemorySegment data, GgufTensorType type, int rows, int columns) {}

  /**
   * One layer. Exactly one of the attention group and the convolution group is populated; the other
   * is null throughout, so a mistake in the layer plan is a null dereference rather than a silently
   * wrong block.
   */
  record LayerWeights(
      float[] attentionNorm,
      float[] ffnNorm,
      float[] postAttentionNorm,
      float[] postFfnNorm,
      Matrix attentionGate,
      Matrix gateProjection,
      Matrix upProjection,
      Matrix downProjection,
      Matrix queryProjection,
      Matrix keyProjection,
      Matrix valueProjection,
      Matrix attentionOutputProjection,
      float[] queryNorm,
      float[] keyNorm,
      Matrix shortConvInProjection,
      float[] shortConvKernel,
      Matrix shortConvOutProjection) {}

  private final MemorySegment tokenEmbedding;
  private final GgufTensorType tokenEmbeddingType;
  private final float[] outputNorm;
  private final Matrix output;
  private final float[] ropeFactors;
  private final LayerWeights[] layers;
  private final int embeddingDim;
  private final int vocabSize;

  private Lfm3Weights(
      MemorySegment tokenEmbedding,
      GgufTensorType tokenEmbeddingType,
      float[] outputNorm,
      Matrix output,
      float[] ropeFactors,
      LayerWeights[] layers,
      int embeddingDim,
      int vocabSize) {
    this.tokenEmbedding = tokenEmbedding;
    this.tokenEmbeddingType = tokenEmbeddingType;
    this.outputNorm = outputNorm;
    this.output = output;
    this.ropeFactors = ropeFactors;
    this.layers = layers;
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
  }

  static Lfm3Weights fromGgufFile(GgufFile file, Lfm3Config config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    int dim = config.embeddingDim();
    GgufTensorData tokenEmbedding =
        matrixTensor(file, "token_embd.weight", config.vocabSize(), dim);

    LayerWeights[] layers = new LayerWeights[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "blk." + layer + ".";
      boolean attends = config.usesAttention(layer);
      layers[layer] =
          new LayerWeights(
              vector(file, prefix + "attn_norm.weight", dim),
              vector(file, prefix + "ffn_norm.weight", dim),
              optionalVector(file, prefix + "post_attention_norm.weight", dim),
              optionalVector(file, prefix + "post_ffw_norm.weight", dim),
              attends ? matrix(file, prefix + "attn_gate.weight", config.numHeads(), dim) : null,
              matrix(file, prefix + "ffn_gate.weight", config.hiddenDim(), dim),
              matrix(file, prefix + "ffn_up.weight", config.hiddenDim(), dim),
              matrix(file, prefix + "ffn_down.weight", dim, config.hiddenDim()),
              attends ? matrix(file, prefix + "attn_q.weight", config.queryDim(), dim) : null,
              attends ? matrix(file, prefix + "attn_k.weight", config.keyDim(layer), dim) : null,
              attends ? matrix(file, prefix + "attn_v.weight", config.keyDim(layer), dim) : null,
              attends ? matrix(file, prefix + "attn_output.weight", dim, config.queryDim()) : null,
              attends ? vector(file, prefix + "attn_q_norm.weight", config.headDim()) : null,
              attends ? vector(file, prefix + "attn_k_norm.weight", config.headDim()) : null,
              attends ? null : matrix(file, prefix + "shortconv.in_proj.weight", 3 * dim, dim),
              attends
                  ? null
                  : convKernel(
                      file, prefix + "shortconv.conv.weight", config.shortConvCache(), dim),
              attends ? null : matrix(file, prefix + "shortconv.out_proj.weight", dim, dim));
    }

    for (LayerWeights layer : layers) {
      if ((layer.postAttentionNorm() == null) != (layer.postFfnNorm() == null))
        throw new IllegalArgumentException("LFM3 requires both post norms or neither");
    }
    return new Lfm3Weights(
        tokenEmbedding.dataSegment(),
        tokenEmbedding.type(),
        vector(file, "output_norm.weight", dim),
        matrix(
            file,
            file.hasTensor("output.weight") ? "output.weight" : "token_embd.weight",
            config.vocabSize(),
            dim),
        vector(file, "rope_freqs.weight", config.headDim() / 2),
        layers,
        dim,
        config.vocabSize());
  }

  float[] ropeFactors() {
    return ropeFactors;
  }

  private static float[] optionalVector(GgufFile file, String name, int length) {
    return file.hasTensor(name) ? vector(file, name, length) : null;
  }

  LayerWeights layer(int index) {
    return layers[index];
  }

  float[] outputNorm() {
    return outputNorm;
  }

  /**
   * The widest Q4_0 projection in the model, which sizes the batched kernel's lane scratch.
   *
   * <p>Zero when the file carries no Q4_0 matrix, in which case no lane scratch is needed at all.
   */
  int maxQ40ProjectionRows() {
    int rows = 0;
    for (LayerWeights layer : layers) {
      rows = maxQ40Rows(rows, layer.gateProjection());
      rows = maxQ40Rows(rows, layer.upProjection());
      rows = maxQ40Rows(rows, layer.downProjection());
      rows = maxQ40Rows(rows, layer.queryProjection());
      rows = maxQ40Rows(rows, layer.keyProjection());
      rows = maxQ40Rows(rows, layer.valueProjection());
      rows = maxQ40Rows(rows, layer.attentionOutputProjection());
      rows = maxQ40Rows(rows, layer.shortConvInProjection());
      rows = maxQ40Rows(rows, layer.shortConvOutProjection());
    }
    return rows;
  }

  private static int maxQ40Rows(int rows, Matrix matrix) {
    if (matrix == null || matrix.type() != GgufTensorType.Q4_0) {
      return rows;
    }
    return Math.max(rows, matrix.rows());
  }

  /** Output projection, with a tied embedding fallback. */
  MemorySegment outputProjection() {
    return output.data();
  }

  GgufTensorType outputProjectionType() {
    return output.type();
  }

  void embedToken(int token, float[] output) {
    if (token < 0 || token >= vocabSize) {
      throw new IllegalArgumentException("token out of range: " + token);
    }
    GgufTensorValues.dequantizeRow(tokenEmbedding, tokenEmbeddingType, token, embeddingDim, output);
  }

  /** Converts the GGUF channel-major taps once at load into vectors' tap-major layout. */
  private static float[] convKernel(GgufFile file, String name, int lCache, int dim) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, lCache, dim);
    if (tensor.type() != GgufTensorType.F32) {
      throw new IllegalArgumentException(name + " must be F32, found " + tensor.type());
    }
    float[] original = GgufTensorValues.toFloatArray(tensor), tapMajor = new float[original.length];
    for (int channel = 0; channel < dim; channel++)
      for (int tap = 0; tap < lCache; tap++)
        tapMajor[tap * dim + channel] = original[channel * lCache + tap];
    return tapMajor;
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
    requireShape(tensor, length);
    if (tensor.type() != GgufTensorType.F32) {
      throw new IllegalArgumentException(name + " must be F32, found " + tensor.type());
    }
    return GgufTensorValues.toFloatArray(tensor);
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
