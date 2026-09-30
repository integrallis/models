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
package com.integrallis.models.backend.purejava.lfm2;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;

/**
 * LFM2 weights, loaded from a GGUF.
 *
 * <p>Two things differ from the Llama tensor set and both are read from the file rather than
 * assumed:
 *
 * <ul>
 *   <li><b>Layers are not uniform.</b> An attention layer carries the usual projections and
 *       per-head norms; a convolutional layer carries {@code shortconv.in_proj}, {@code
 *       shortconv.conv} and {@code shortconv.out_proj} and no attention tensors at all. Which is
 *       which comes from the config's per-layer kv head counts.
 *   <li><b>The output projection is tied to the embedding, and the final norm is misnamed.</b>
 *       LFM2.5-1.2B publishes no {@code output.weight}, so the vocabulary projection reuses {@code
 *       token_embd.weight}; and it publishes no {@code output_norm.weight} either -- its final norm
 *       is the tensor called {@code token_embd_norm}. llama.cpp maps its LFM2 output-norm constant
 *       to that string with the comment "fix for wrong tensor name". Taking the name at face value
 *       would normalise the embedding before the first layer instead of the state after the last.
 * </ul>
 */
final class Lfm2Weights {

  private static final Set<GgufTensorType> MATRIX_TYPES =
      Set.of(
          GgufTensorType.F32,
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
  private final LayerWeights[] layers;
  private final int embeddingDim;
  private final int vocabSize;

  private Lfm2Weights(
      MemorySegment tokenEmbedding,
      GgufTensorType tokenEmbeddingType,
      float[] outputNorm,
      LayerWeights[] layers,
      int embeddingDim,
      int vocabSize) {
    this.tokenEmbedding = tokenEmbedding;
    this.tokenEmbeddingType = tokenEmbeddingType;
    this.outputNorm = outputNorm;
    this.layers = layers;
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
  }

  static Lfm2Weights fromGgufFile(GgufFile file, Lfm2Config config) {
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

    return new Lfm2Weights(
        tokenEmbedding.dataSegment(),
        tokenEmbedding.type(),
        // NOT an embedding norm, despite the name. llama.cpp maps its LFM2 output-norm tensor to
        // the
        // string "token_embd_norm" with the comment "fix for wrong tensor name", and LFM2.5-1.2B
        // carries no output_norm.weight at all -- verified from the file's own tensor list. Read as
        // an
        // embedding norm it would be applied before the first layer instead of after the last.
        vector(file, "token_embd_norm.weight", dim),
        layers,
        dim,
        config.vocabSize());
  }

  LayerWeights layer(int index) {
    return layers[index];
  }

  float[] outputNorm() {
    return outputNorm;
  }

  /** The vocabulary projection, which is the embedding table: LFM2 ties them. */
  MemorySegment outputProjection() {
    return tokenEmbedding;
  }

  GgufTensorType outputProjectionType() {
    return tokenEmbeddingType;
  }

  void embedToken(int token, float[] output) {
    if (token < 0 || token >= vocabSize) {
      throw new IllegalArgumentException("token out of range: " + token);
    }
    GgufTensorValues.dequantizeRow(tokenEmbedding, tokenEmbeddingType, token, embeddingDim, output);
  }

  /**
   * The depthwise taps, as a flat array with the taps contiguous per channel.
   *
   * <p>The GGUF shape is {@code [l_cache, n_embd]}, whose first dimension is the fastest-varying,
   * so the file already stores the taps contiguously per channel -- which is the layout {@link
   * Lfm2ShortConv} reads and the same one ggml's {@code ssm_conv} indexes with {@code i0 + i1*nc}.
   */
  private static float[] convKernel(GgufFile file, String name, int lCache, int dim) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, lCache, dim);
    if (tensor.type() != GgufTensorType.F32) {
      throw new IllegalArgumentException(name + " must be F32, found " + tensor.type());
    }
    return GgufTensorValues.toFloatArray(tensor);
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
