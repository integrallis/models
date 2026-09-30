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
package com.integrallis.models.backend.purejava.gemma3n;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;

/**
 * Mapped weights for a Gemma 3n decoder.
 *
 * <p>Two things here are not shaped like any other decoder in this backend:
 *
 * <ul>
 *   <li>{@code altup_proj} and {@code altup_unembd_proj} are <b>three-dimensional</b>: one {@code
 *       [n_embd, n_embd]} matrix per inactive stream, stacked. They are sliced the way stacked
 *       expert tensors are, on whole rows so no quantization block is straddled.
 *   <li>A sharing layer carries <b>no key or value projection at all</b>, exactly as the Gemma 4
 *       E-series does. Those fields are null, and {@link Gemma3nConfig#kvSourceLayer(int)} says
 *       which layer's cache to read instead.
 * </ul>
 *
 * <p>{@code laurel_l}, {@code laurel_r} and {@code altup_router} are <b>F16</b> in the published
 * files while the rest of the layer is Q8_0, so F16 has to be an accepted matrix type and the
 * single-vector F16 matmul has to exist. Both were added on 2026-09-29 for Gemma 4 E4B, which
 * carries its per-layer projection as F16 for the same reason.
 */
final class Gemma3nWeights {

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

  record Matrix(MemorySegment data, GgufTensorType type, int rows, int columns) {
    Matrix {
      Objects.requireNonNull(data, "data");
      Objects.requireNonNull(type, "type");
      if (rows <= 0 || columns <= 0) {
        throw new IllegalArgumentException("matrix dimensions must be positive");
      }
    }
  }

  /**
   * One transformer layer.
   *
   * @param keyProjection null on a layer that reads another layer's key-value cache
   * @param valueProjection null on the same layers as {@code keyProjection}
   */
  record LayerWeights(
      float[] attentionNorm,
      Matrix queryProjection,
      Matrix keyProjection,
      Matrix valueProjection,
      Matrix attentionOutput,
      float[] queryNorm,
      float[] keyNorm,
      float[] postAttentionNorm,
      float[] ffnNorm,
      Matrix ffnGate,
      Matrix ffnUp,
      Matrix ffnDown,
      float[] postFfnNorm,
      Matrix perLayerInputGate,
      Matrix perLayerProjection,
      float[] perLayerPostNorm,
      Matrix altupRouter,
      float[] altupRouterNorm,
      Matrix altupPredictCoefficients,
      Matrix altupCorrectCoefficients,
      float[] altupCorrectScale,
      Matrix laurelLeft,
      Matrix laurelRight,
      float[] laurelPostNorm) {}

  private final MemorySegment tokenEmbedding;
  private final GgufTensorType tokenEmbeddingType;
  private final MemorySegment perLayerTokenEmbedding;
  private final GgufTensorType perLayerTokenEmbeddingType;
  private final Matrix perLayerModelProjection;
  private final float[] perLayerProjectionNorm;
  private final Matrix[] altupProjection;
  private final Matrix[] altupUnembedProjection;
  private final float[] outputNorm;
  private final LayerWeights[] layers;
  private final int embeddingDim;
  private final int vocabSize;
  private final int perLayerTotalDim;
  private final int laurelRank;

  private Gemma3nWeights(
      MemorySegment tokenEmbedding,
      GgufTensorType tokenEmbeddingType,
      MemorySegment perLayerTokenEmbedding,
      GgufTensorType perLayerTokenEmbeddingType,
      Matrix perLayerModelProjection,
      float[] perLayerProjectionNorm,
      Matrix[] altupProjection,
      Matrix[] altupUnembedProjection,
      float[] outputNorm,
      LayerWeights[] layers,
      int embeddingDim,
      int vocabSize,
      int perLayerTotalDim,
      int laurelRank) {
    this.tokenEmbedding = tokenEmbedding;
    this.tokenEmbeddingType = tokenEmbeddingType;
    this.perLayerTokenEmbedding = perLayerTokenEmbedding;
    this.perLayerTokenEmbeddingType = perLayerTokenEmbeddingType;
    this.perLayerModelProjection = perLayerModelProjection;
    this.perLayerProjectionNorm = perLayerProjectionNorm;
    this.altupProjection = altupProjection;
    this.altupUnembedProjection = altupUnembedProjection;
    this.outputNorm = outputNorm;
    this.layers = layers;
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
    this.perLayerTotalDim = perLayerTotalDim;
    this.laurelRank = laurelRank;
  }

  static Gemma3nWeights fromGgufFile(GgufFile file, Gemma3nConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    int dim = config.embeddingDim();
    int altup = config.altupInputs();

    GgufTensorData embedding = matrixTensor(file, "token_embd.weight", config.vocabSize(), dim);
    GgufTensorData perLayerEmbedding =
        matrixTensor(
            file, "per_layer_token_embd.weight", config.vocabSize(), config.perLayerTotalDim());

    // The LAuReL rank has no metadata key at all -- the reference hard-codes 64 -- so it is taken
    // from the tensor that defines it. Structural rather than assumed: a model with a different
    // rank
    // loads, and a file whose two LAuReL halves disagree is refused below.
    int laurelRank = (int) file.getTensor("blk.0.laurel_l.weight").shape()[1];
    if (laurelRank <= 0) {
      throw new IllegalArgumentException("blk.0.laurel_l.weight declares a non-positive rank");
    }

    LayerWeights[] layers = new LayerWeights[config.numLayers()];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "blk." + layer + ".";
      boolean ownsKv = config.ownsKvCache(layer);
      layers[layer] =
          new LayerWeights(
              vector(file, prefix + "attn_norm.weight", dim),
              matrix(file, prefix + "attn_q.weight", config.queryDim(), dim),
              ownsKv ? matrix(file, prefix + "attn_k.weight", config.keyDim(), dim) : null,
              ownsKv ? matrix(file, prefix + "attn_v.weight", config.keyDim(), dim) : null,
              matrix(file, prefix + "attn_output.weight", dim, config.queryDim()),
              vector(file, prefix + "attn_q_norm.weight", config.headDim()),
              vector(file, prefix + "attn_k_norm.weight", config.headDim()),
              vector(file, prefix + "post_attention_norm.weight", dim),
              vector(file, prefix + "ffn_norm.weight", dim),
              matrix(file, prefix + "ffn_gate.weight", config.hiddenDim(), dim),
              matrix(file, prefix + "ffn_up.weight", config.hiddenDim(), dim),
              matrix(file, prefix + "ffn_down.weight", dim, config.hiddenDim()),
              vector(file, prefix + "post_ffw_norm.weight", dim),
              matrix(file, prefix + "inp_gate.weight", config.perLayerEmbeddingDim(), dim),
              matrix(file, prefix + "proj.weight", dim, config.perLayerEmbeddingDim()),
              vector(file, prefix + "post_norm.weight", dim),
              matrix(file, prefix + "altup_router.weight", altup, dim),
              vector(file, prefix + "altup_router_norm.weight", dim),
              // altup^2 rows: one coefficient per (source, destination) stream pair.
              matrix(file, prefix + "altup_predict_coef.weight", altup * altup, altup),
              matrix(file, prefix + "altup_correct_coef.weight", altup, altup),
              vector(file, prefix + "altup_correct_scale.weight", dim),
              matrix(file, prefix + "laurel_l.weight", laurelRank, dim),
              matrix(file, prefix + "laurel_r.weight", dim, laurelRank),
              vector(file, prefix + "laurel_post_norm.weight", dim));
    }

    return new Gemma3nWeights(
        embedding.dataSegment(),
        embedding.type(),
        perLayerEmbedding.dataSegment(),
        perLayerEmbedding.type(),
        matrix(file, "per_layer_model_proj.weight", config.perLayerTotalDim(), dim),
        vector(file, "per_layer_proj_norm.weight", config.perLayerEmbeddingDim()),
        streamSlices(file, "altup_proj.weight", dim, altup),
        streamSlices(file, "altup_unembd_proj.weight", dim, altup),
        vector(file, "output_norm.weight", dim),
        layers,
        dim,
        config.vocabSize(),
        config.perLayerTotalDim(),
        laurelRank);
  }

  /**
   * Slices a stacked {@code [dim, dim, altup - 1]} projection into one matrix per inactive stream.
   *
   * <p>{@code altup - 1}, not {@code altup}: the active stream is never projected -- it is the
   * source the others are built from, and the reference concatenates it back unchanged. A loader
   * that expected {@code altup} slices would find the tensor a quarter too small and refuse, which
   * is at least loud; one that sliced {@code altup} out of an {@code altup - 1} tensor would read
   * past it.
   *
   * <p>The cut lands on whole rows, and a row is a whole number of quantization blocks, so no block
   * is straddled. Straddling one reinterprets a block's scale bytes as weights and yields
   * plausible, wrong numbers rather than an error -- the same reasoning as the stacked expert
   * tensors.
   */
  private static Matrix[] streamSlices(GgufFile file, String name, int dim, int altup) {
    int slices = altup - 1;
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, dim, dim, slices);
    if (!MATRIX_TYPES.contains(tensor.type())) {
      throw new IllegalArgumentException(name + " has unsupported matrix type " + tensor.type());
    }
    GgufTensorType type = tensor.type();
    if (dim % type.blockSize() != 0) {
      throw new IllegalArgumentException(
          name
              + ": row length "
              + dim
              + " is not a multiple of the "
              + type.blockSize()
              + "-element "
              + type
              + " block, so the streams cannot be sliced exactly");
    }
    long bytesPerRow = (long) (dim / type.blockSize()) * type.typeSize();
    long bytesPerSlice = bytesPerRow * dim;
    MemorySegment segment = tensor.dataSegment();
    long expected = bytesPerSlice * slices;
    if (segment.byteSize() != expected) {
      throw new IllegalArgumentException(
          name
              + " is "
              + segment.byteSize()
              + " bytes but "
              + slices
              + " slices of "
              + dim
              + "x"
              + dim
              + " need "
              + expected);
    }
    Matrix[] result = new Matrix[slices];
    for (int slice = 0; slice < slices; slice++) {
      result[slice] =
          new Matrix(segment.asSlice(bytesPerSlice * slice, bytesPerSlice), type, dim, dim);
    }
    return result;
  }

  LayerWeights layer(int layer) {
    return layers[layer];
  }

  Matrix perLayerModelProjection() {
    return perLayerModelProjection;
  }

  float[] perLayerProjectionNorm() {
    return perLayerProjectionNorm;
  }

  Matrix altupProjection(int slice) {
    return altupProjection[slice];
  }

  Matrix altupUnembedProjection(int slice) {
    return altupUnembedProjection[slice];
  }

  float[] outputNorm() {
    return outputNorm;
  }

  int laurelRank() {
    return laurelRank;
  }

  /** The output head, which Gemma 3n ties to the token embedding: it carries no output.weight. */
  Matrix output() {
    return new Matrix(tokenEmbedding, tokenEmbeddingType, vocabSize, embeddingDim);
  }

  void embedToken(int token, float[] output) {
    requireToken(token);
    GgufTensorValues.dequantizeRow(tokenEmbedding, tokenEmbeddingType, token, embeddingDim, output);
  }

  /** One token's row of the second embedding table: every layer's slice, end to end. */
  void embedPerLayerToken(int token, float[] output) {
    requireToken(token);
    GgufTensorValues.dequantizeRow(
        perLayerTokenEmbedding, perLayerTokenEmbeddingType, token, perLayerTotalDim, output);
  }

  private void requireToken(int token) {
    if (token < 0 || token >= vocabSize) {
      throw new IllegalArgumentException("token out of range: " + token);
    }
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
