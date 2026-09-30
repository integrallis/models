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
package com.integrallis.models.backend.purejava.deepseek2;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;

/**
 * Mapped weights for a DeepSeek-V2 family decoder.
 *
 * <p>Latent attention: a layer carries {@code attn_kv_a_mqa} (the compressing projection), a norm
 * over the latent, and {@code attn_kv_b} (the decompressing one) in place of the usual key and
 * value projections. Only the unsplit {@code attn_kv_b} is read -- the {@code attn_k_b}/{@code
 * attn_v_b} pair the absorbed form uses is absent from the published DeepSeek-Coder-V2-Lite file,
 * and a loader that required it would refuse that file.
 *
 * <p>The feed-forward differs by layer: the leading ones are dense and the rest routed, per {@code
 * leading_dense_block_count}. The shared experts are <b>already fused</b> in the published tensors
 * -- {@code ffn_gate_shexp} is 2816 wide for two 1408-wide experts -- so they load as one wide
 * expert and nothing downstream needs to know there are two.
 */
final class Deepseek2Weights {

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
          GgufTensorType.Q6_K,
          GgufTensorType.MXFP4);

  record Matrix(MemorySegment data, GgufTensorType type, int rows, int columns) {
    Matrix {
      Objects.requireNonNull(data, "data");
      Objects.requireNonNull(type, "type");
      if (rows <= 0 || columns <= 0) {
        throw new IllegalArgumentException("matrix dimensions must be positive");
      }
    }
  }

  /** The routed feed-forward of one layer, absent on the leading dense ones. */
  record MoeFeedForward(
      Matrix router,
      Matrix[] gate,
      Matrix[] up,
      Matrix[] down,
      Matrix sharedGate,
      Matrix sharedUp,
      Matrix sharedDown) {}

  /**
   * @param ffnGate null on a routed layer
   * @param moe null on a leading dense layer
   */
  record LayerWeights(
      float[] attentionNorm,
      Matrix query,
      Matrix keyValueCompress,
      float[] keyValueLatentNorm,
      Matrix keyValueDecompress,
      Matrix attentionOutput,
      float[] ffnNorm,
      Matrix ffnGate,
      Matrix ffnUp,
      Matrix ffnDown,
      MoeFeedForward moe) {}

  private final MemorySegment tokenEmbedding;
  private final GgufTensorType tokenEmbeddingType;
  private final Matrix outputHead;
  private final float[] outputNorm;
  private final LayerWeights[] layers;
  private final int embeddingDim;
  private final int vocabSize;

  private Deepseek2Weights(
      MemorySegment tokenEmbedding,
      GgufTensorType tokenEmbeddingType,
      Matrix outputHead,
      float[] outputNorm,
      LayerWeights[] layers,
      int embeddingDim,
      int vocabSize) {
    this.tokenEmbedding = tokenEmbedding;
    this.tokenEmbeddingType = tokenEmbeddingType;
    this.outputHead = outputHead;
    this.outputNorm = outputNorm;
    this.layers = layers;
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
  }

  static Deepseek2Weights fromGgufFile(GgufFile file, Deepseek2Config config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    int dim = config.embeddingDim();
    GgufTensorData embedding = matrixTensor(file, "token_embd.weight", config.vocabSize(), dim);

    LayerWeights[] layers = new LayerWeights[config.numLayers()];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "blk." + layer + ".";
      boolean routed = config.usesMixtureOfExperts(layer);
      layers[layer] =
          new LayerWeights(
              vector(file, prefix + "attn_norm.weight", dim),
              matrix(file, prefix + "attn_q.weight", config.queryDim(), dim),
              matrix(file, prefix + "attn_kv_a_mqa.weight", config.compressedKeyValueDim(), dim),
              vector(file, prefix + "attn_kv_a_norm.weight", config.kvLoraRank()),
              matrix(
                  file,
                  prefix + "attn_kv_b.weight",
                  config.decompressedKeyValueDim(),
                  config.kvLoraRank()),
              matrix(file, prefix + "attn_output.weight", dim, config.cachedValueDim()),
              vector(file, prefix + "ffn_norm.weight", dim),
              routed ? null : matrix(file, prefix + "ffn_gate.weight", config.hiddenDim(), dim),
              routed ? null : matrix(file, prefix + "ffn_up.weight", config.hiddenDim(), dim),
              routed ? null : matrix(file, prefix + "ffn_down.weight", dim, config.hiddenDim()),
              routed ? moeFeedForward(file, prefix, config) : null);
    }

    return new Deepseek2Weights(
        embedding.dataSegment(),
        embedding.type(),
        optionalOutputHead(file, config),
        vector(file, "output_norm.weight", dim),
        layers,
        dim,
        config.vocabSize());
  }

  /** The model's own output head, or null when it ties the head to the token embedding. */
  private static Matrix optionalOutputHead(GgufFile file, Deepseek2Config config) {
    if (!file.hasTensor("output.weight")) {
      return null;
    }
    return matrix(file, "output.weight", config.vocabSize(), config.embeddingDim());
  }

  private static MoeFeedForward moeFeedForward(
      GgufFile file, String prefix, Deepseek2Config config) {
    int dim = config.embeddingDim();
    int expertHidden = config.expertHiddenDim();
    int experts = config.numExperts();
    int sharedHidden = config.sharedExpertHiddenDim();
    return new MoeFeedForward(
        matrix(file, prefix + "ffn_gate_inp.weight", experts, dim),
        expertSlices(
            file.getTensor(prefix + "ffn_gate_exps.weight"),
            expertHidden,
            dim,
            experts,
            prefix + "ffn_gate_exps.weight"),
        expertSlices(
            file.getTensor(prefix + "ffn_up_exps.weight"),
            expertHidden,
            dim,
            experts,
            prefix + "ffn_up_exps.weight"),
        // The down projection maps the expert width back to the model width, so its rows and
        // columns
        // are the other way round from gate and up.
        expertSlices(
            file.getTensor(prefix + "ffn_down_exps.weight"),
            dim,
            expertHidden,
            experts,
            prefix + "ffn_down_exps.weight"),
        matrix(file, prefix + "ffn_gate_shexp.weight", sharedHidden, dim),
        matrix(file, prefix + "ffn_up_shexp.weight", sharedHidden, dim),
        matrix(file, prefix + "ffn_down_shexp.weight", dim, sharedHidden));
  }

  /**
   * Slices one stacked expert tensor into per-expert views.
   *
   * <p>GGUF stacks experts along the last dimension, so expert {@code e} occupies a contiguous
   * block of {@code rows} rows. The cut lands on whole rows, and a row is a whole number of
   * quantization blocks, so no block is straddled -- straddling one reinterprets a block's scale
   * bytes as weights and yields plausible, wrong numbers rather than an error.
   */
  private static Matrix[] expertSlices(
      GgufTensorData tensor, int rows, int columns, int experts, String name) {
    GgufTensorType type = tensor.type();
    if (!MATRIX_TYPES.contains(type)) {
      throw new IllegalArgumentException(name + " has unsupported matrix type " + type);
    }
    if (columns % type.blockSize() != 0) {
      throw new IllegalArgumentException(
          name
              + ": row length "
              + columns
              + " is not a multiple of the "
              + type.blockSize()
              + "-element "
              + type
              + " block, so experts cannot be sliced exactly");
    }
    long bytesPerRow = (long) (columns / type.blockSize()) * type.typeSize();
    long bytesPerExpert = bytesPerRow * rows;
    MemorySegment segment = tensor.dataSegment();
    long expected = bytesPerExpert * experts;
    if (segment.byteSize() != expected) {
      throw new IllegalArgumentException(
          name
              + " is "
              + segment.byteSize()
              + " bytes but "
              + experts
              + " experts of "
              + rows
              + "x"
              + columns
              + " need "
              + expected);
    }
    Matrix[] slices = new Matrix[experts];
    for (int expert = 0; expert < experts; expert++) {
      slices[expert] =
          new Matrix(segment.asSlice(bytesPerExpert * expert, bytesPerExpert), type, rows, columns);
    }
    return slices;
  }

  LayerWeights layer(int layer) {
    return layers[layer];
  }

  float[] outputNorm() {
    return outputNorm;
  }

  Matrix output() {
    return outputHead != null
        ? outputHead
        : new Matrix(tokenEmbedding, tokenEmbeddingType, vocabSize, embeddingDim);
  }

  void embedToken(int token, float[] output) {
    if (token < 0 || token >= vocabSize) {
      throw new IllegalArgumentException("token out of range: " + token);
    }
    GgufTensorValues.dequantizeRow(tokenEmbedding, tokenEmbeddingType, token, embeddingDim, output);
  }

  /**
   * The widest input any projection takes, which is what the quantized-activation scratch holds.
   */
  int widestProjectionInput() {
    int widest = embeddingDim;
    for (LayerWeights layer : layers) {
      for (Matrix matrix : layerMatrices(layer)) {
        widest = Math.max(widest, matrix.columns());
      }
    }
    return widest;
  }

  /** Every projection a layer holds, dense or routed. Asked of the layer, never assumed. */
  private static java.util.List<Matrix> layerMatrices(LayerWeights layer) {
    java.util.List<Matrix> matrices = new java.util.ArrayList<>();
    matrices.add(layer.query());
    matrices.add(layer.keyValueCompress());
    matrices.add(layer.keyValueDecompress());
    matrices.add(layer.attentionOutput());
    if (layer.moe() == null) {
      matrices.add(layer.ffnGate());
      matrices.add(layer.ffnUp());
      matrices.add(layer.ffnDown());
      return matrices;
    }
    MoeFeedForward moe = layer.moe();
    matrices.add(moe.router());
    matrices.addAll(Arrays.asList(moe.gate()));
    matrices.addAll(Arrays.asList(moe.up()));
    matrices.addAll(Arrays.asList(moe.down()));
    matrices.add(moe.sharedGate());
    matrices.add(moe.sharedUp());
    matrices.add(moe.sharedDown());
    return matrices;
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
    // GGUF drops trailing dimensions of 1, so a single-row matrix is written as a 1-D tensor.
    if (expected.length == 2
        && expected[1] == 1L
        && tensor.shape().length == 1
        && tensor.shape()[0] == expected[0]) {
      return;
    }
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
