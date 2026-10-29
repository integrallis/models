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
package com.integrallis.models.backend.purejava.flint;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;

/** Shape-checked, zero-copy Flint weights. Expert stacks remain mapped in their GGUF format. */
final class FlintWeights {

  private static final Set<GgufTensorType> MATRIX_TYPES =
      Set.of(
          GgufTensorType.F32,
          GgufTensorType.F16,
          GgufTensorType.BF16,
          GgufTensorType.Q4_0,
          GgufTensorType.Q4_1,
          GgufTensorType.Q5_1,
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
      float[] selectionBias,
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
      Matrix key,
      Matrix value,
      float[] queryNorm,
      float[] keyNorm,
      float[] sinks,
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

  private FlintWeights(
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

  static FlintWeights fromGgufFile(GgufFile file, FlintConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    int dim = config.hiddenSize();
    GgufTensorData embedding = matrixTensor(file, "token_embd.weight", config.vocabSize(), dim);

    LayerWeights[] layers = new LayerWeights[config.numLayers()];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "blk." + layer + ".";
      boolean routed = layer >= config.leadingDenseLayers();
      layers[layer] =
          new LayerWeights(
              vector(file, prefix + "attn_norm.weight", dim),
              matrix(file, prefix + "attn_q.weight", config.queryDimension(), dim),
              matrix(file, prefix + "attn_k.weight", config.keyValueDimension(), dim),
              matrix(file, prefix + "attn_v.weight", config.keyValueDimension(), dim),
              vector(file, prefix + "attn_q_norm.weight", config.headDim()),
              vector(file, prefix + "attn_k_norm.weight", config.headDim()),
              vector(file, prefix + "attn_sinks.weight", config.numHeads()),
              matrix(file, prefix + "attn_output.weight", dim, config.queryDimension()),
              vector(file, prefix + "ffn_norm.weight", dim),
              routed
                  ? null
                  : matrix(file, prefix + "ffn_gate.weight", config.denseHiddenSize(), dim),
              routed ? null : matrix(file, prefix + "ffn_up.weight", config.denseHiddenSize(), dim),
              routed
                  ? null
                  : matrix(file, prefix + "ffn_down.weight", dim, config.denseHiddenSize()),
              routed ? moeFeedForward(file, prefix, config) : null);
    }

    return new FlintWeights(
        embedding.dataSegment(),
        embedding.type(),
        matrix(file, "output.weight", config.vocabSize(), dim),
        vector(file, "output_norm.weight", dim),
        layers,
        dim,
        config.vocabSize());
  }

  private static MoeFeedForward moeFeedForward(GgufFile file, String prefix, FlintConfig config) {
    int dim = config.hiddenSize();
    int expertHidden = config.expertHiddenSize();
    int experts = config.numExperts();
    int sharedHidden = config.sharedHiddenSize();
    return new MoeFeedForward(
        matrix(file, prefix + "ffn_gate_inp.weight", experts, dim),
        file.hasTensor(prefix + "exp_probs_b.bias")
            ? vector(file, prefix + "exp_probs_b.bias", experts)
            : new float[experts],
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
    requireShape(tensor, columns, rows, experts);
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
    return outputHead;
  }

  void embedToken(int token, float[] output) {
    if (token < 0 || token >= vocabSize) {
      throw new IllegalArgumentException("token out of range: " + token);
    }
    GgufTensorValues.dequantizeRow(tokenEmbedding, tokenEmbeddingType, token, embeddingDim, output);
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
    if (columns % tensor.type().blockSize() != 0) {
      throw new IllegalArgumentException(name + " rows must align with quantization blocks");
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
