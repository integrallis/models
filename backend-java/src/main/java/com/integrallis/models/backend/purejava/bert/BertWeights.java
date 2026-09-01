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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Mapped BERT GGUF tensors plus the small F32 vectors decoded at load time. */
final class BertWeights {

  /** A row-major matrix stored in GGUF's [columns, rows] shape convention. */
  record Matrix(MemorySegment data, GgufTensorType type, int rows, int columns) {
    Matrix {
      Objects.requireNonNull(data, "data");
      Objects.requireNonNull(type, "type");
      if (rows <= 0 || columns <= 0) {
        throw new IllegalArgumentException("matrix dimensions must be positive");
      }
    }
  }

  /** One post-normalized BERT transformer block. */
  record Layer(
      Matrix query,
      float[] queryBias,
      Matrix key,
      float[] keyBias,
      Matrix value,
      float[] valueBias,
      Matrix attentionOutput,
      float[] attentionOutputBias,
      float[] attentionNormWeight,
      float[] attentionNormBias,
      Matrix feedForwardUp,
      float[] feedForwardUpBias,
      Matrix feedForwardDown,
      float[] feedForwardDownBias,
      float[] outputNormWeight,
      float[] outputNormBias) {}

  private final Matrix tokenEmbeddings;
  private final Matrix positionEmbeddings;
  private final Matrix tokenTypeEmbeddings;
  private final float[] inputNormWeight;
  private final float[] inputNormBias;
  private final Layer[] layers;

  private BertWeights(
      Matrix tokenEmbeddings,
      Matrix positionEmbeddings,
      Matrix tokenTypeEmbeddings,
      float[] inputNormWeight,
      float[] inputNormBias,
      Layer[] layers) {
    this.tokenEmbeddings = tokenEmbeddings;
    this.positionEmbeddings = positionEmbeddings;
    this.tokenTypeEmbeddings = tokenTypeEmbeddings;
    this.inputNormWeight = inputNormWeight;
    this.inputNormBias = inputNormBias;
    this.layers = layers;
  }

  /** Loads and validates the tensor names and dimensions defined by the BERT GGUF convention. */
  static BertWeights load(GgufFile file, BertConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    int dim = config.embeddingDim();
    Matrix tokenEmbeddings = table(file, "token_embd.weight", config.vocabSize(), dim);
    Matrix positionEmbeddings = table(file, "position_embd.weight", config.contextLength(), dim);
    GgufTensorData tokenTypes = file.getTensor("token_types.weight");
    long[] tokenTypeShape = tokenTypes.shape();
    if (tokenTypeShape.length != 2 || tokenTypeShape[0] != dim || tokenTypeShape[1] < 1) {
      throw new IllegalArgumentException(
          "token_types.weight shape must be ["
              + dim
              + ", >=1]: "
              + Arrays.toString(tokenTypeShape));
    }
    Matrix tokenTypeEmbeddings =
        new Matrix(
            tokenTypes.dataSegment(), tokenTypes.type(), Math.toIntExact(tokenTypeShape[1]), dim);

    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "blk." + layer + ".";
      layers[layer] =
          new Layer(
              projection(file, prefix + "attn_q.weight", dim, dim),
              vector(file, prefix + "attn_q.bias", dim),
              projection(file, prefix + "attn_k.weight", dim, dim),
              vector(file, prefix + "attn_k.bias", dim),
              projection(file, prefix + "attn_v.weight", dim, dim),
              vector(file, prefix + "attn_v.bias", dim),
              projection(file, prefix + "attn_output.weight", dim, dim),
              vector(file, prefix + "attn_output.bias", dim),
              vector(file, prefix + "attn_output_norm.weight", dim),
              vector(file, prefix + "attn_output_norm.bias", dim),
              projection(file, prefix + "ffn_up.weight", config.hiddenDim(), dim),
              vector(file, prefix + "ffn_up.bias", config.hiddenDim()),
              projection(file, prefix + "ffn_down.weight", dim, config.hiddenDim()),
              vector(file, prefix + "ffn_down.bias", dim),
              vector(file, prefix + "layer_output_norm.weight", dim),
              vector(file, prefix + "layer_output_norm.bias", dim));
    }

    return new BertWeights(
        tokenEmbeddings,
        positionEmbeddings,
        tokenTypeEmbeddings,
        vector(file, "token_embd_norm.weight", dim),
        vector(file, "token_embd_norm.bias", dim),
        layers);
  }

  Layer layer(int index) {
    return layers[index];
  }

  float[] inputNormWeight() {
    return inputNormWeight;
  }

  float[] inputNormBias() {
    return inputNormBias;
  }

  void tokenEmbedding(int token, float[] output) {
    row(tokenEmbeddings, token, output);
  }

  void positionEmbedding(int position, float[] output) {
    row(positionEmbeddings, position, output);
  }

  int tokenTypeCount() {
    return tokenTypeEmbeddings.rows();
  }

  void tokenTypeEmbedding(int tokenType, float[] output) {
    row(tokenTypeEmbeddings, tokenType, output);
  }

  /** Largest Q4_0 output width, used to size the kernel's lane scratch once per sequence. */
  int maxQ40ProjectionRows() {
    int rows = 0;
    for (Layer layer : layers) {
      rows = maxQ40Rows(rows, layer.query());
      rows = maxQ40Rows(rows, layer.key());
      rows = maxQ40Rows(rows, layer.value());
      rows = maxQ40Rows(rows, layer.attentionOutput());
      rows = maxQ40Rows(rows, layer.feedForwardUp());
      rows = maxQ40Rows(rows, layer.feedForwardDown());
    }
    return rows;
  }

  private static void row(Matrix table, int row, float[] output) {
    if (row < 0 || row >= table.rows()) {
      throw new IllegalArgumentException("embedding row out of range: " + row);
    }
    GgufTensorValues.dequantizeRow(table.data(), table.type(), row, table.columns(), output);
  }

  private static int maxQ40Rows(int current, Matrix matrix) {
    return matrix.type() == GgufTensorType.Q4_0 ? Math.max(current, matrix.rows()) : current;
  }

  private static Matrix table(GgufFile file, String name, int rows, int columns) {
    return matrix(file, name, rows, columns, false);
  }

  private static Matrix projection(GgufFile file, String name, int rows, int columns) {
    return matrix(file, name, rows, columns, true);
  }

  private static Matrix matrix(
      GgufFile file, String name, int rows, int columns, boolean projection) {
    GgufTensorData tensor = file.getTensor(name);
    long[] expected = {columns, rows};
    if (!Arrays.equals(tensor.shape(), expected)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expected)
              + ": "
              + Arrays.toString(tensor.shape()));
    }
    if (projection && !TensorOps.supportsBatchedMatmul(tensor.type())) {
      throw new IllegalArgumentException(
          name + " uses unsupported batched projection type " + tensor.type());
    }
    return new Matrix(tensor.dataSegment(), tensor.type(), rows, columns);
  }

  private static float[] vector(GgufFile file, String name, int size) {
    GgufTensorData tensor = file.getTensor(name);
    long[] expected = {size};
    if (!Arrays.equals(tensor.shape(), expected)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expected)
              + ": "
              + Arrays.toString(tensor.shape()));
    }
    return GgufTensorValues.toFloatArray(tensor);
  }
}
