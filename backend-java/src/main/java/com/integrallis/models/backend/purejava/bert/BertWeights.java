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
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Mapped BERT GGUF tensors plus the small F32 vectors decoded at load time. */
final class BertWeights {

  /**
   * Opt-in: widen K-quant projection weights to F32 at load (#360). Default off, because it changes
   * the scores a qualified reranker produces past its qualification tolerance (measured: up to
   * 0.038 on the MS MARCO MiniLM-L6 probe, ordering preserved, and the pinned ONNX equivalence test
   * fails by 0.075 against a 0.05 tolerance) and makes cold load 1.77x slower, in exchange for
   * 1.33x faster warm scoring and +12.2 MB on that model. A pinned qualification record is a claim
   * about exact outputs, so an arithmetic change is selectable, never the default - see {@link
   * #preDequantizes}.
   */
  static final String PRE_DEQUANTIZE_K_QUANT_PROPERTY = "models.purejava.bert.preDequantizeKQuant";

  private static final boolean PRE_DEQUANTIZE_K_QUANT =
      Boolean.getBoolean(PRE_DEQUANTIZE_K_QUANT_PROPERTY);

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
      float[] outputNormBias,
      Matrix queryKeyValue,
      Matrix feedForwardGate) {}

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
    return load(file, config, GgufBatchedMatrixKernel.none());
  }

  /**
   * Loads the weights, deciding each projection's storage against the kernel that will consume it.
   *
   * <p>The kernel is needed here because #360's pre-dequantization is only ever a win where no
   * kernel can take the quantized form: see {@link #matrix}.
   */
  static BertWeights load(GgufFile file, BertConfig config, GgufBatchedMatrixKernel matrixKernel) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    Objects.requireNonNull(matrixKernel, "matrixKernel");
    int dim = config.embeddingDim();
    Matrix tokenEmbeddings = table(file, "token_embd.weight", config.vocabSize(), dim);
    Matrix positionEmbeddings =
        config.usesRotaryPositions()
            ? null
            : table(file, "position_embd.weight", config.contextLength(), dim);
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
      boolean nomic = config.usesRotaryPositions();
      layers[layer] =
          new Layer(
              nomic ? null : projection(file, prefix + "attn_q.weight", dim, dim, matrixKernel),
              nomic ? zero(dim) : vector(file, prefix + "attn_q.bias", dim),
              nomic ? null : projection(file, prefix + "attn_k.weight", dim, dim, matrixKernel),
              nomic ? zero(dim) : vector(file, prefix + "attn_k.bias", dim),
              nomic ? null : projection(file, prefix + "attn_v.weight", dim, dim, matrixKernel),
              nomic ? zero(dim) : vector(file, prefix + "attn_v.bias", dim),
              projection(file, prefix + "attn_output.weight", dim, dim, matrixKernel),
              nomic ? zero(dim) : vector(file, prefix + "attn_output.bias", dim),
              vector(file, prefix + "attn_output_norm.weight", dim),
              vector(file, prefix + "attn_output_norm.bias", dim),
              projection(file, prefix + "ffn_up.weight", config.hiddenDim(), dim, matrixKernel),
              nomic
                  ? zero(config.hiddenDim())
                  : vector(file, prefix + "ffn_up.bias", config.hiddenDim()),
              projection(file, prefix + "ffn_down.weight", dim, config.hiddenDim(), matrixKernel),
              nomic ? zero(dim) : vector(file, prefix + "ffn_down.bias", dim),
              vector(file, prefix + "layer_output_norm.weight", dim),
              vector(file, prefix + "layer_output_norm.bias", dim),
              nomic
                  ? projection(file, prefix + "attn_qkv.weight", dim * 3, dim, matrixKernel)
                  : null,
              nomic
                  ? projection(
                      file, prefix + "ffn_gate.weight", config.hiddenDim(), dim, matrixKernel)
                  : null);
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
    if (positionEmbeddings == null) {
      throw new IllegalStateException(
          "Nomic-BERT uses rotary positions, not a position embedding table");
    }
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
      if (layer.query() != null) {
        rows = maxQ40Rows(rows, layer.query());
        rows = maxQ40Rows(rows, layer.key());
        rows = maxQ40Rows(rows, layer.value());
      }
      if (layer.queryKeyValue() != null) {
        rows = maxQ40Rows(rows, layer.queryKeyValue());
      }
      rows = maxQ40Rows(rows, layer.attentionOutput());
      rows = maxQ40Rows(rows, layer.feedForwardUp());
      if (layer.feedForwardGate() != null) {
        rows = maxQ40Rows(rows, layer.feedForwardGate());
      }
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

  private static float[] zero(int size) {
    return new float[size];
  }

  /**
   * Whether a tensor is widened to F32 at load rather than kept in its stored K-quant form.
   *
   * <p>#360 made this trade - roughly 7x the memory of those weights for per-token latency -
   * because on-the-fly K-quant dequantization in pure Java is slower than F32 dot products on
   * already-widened weights. Measured on the registered host class it is real: 1.33x faster warm
   * scoring on the MS MARCO MiniLM-L6 reranker. It also costs 1.77x on cold load and <b>changes the
   * scores</b>: F32 matmul on widened weights is different arithmetic from the quantized-activation
   * K-quant matmul, not a reordering of it. That is why {@code enabled} is an explicit input and
   * the property defaults to off - a qualified model's record is a claim about exact outputs. It
   * pays in a narrower set of cases than it first appears (#452), and both restrictions below are
   * the difference between a trade and a straight loss:
   *
   * <ul>
   *   <li><b>Projections only.</b> The tables - {@code token_embd} and {@code position_embd} - are
   *       read one row at a time by lookup and never multiplied, so there is no per-token
   *       whole-matrix dequantization to remove. Widening them spends the most memory in the model,
   *       {@code vocabSize * dim}, to buy nothing.
   *   <li><b>Only where no kernel takes the quantized form.</b> {@code BertForwardPass.project}
   *       picks its route with {@code matrixKernel.isEligible(matrix.type(), ...)}. Widening first
   *       would hand the kernel F32 and decide, at load time and invisibly, a route the kernel
   *       exists to decide. {@code GgufRerankingModel} installs no kernel and is where this pays;
   *       {@code PureJavaBackend}'s {@code bert} and {@code nomic-bert} families install a real one
   *       and keep their quantized path.
   * </ul>
   */
  static boolean preDequantizes(
      boolean enabled,
      boolean projection,
      GgufTensorType type,
      GgufBatchedMatrixKernel matrixKernel) {
    return enabled
        && projection
        && !matrixKernel.supports(type)
        && (type == GgufTensorType.Q4_K
            || type == GgufTensorType.Q5_K
            || type == GgufTensorType.Q6_K);
  }

  private static Matrix table(GgufFile file, String name, int rows, int columns) {
    return matrix(file, name, rows, columns, false, GgufBatchedMatrixKernel.none());
  }

  private static Matrix projection(
      GgufFile file, String name, int rows, int columns, GgufBatchedMatrixKernel matrixKernel) {
    return matrix(file, name, rows, columns, true, matrixKernel);
  }

  private static Matrix matrix(
      GgufFile file,
      String name,
      int rows,
      int columns,
      boolean projection,
      GgufBatchedMatrixKernel matrixKernel) {
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
    if (preDequantizes(PRE_DEQUANTIZE_K_QUANT, projection, tensor.type(), matrixKernel)) {
      float[] values = GgufTensorValues.toFloatArray(tensor);
      MemorySegment f32Data = MemorySegment.ofArray(values);
      return new Matrix(f32Data, GgufTensorType.F32, rows, columns);
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
