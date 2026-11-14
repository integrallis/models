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

import com.integrallis.models.backend.purejava.SequenceEncoder;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Objects;

/** Whole-sequence, bidirectional BERT encoder over mapped GGUF weights. */
public final class BertForwardPass implements SequenceEncoder {

  private final BertConfig config;
  private final BertWeights weights;
  private final float[] tokenEmbedding;
  private final float[] positionEmbedding;
  private final float[][] tokenTypeEmbeddings;
  private final double[] pooledSum;

  private float[] hidden = new float[0];
  private float[] normalized = new float[0];
  private float[] queries = new float[0];
  private float[] keys = new float[0];
  private float[] values = new float[0];
  private float[] queryKeyValues = new float[0];
  private float[] attention = new float[0];
  private float[] attentionProjected = new float[0];
  private float[] feedForward = new float[0];
  private float[] feedForwardGate = new float[0];
  private float[] feedForwardProjected = new float[0];
  private float[] scores = new float[0];
  private byte[] quantizedActivations = new byte[0];
  private float[] quantizedActivationScales = new float[0];
  private int[] quantizedActivationZeroPointCorrections = new int[0];
  private short[] quantizedActivationSums = new short[0];
  private float[] q4LaneScratch = new float[0];
  private float[] exactHiddenWeightRow = new float[0];
  private float[] exactFeedForwardWeightRow = new float[0];
  private int capacity;

  BertForwardPass(BertConfig config, BertWeights weights) {
    this(config, weights, GgufBatchedMatrixKernel.none());
  }

  BertForwardPass(BertConfig config, BertWeights weights, GgufBatchedMatrixKernel matrixKernel) {
    this.matrixKernel = Objects.requireNonNull(matrixKernel, "matrixKernel");
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    this.tokenEmbedding = new float[config.embeddingDim()];
    this.positionEmbedding = new float[config.embeddingDim()];
    this.tokenTypeEmbeddings = new float[weights.tokenTypeCount()][config.embeddingDim()];
    this.pooledSum = new double[config.embeddingDim()];
    for (int tokenType = 0; tokenType < tokenTypeEmbeddings.length; tokenType++) {
      weights.tokenTypeEmbedding(tokenType, tokenTypeEmbeddings[tokenType]);
    }
  }

  /** Loads a complete bidirectional encoder from an already parsed GGUF. */
  public static BertForwardPass fromGgufFile(GgufFile file, BertConfig config) {
    return fromGgufFile(file, config, GgufBatchedMatrixKernel.none());
  }

  /** Loads an encoder that projects through the injected kernel where it is eligible. */
  public static BertForwardPass fromGgufFile(
      GgufFile file, BertConfig config, GgufBatchedMatrixKernel matrixKernel) {
    return new BertForwardPass(config, BertWeights.load(file, config, matrixKernel), matrixKernel);
  }

  @Override
  public synchronized float[] encode(int[] tokens) {
    execute(tokens, null);
    return pool(tokens.length);
  }

  /** Encodes a sequence with one BERT segment id per token and applies the declared pooling. */
  public synchronized float[] encode(int[] tokens, int[] tokenTypes) {
    execute(tokens, Objects.requireNonNull(tokenTypes, "tokenTypes"));
    return pool(tokens.length);
  }

  /** Returns the final hidden state of the leading CLS token for a cross-encoder head. */
  public synchronized float[] encodeCls(int[] tokens, int[] tokenTypes) {
    execute(tokens, Objects.requireNonNull(tokenTypes, "tokenTypes"));
    return Arrays.copyOf(hidden, config.embeddingDim());
  }

  /**
   * Processes multiple tokenized sequences in one BERT forward pass.
   *
   * <p>Every inner array must have the same length; callers pad short sequences. The returned array
   * has one CLS vector per document.
   */
  public synchronized float[][] encodeBatchCls(int[][] tokens, int[][] tokenTypes) {
    return encodeBatchCls(tokens, tokenTypes, null);
  }

  /**
   * Processes multiple tokenized sequences in one BERT forward pass, with optional per-document
   * attention masking for padded positions.
   *
   * <p>When {@code lengths} is non-null, positions at and beyond {@code lengths[doc]} within each
   * document are masked in attention (their attention scores are set to a large negative so softmax
   * yields zero weight). Padding tokens still participate in projections and receive embeddings;
   * only attention is masked, matching standard BERT practice.
   */
  public synchronized float[][] encodeBatchCls(int[][] tokens, int[][] tokenTypes, int[] lengths) {
    Objects.requireNonNull(tokens, "tokens");
    Objects.requireNonNull(tokenTypes, "tokenTypes");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("batch must not be empty");
    }
    int batchSize = tokens.length;
    int positionsPerDoc = tokens[0].length;
    if (positionsPerDoc == 0) {
      throw new IllegalArgumentException("each sequence must have at least one token");
    }
    if (positionsPerDoc > config.contextLength()) {
      throw new IllegalArgumentException(
          "sequence length "
              + positionsPerDoc
              + " exceeds context length "
              + config.contextLength());
    }
    for (int i = 0; i < batchSize; i++) {
      if (tokens[i] == null || tokenTypes[i] == null) {
        throw new IllegalArgumentException("null sequence at index " + i);
      }
      if (tokens[i].length != positionsPerDoc || tokenTypes[i].length != positionsPerDoc) {
        throw new IllegalArgumentException(
            "all sequences must have the same length "
                + positionsPerDoc
                + "; index "
                + i
                + " differs");
      }
      if (lengths != null && (lengths[i] < 1 || lengths[i] > positionsPerDoc)) {
        throw new IllegalArgumentException(
            "lengths[" + i + "] must be in [1, " + positionsPerDoc + "]: " + lengths[i]);
      }
    }
    executeBatch(tokens, tokenTypes, batchSize, positionsPerDoc, lengths);
    int dim = config.embeddingDim();
    float[][] result = new float[batchSize][dim];
    for (int doc = 0; doc < batchSize; doc++) {
      System.arraycopy(hidden, doc * positionsPerDoc * dim, result[doc], 0, dim);
    }
    return result;
  }

  private void execute(int[] tokens, int[] tokenTypes) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (tokenTypes != null && tokenTypes.length != tokens.length) {
      throw new IllegalArgumentException(
          "tokenTypes length must match tokens: " + tokenTypes.length + " != " + tokens.length);
    }
    if (tokens.length > config.contextLength()) {
      throw new IllegalArgumentException(
          "sequence of "
              + tokens.length
              + " tokens exceeds the model's context length of "
              + config.contextLength());
    }

    int sequenceLength = tokens.length;
    int dim = config.embeddingDim();
    ensureCapacity(sequenceLength);
    for (int position = 0; position < sequenceLength; position++) {
      int tokenType = tokenTypes == null ? 0 : tokenTypes[position];
      if (tokenType < 0 || tokenType >= tokenTypeEmbeddings.length) {
        throw new IllegalArgumentException(
            "token type " + tokenType + " is outside the model's token type table");
      }
      weights.tokenEmbedding(tokens[position], tokenEmbedding);
      if (!config.usesRotaryPositions()) {
        weights.positionEmbedding(position, positionEmbedding);
      }
      int offset = position * dim;
      for (int index = 0; index < dim; index++) {
        normalized[offset + index] =
            tokenEmbedding[index]
                + (config.usesRotaryPositions() ? 0.0f : positionEmbedding[index])
                + tokenTypeEmbeddings[tokenType][index];
      }
      TensorOps.layerNorm(
          hidden,
          offset,
          normalized,
          offset,
          weights.inputNormWeight(),
          weights.inputNormBias(),
          dim,
          config.layerNormEps());
    }
    for (int layerIndex = 0; layerIndex < config.numLayers(); layerIndex++) {
      executeLayer(weights.layer(layerIndex), sequenceLength);
    }
  }

  private void executeBatch(
      int[][] tokens, int[][] tokenTypes, int batchSize, int positionsPerDoc, int[] lengths) {
    int dim = config.embeddingDim();
    int totalPositions = Math.multiplyExact(batchSize, positionsPerDoc);
    ensureCapacity(totalPositions);
    for (int doc = 0; doc < batchSize; doc++) {
      for (int position = 0; position < positionsPerDoc; position++) {
        int tokenType = tokenTypes[doc][position];
        if (tokenType < 0 || tokenType >= tokenTypeEmbeddings.length) {
          throw new IllegalArgumentException(
              "token type " + tokenType + " is outside the model's token type table");
        }
        weights.tokenEmbedding(tokens[doc][position], tokenEmbedding);
        if (!config.usesRotaryPositions()) {
          weights.positionEmbedding(position, positionEmbedding);
        }
        int offset = (doc * positionsPerDoc + position) * dim;
        for (int index = 0; index < dim; index++) {
          normalized[offset + index] =
              tokenEmbedding[index]
                  + (config.usesRotaryPositions() ? 0.0f : positionEmbedding[index])
                  + tokenTypeEmbeddings[tokenType][index];
        }
        TensorOps.layerNorm(
            hidden,
            offset,
            normalized,
            offset,
            weights.inputNormWeight(),
            weights.inputNormBias(),
            dim,
            config.layerNormEps());
      }
    }
    for (int layerIndex = 0; layerIndex < config.numLayers(); layerIndex++) {
      executeLayerBatched(weights.layer(layerIndex), batchSize, positionsPerDoc, lengths);
    }
  }

  private void executeLayer(BertWeights.Layer layer, int sequenceLength) {
    int dim = config.embeddingDim();
    int batchElements = sequenceLength * dim;
    if (layer.queryKeyValue() == null) {
      project(queries, hidden, sequenceLength, layer.query());
      project(keys, hidden, sequenceLength, layer.key());
      project(values, hidden, sequenceLength, layer.value());
      addBias(queries, sequenceLength, dim, layer.queryBias());
      addBias(keys, sequenceLength, dim, layer.keyBias());
      addBias(values, sequenceLength, dim, layer.valueBias());
    } else {
      project(queryKeyValues, hidden, sequenceLength, layer.queryKeyValue());
      splitQueryKeyValue(sequenceLength, dim);
      applyRotaryPositions(sequenceLength, dim);
    }

    attend(sequenceLength);
    project(attentionProjected, attention, sequenceLength, layer.attentionOutput());
    addBias(attentionProjected, sequenceLength, dim, layer.attentionOutputBias());
    for (int index = 0; index < batchElements; index++) {
      attentionProjected[index] += hidden[index];
    }
    normalizeRows(
        normalized,
        attentionProjected,
        sequenceLength,
        layer.attentionNormWeight(),
        layer.attentionNormBias());

    project(feedForward, normalized, sequenceLength, layer.feedForwardUp());
    addBias(feedForward, sequenceLength, config.hiddenDim(), layer.feedForwardUpBias());
    if (layer.feedForwardGate() == null) {
      TensorOps.geluErf(feedForward, 0, feedForward, 0, sequenceLength * config.hiddenDim());
    } else {
      project(feedForwardGate, normalized, sequenceLength, layer.feedForwardGate());
      TensorOps.swiGlu(
          feedForward, feedForwardGate, feedForward, sequenceLength * config.hiddenDim());
    }
    project(feedForwardProjected, feedForward, sequenceLength, layer.feedForwardDown());
    addBias(feedForwardProjected, sequenceLength, dim, layer.feedForwardDownBias());
    for (int index = 0; index < batchElements; index++) {
      feedForwardProjected[index] += normalized[index];
    }
    normalizeRows(
        hidden,
        feedForwardProjected,
        sequenceLength,
        layer.outputNormWeight(),
        layer.outputNormBias());
  }

  /**
   * Executes one BERT layer over a batch of {@code batchSize} documents each of length {@code
   * positionsPerDoc}. Total rows = {@code batchSize * positionsPerDoc}.
   */
  private void executeLayerBatched(
      BertWeights.Layer layer, int batchSize, int positionsPerDoc, int[] lengths) {
    int dim = config.embeddingDim();
    int totalPositions = Math.multiplyExact(batchSize, positionsPerDoc);
    int batchElements = totalPositions * dim;
    if (layer.queryKeyValue() == null) {
      project(queries, hidden, totalPositions, layer.query());
      project(keys, hidden, totalPositions, layer.key());
      project(values, hidden, totalPositions, layer.value());
      addBias(queries, totalPositions, dim, layer.queryBias());
      addBias(keys, totalPositions, dim, layer.keyBias());
      addBias(values, totalPositions, dim, layer.valueBias());
    } else {
      project(queryKeyValues, hidden, totalPositions, layer.queryKeyValue());
      splitQueryKeyValueBatched(batchSize, positionsPerDoc, dim);
      applyRotaryPositionsBatched(batchSize, positionsPerDoc, dim);
    }

    attendBatched(batchSize, positionsPerDoc, lengths);
    project(attentionProjected, attention, totalPositions, layer.attentionOutput());
    addBias(attentionProjected, totalPositions, dim, layer.attentionOutputBias());
    for (int index = 0; index < batchElements; index++) {
      attentionProjected[index] += hidden[index];
    }
    normalizeRows(
        normalized,
        attentionProjected,
        totalPositions,
        layer.attentionNormWeight(),
        layer.attentionNormBias());

    project(feedForward, normalized, totalPositions, layer.feedForwardUp());
    addBias(feedForward, totalPositions, config.hiddenDim(), layer.feedForwardUpBias());
    if (layer.feedForwardGate() == null) {
      TensorOps.geluErf(feedForward, 0, feedForward, 0, totalPositions * config.hiddenDim());
    } else {
      project(feedForwardGate, normalized, totalPositions, layer.feedForwardGate());
      TensorOps.swiGlu(
          feedForward, feedForwardGate, feedForward, totalPositions * config.hiddenDim());
    }
    project(feedForwardProjected, feedForward, totalPositions, layer.feedForwardDown());
    addBias(feedForwardProjected, totalPositions, dim, layer.feedForwardDownBias());
    for (int index = 0; index < batchElements; index++) {
      feedForwardProjected[index] += normalized[index];
    }
    normalizeRows(
        hidden,
        feedForwardProjected,
        totalPositions,
        layer.outputNormWeight(),
        layer.outputNormBias());
  }

  private void attendBatched(int batchSize, int positionsPerDoc, int[] lengths) {
    int dim = config.embeddingDim();
    int headDim = config.headDim();
    float scale = (float) (1.0 / Math.sqrt(headDim));
    int totalPositions = Math.multiplyExact(batchSize, positionsPerDoc);
    Arrays.fill(attention, 0, totalPositions * dim, 0.0f);
    for (int doc = 0; doc < batchSize; doc++) {
      int docBase = doc * positionsPerDoc;
      int actualLength = lengths == null ? positionsPerDoc : lengths[doc];
      for (int position = 0; position < positionsPerDoc; position++) {
        int globalQueryOffset = (docBase + position) * dim;
        int scoreOffset = position * positionsPerDoc;
        for (int head = 0; head < config.numHeads(); head++) {
          int headOffset = head * headDim;
          VectorUtil.batchDotProductExact(
              queries,
              globalQueryOffset + headOffset,
              keys,
              docBase * dim + headOffset,
              dim,
              positionsPerDoc,
              headDim,
              scores,
              scoreOffset);
          for (int keyPosition = 0; keyPosition < positionsPerDoc; keyPosition++) {
            scores[scoreOffset + keyPosition] *= scale;
            if (keyPosition >= actualLength || position >= actualLength) {
              scores[scoreOffset + keyPosition] = Float.NEGATIVE_INFINITY;
            }
          }
          TensorOps.softmax(scores, scoreOffset, positionsPerDoc);
          VectorUtil.addWeightedRowsInPlace(
              attention,
              globalQueryOffset + headOffset,
              values,
              docBase * dim + headOffset,
              dim,
              scores,
              scoreOffset,
              positionsPerDoc,
              headDim);
        }
      }
    }
  }

  private void splitQueryKeyValueBatched(int batchSize, int positionsPerDoc, int dim) {
    int totalPositions = batchSize * positionsPerDoc;
    for (int position = 0; position < totalPositions; position++) {
      int source = position * dim * 3;
      int target = position * dim;
      System.arraycopy(queryKeyValues, source, queries, target, dim);
      System.arraycopy(queryKeyValues, source + dim, keys, target, dim);
      System.arraycopy(queryKeyValues, source + dim * 2, values, target, dim);
    }
  }

  private void applyRotaryPositionsBatched(int batchSize, int positionsPerDoc, int dim) {
    int headDim = config.headDim();
    for (int doc = 0; doc < batchSize; doc++) {
      int docBase = doc * positionsPerDoc;
      for (int position = 0; position < positionsPerDoc; position++) {
        for (int head = 0; head < config.numHeads(); head++) {
          int offset = (docBase + position) * dim + head * headDim;
          TensorOps.ropeNeox(queries, offset, position, headDim, config.ropeTheta());
          TensorOps.ropeNeox(keys, offset, position, headDim, config.ropeTheta());
        }
      }
    }
  }

  private void attend(int sequenceLength) {
    int dim = config.embeddingDim();
    int headDim = config.headDim();
    float scale = (float) (1.0 / Math.sqrt(headDim));
    Arrays.fill(attention, 0, sequenceLength * dim, 0.0f);
    for (int position = 0; position < sequenceLength; position++) {
      int scoreOffset = position * sequenceLength;
      for (int head = 0; head < config.numHeads(); head++) {
        int headOffset = head * headDim;
        VectorUtil.batchDotProductExact(
            queries,
            position * dim + headOffset,
            keys,
            headOffset,
            dim,
            sequenceLength,
            headDim,
            scores,
            scoreOffset);
        for (int keyPosition = 0; keyPosition < sequenceLength; keyPosition++) {
          scores[scoreOffset + keyPosition] *= scale;
        }
        TensorOps.softmax(scores, scoreOffset, sequenceLength);
        VectorUtil.addWeightedRowsInPlace(
            attention,
            position * dim + headOffset,
            values,
            headOffset,
            dim,
            scores,
            scoreOffset,
            sequenceLength,
            headDim);
      }
    }
  }

  private void splitQueryKeyValue(int sequenceLength, int dim) {
    for (int position = 0; position < sequenceLength; position++) {
      int source = position * dim * 3;
      int target = position * dim;
      System.arraycopy(queryKeyValues, source, queries, target, dim);
      System.arraycopy(queryKeyValues, source + dim, keys, target, dim);
      System.arraycopy(queryKeyValues, source + dim * 2, values, target, dim);
    }
  }

  private void applyRotaryPositions(int sequenceLength, int dim) {
    int headDim = config.headDim();
    for (int position = 0; position < sequenceLength; position++) {
      for (int head = 0; head < config.numHeads(); head++) {
        int offset = position * dim + head * headDim;
        // Nomic-BERT follows the same split-half (NeoX) RoPE layout as llama.cpp's
        // LLM_ARCH_NOMIC_BERT, not the adjacent-pair layout used by Llama.
        TensorOps.ropeNeox(queries, offset, position, headDim, config.ropeTheta());
        TensorOps.ropeNeox(keys, offset, position, headDim, config.ropeTheta());
      }
    }
  }

  private void normalizeRows(
      float[] output, float[] input, int sequenceLength, float[] normWeight, float[] normBias) {
    int dim = config.embeddingDim();
    for (int position = 0; position < sequenceLength; position++) {
      int offset = position * dim;
      TensorOps.layerNorm(
          output, offset, input, offset, normWeight, normBias, dim, config.layerNormEps());
    }
  }

  /**
   * The injected batched matrix kernel, or {@link GgufBatchedMatrixKernel#none()}.
   *
   * <p>This pass had no kernel seam: projections went straight to {@code TensorOps} and the kernel
   * the caller injected was never consulted, so it ran unaccelerated while diagnostics reported one
   * was loaded. Found by audit on 2026-11-01 after the same gap cost {@code lfm2moe} 166% of its
   * decode throughput -- 8.94 to 23.80 tok/s, our own arm before and after on one host. (The gate
   * ratios first reported beside those figures were withdrawn: the comparator arm carried no
   * artifact SHA-256 and the policy had excluded it. The admitted ollama comparison of 2026-11-01
   * is decodeThroughputRatio 0.3974.) No qualified catalog entry used this pass, so no published
   * number moved; the cost was that a future run here would read "the model is slow".
   */
  private final GgufBatchedMatrixKernel matrixKernel;

  private void project(
      float[] output, float[] input, int sequenceLength, BertWeights.Matrix matrix) {
    if (matrix.type() == GgufTensorType.Q5_1) {
      BertQ51Projection.multiply(
          output,
          input,
          matrix.data(),
          sequenceLength,
          matrix.rows(),
          matrix.columns(),
          matrix.columns() == config.hiddenDim()
              ? exactFeedForwardWeightRow
              : exactHiddenWeightRow);
      return;
    }
    if (config.usesRotaryPositions()
        || matrix.type() == GgufTensorType.F32
        || matrix.type() == GgufTensorType.Q4_1) {
      TensorOps.ggufExactBatchedMatmul(
          output,
          input,
          matrix.data(),
          matrix.type(),
          sequenceLength,
          matrix.rows(),
          matrix.columns(),
          matrix.columns() == config.hiddenDim()
              ? exactFeedForwardWeightRow
              : exactHiddenWeightRow);
      return;
    }
    // The kernel only on the GENERAL path. The Q5_1 and rotary/Q4_1 branches above are
    // deliberately exact -- BertQ51Projection and ggufExactBatchedMatmul keep F32 activations --
    // and routing them through a quantising kernel would change embeddings those models were
    // qualified on.
    if (matrixKernel.isEligible(matrix.type(), sequenceLength, matrix.rows(), matrix.columns())) {
      matrixKernel.multiply(
          output,
          input,
          matrix.data(),
          matrix.type(),
          sequenceLength,
          matrix.rows(),
          matrix.columns());
      return;
    }
    TensorOps.ggufBatchedMatmul(
        output,
        input,
        matrix.data(),
        matrix.type(),
        sequenceLength,
        matrix.rows(),
        matrix.columns(),
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        GgufQ4Kernel.WIDENED);
  }

  private static void addBias(float[] values, int rows, int columns, float[] bias) {
    for (int row = 0; row < rows; row++) {
      int offset = row * columns;
      for (int column = 0; column < columns; column++) {
        values[offset + column] += bias[column];
      }
    }
  }

  private float[] pool(int sequenceLength) {
    int dim = config.embeddingDim();
    float[] pooled = new float[dim];
    if (config.pooling() == BertConfig.Pooling.CLS || config.pooling() == BertConfig.Pooling.RANK) {
      System.arraycopy(hidden, 0, pooled, 0, dim);
      return pooled;
    }
    Arrays.fill(pooledSum, 0.0);
    for (int position = 0; position < sequenceLength; position++) {
      int offset = position * dim;
      for (int index = 0; index < dim; index++) {
        pooledSum[index] += hidden[offset + index];
      }
    }
    for (int index = 0; index < dim; index++) {
      pooled[index] = (float) (pooledSum[index] / sequenceLength);
    }
    return pooled;
  }

  private void ensureCapacity(int sequenceLength) {
    if (capacity >= sequenceLength) {
      return;
    }
    int dim = config.embeddingDim();
    int hiddenDim = config.hiddenDim();
    int stateElements = Math.multiplyExact(sequenceLength, dim);
    hidden = new float[stateElements];
    normalized = new float[stateElements];
    queries = new float[stateElements];
    keys = new float[stateElements];
    values = new float[stateElements];
    queryKeyValues = new float[Math.multiplyExact(stateElements, 3)];
    attention = new float[stateElements];
    attentionProjected = new float[stateElements];
    feedForward = new float[Math.multiplyExact(sequenceLength, hiddenDim)];
    feedForwardGate = new float[Math.multiplyExact(sequenceLength, hiddenDim)];
    feedForwardProjected = new float[stateElements];
    scores = new float[Math.multiplyExact(sequenceLength, sequenceLength)];

    int widestActivation = Math.max(dim, hiddenDim);
    quantizedActivations = new byte[Math.multiplyExact(sequenceLength, widestActivation)];
    quantizedActivationScales =
        new float[Math.multiplyExact(sequenceLength, (widestActivation + 31) / 32)];
    quantizedActivationZeroPointCorrections =
        new int[Math.multiplyExact(sequenceLength, (widestActivation + 3) / 4)];
    quantizedActivationSums =
        new short[Math.multiplyExact(sequenceLength, (widestActivation + 15) / 16)];
    int q4Rows = weights.maxQ40ProjectionRows();
    q4LaneScratch =
        q4Rows == 0
            ? new float[0]
            : new float[Math.multiplyExact(Math.multiplyExact(sequenceLength, q4Rows), 8)];
    exactHiddenWeightRow = new float[dim];
    exactFeedForwardWeightRow = new float[hiddenDim];
    capacity = sequenceLength;
  }
}
