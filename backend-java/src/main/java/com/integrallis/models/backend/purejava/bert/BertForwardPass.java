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
import com.integrallis.models.backend.purejava.ops.TensorOps;
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
  private float[] attention = new float[0];
  private float[] attentionProjected = new float[0];
  private float[] feedForward = new float[0];
  private float[] feedForwardProjected = new float[0];
  private float[] scores = new float[0];
  private byte[] quantizedActivations = new byte[0];
  private float[] quantizedActivationScales = new float[0];
  private int[] quantizedActivationZeroPointCorrections = new int[0];
  private short[] quantizedActivationSums = new short[0];
  private float[] q4LaneScratch = new float[0];
  private int capacity;

  BertForwardPass(BertConfig config, BertWeights weights) {
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
    return new BertForwardPass(config, BertWeights.load(file, config));
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
      weights.positionEmbedding(position, positionEmbedding);
      int offset = position * dim;
      for (int index = 0; index < dim; index++) {
        normalized[offset + index] =
            tokenEmbedding[index]
                + positionEmbedding[index]
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

  private void executeLayer(BertWeights.Layer layer, int sequenceLength) {
    int dim = config.embeddingDim();
    int batchElements = sequenceLength * dim;
    project(queries, hidden, sequenceLength, layer.query());
    project(keys, hidden, sequenceLength, layer.key());
    project(values, hidden, sequenceLength, layer.value());
    addBias(queries, sequenceLength, dim, layer.queryBias());
    addBias(keys, sequenceLength, dim, layer.keyBias());
    addBias(values, sequenceLength, dim, layer.valueBias());

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
    TensorOps.geluErf(feedForward, 0, feedForward, 0, sequenceLength * config.hiddenDim());
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

  private void normalizeRows(
      float[] output, float[] input, int sequenceLength, float[] normWeight, float[] normBias) {
    int dim = config.embeddingDim();
    for (int position = 0; position < sequenceLength; position++) {
      int offset = position * dim;
      TensorOps.layerNorm(
          output, offset, input, offset, normWeight, normBias, dim, config.layerNormEps());
    }
  }

  private void project(
      float[] output, float[] input, int sequenceLength, BertWeights.Matrix matrix) {
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
    if (config.pooling() == BertConfig.Pooling.CLS) {
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
    attention = new float[stateElements];
    attentionProjected = new float[stateElements];
    feedForward = new float[Math.multiplyExact(sequenceLength, hiddenDim)];
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
    capacity = sequenceLength;
  }
}
