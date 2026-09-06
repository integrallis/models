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
package com.integrallis.models.backend.purejava.deberta;

import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/** Whole-sequence DeBERTa-v2 encoder with disentangled relative attention. */
final class DebertaV2ForwardPass {

  static final String PARALLELISM_PROPERTY = "models.deberta.threads";

  private record RelativeProjections(float[] queries, float[] keys) {}

  private record Worker(float[] scores) {}

  private final DebertaV2Config config;
  private final DebertaV2Weights weights;
  private final RelativeProjections[] relativeProjections;
  private final float[] tokenEmbedding;
  private final float[] pooled;
  private final Worker[] workers;

  private float[] hidden = new float[0];
  private float[] queries = new float[0];
  private float[] keys = new float[0];
  private float[] values = new float[0];
  private float[] attention = new float[0];
  private float[] attentionProjected = new float[0];
  private float[] normalized = new float[0];
  private float[] feedForward = new float[0];
  private float[] feedForwardProjected = new float[0];
  private int capacity;

  DebertaV2ForwardPass(DebertaV2Config config, DebertaV2Weights weights) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    this.tokenEmbedding = new float[config.hiddenSize()];
    this.pooled = new float[config.poolerHiddenSize()];
    this.workers = new Worker[resolveParallelism()];
    for (int worker = 0; worker < workers.length; worker++) {
      workers[worker] = new Worker(new float[config.maxPositions()]);
    }
    this.relativeProjections = prepareRelativeProjections();
  }

  synchronized double score(int[] tokens) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0 || tokens.length > config.maxPositions()) {
      throw new IllegalArgumentException(
          "token count must be between 1 and " + config.maxPositions() + ": " + tokens.length);
    }
    execute(tokens);
    projectSingle(pooled, hidden, 0, weights.pooler());
    addBias(pooled, 1, pooled.length, weights.poolerBias());
    TensorOps.geluErf(pooled, 0, pooled, 0, pooled.length);
    return VectorUtil.dotProduct(pooled, weights.classifierWeight()) + weights.classifierBias()[0];
  }

  static int relativeBucket(int relativePosition, int bucketSize, int maxPosition) {
    if (bucketSize <= 0 || maxPosition <= bucketSize / 2) {
      throw new IllegalArgumentException("invalid relative-position bucket geometry");
    }
    int sign = Integer.signum(relativePosition);
    int absolute = Math.abs(relativePosition);
    int midpoint = bucketSize / 2;
    if (absolute <= midpoint) {
      return relativePosition;
    }
    double logarithmic =
        Math.ceil(
                Math.log(absolute / (double) midpoint)
                    / Math.log((maxPosition - 1.0) / midpoint)
                    * (midpoint - 1))
            + midpoint;
    return sign * Math.min(bucketSize - 1, (int) logarithmic);
  }

  private void execute(int[] tokens) {
    int sequenceLength = tokens.length;
    int dim = config.hiddenSize();
    ensureCapacity(sequenceLength);
    for (int position = 0; position < sequenceLength; position++) {
      weights.wordEmbedding(tokens[position], tokenEmbedding);
      TensorOps.layerNorm(
          hidden,
          position * dim,
          tokenEmbedding,
          0,
          weights.embeddingNormWeight(),
          weights.embeddingNormBias(),
          dim,
          config.layerNormEpsilon());
    }
    for (int layer = 0; layer < config.numLayers(); layer++) {
      executeLayer(layer, sequenceLength);
    }
  }

  private void executeLayer(int layerIndex, int sequenceLength) {
    DebertaV2Weights.Layer layer = weights.layer(layerIndex);
    int dim = config.hiddenSize();
    int elements = sequenceLength * dim;
    project(queries, hidden, sequenceLength, layer.query());
    project(keys, hidden, sequenceLength, layer.key());
    project(values, hidden, sequenceLength, layer.value());
    addBias(queries, sequenceLength, dim, layer.queryBias());
    addBias(keys, sequenceLength, dim, layer.keyBias());
    addBias(values, sequenceLength, dim, layer.valueBias());

    attend(sequenceLength, relativeProjections[layerIndex]);
    project(attentionProjected, attention, sequenceLength, layer.attentionOutput());
    addBias(attentionProjected, sequenceLength, dim, layer.attentionOutputBias());
    for (int index = 0; index < elements; index++) {
      attentionProjected[index] += hidden[index];
    }
    normalizeRows(
        normalized,
        attentionProjected,
        sequenceLength,
        layer.attentionNormWeight(),
        layer.attentionNormBias());

    project(feedForward, normalized, sequenceLength, layer.feedForwardUp());
    addBias(feedForward, sequenceLength, config.intermediateSize(), layer.feedForwardUpBias());
    TensorOps.geluErf(feedForward, 0, feedForward, 0, sequenceLength * config.intermediateSize());
    project(feedForwardProjected, feedForward, sequenceLength, layer.feedForwardDown());
    addBias(feedForwardProjected, sequenceLength, dim, layer.feedForwardDownBias());
    for (int index = 0; index < elements; index++) {
      feedForwardProjected[index] += normalized[index];
    }
    normalizeRows(
        hidden,
        feedForwardProjected,
        sequenceLength,
        layer.outputNormWeight(),
        layer.outputNormBias());
  }

  private void attend(int sequenceLength, RelativeProjections relative) {
    int dim = config.hiddenSize();
    int headSize = config.headSize();
    int span = config.positionBuckets();
    float scale = (float) Math.sqrt(headSize * 3.0);
    Arrays.fill(attention, 0, sequenceLength * dim, 0.0f);
    forEachRow(
        sequenceLength,
        (queryPosition, worker) -> {
          float[] scores = worker.scores();
          int queryOffset = queryPosition * dim;
          for (int head = 0; head < config.numHeads(); head++) {
            int headOffset = head * headSize;
            VectorUtil.batchDotProductExact(
                queries,
                queryOffset + headOffset,
                keys,
                headOffset,
                dim,
                sequenceLength,
                headSize,
                scores,
                0);
            for (int keyPosition = 0; keyPosition < sequenceLength; keyPosition++) {
              int bucket =
                  relativeBucket(
                      queryPosition - keyPosition,
                      config.positionBuckets(),
                      config.maxRelativePositions());
              int relativeOffset = (Math.max(0, Math.min(span * 2 - 1, bucket + span))) * dim;
              float contentToPosition =
                  VectorUtil.dotProduct(
                      queries,
                      queryOffset + headOffset,
                      relative.keys(),
                      relativeOffset + headOffset,
                      headSize);
              float positionToContent =
                  VectorUtil.dotProduct(
                      keys,
                      keyPosition * dim + headOffset,
                      relative.queries(),
                      relativeOffset + headOffset,
                      headSize);
              scores[keyPosition] =
                  (scores[keyPosition] + contentToPosition + positionToContent) / scale;
            }
            TensorOps.softmax(scores, 0, sequenceLength);
            VectorUtil.addWeightedRowsInPlace(
                attention,
                queryOffset + headOffset,
                values,
                headOffset,
                dim,
                scores,
                0,
                sequenceLength,
                headSize);
          }
        });
  }

  private RelativeProjections[] prepareRelativeProjections() {
    int rows = weights.relativeEmbeddings().rows();
    int dim = config.hiddenSize();
    float[] relative = weights.relativeEmbeddings().values().clone();
    for (int row = 0; row < rows; row++) {
      int offset = row * dim;
      TensorOps.layerNorm(
          relative,
          offset,
          relative,
          offset,
          weights.relativeNormWeight(),
          weights.relativeNormBias(),
          dim,
          config.layerNormEpsilon());
    }
    RelativeProjections[] prepared = new RelativeProjections[config.numLayers()];
    for (int layerIndex = 0; layerIndex < prepared.length; layerIndex++) {
      DebertaV2Weights.Layer layer = weights.layer(layerIndex);
      float[] relativeQueries = new float[rows * dim];
      float[] relativeKeys = new float[rows * dim];
      project(relativeQueries, relative, rows, layer.query());
      project(relativeKeys, relative, rows, layer.key());
      addBias(relativeQueries, rows, dim, layer.queryBias());
      addBias(relativeKeys, rows, dim, layer.keyBias());
      prepared[layerIndex] = new RelativeProjections(relativeQueries, relativeKeys);
    }
    return prepared;
  }

  private void normalizeRows(
      float[] output, float[] input, int rows, float[] normWeight, float[] normBias) {
    int dim = config.hiddenSize();
    for (int row = 0; row < rows; row++) {
      int offset = row * dim;
      TensorOps.layerNorm(
          output, offset, input, offset, normWeight, normBias, dim, config.layerNormEpsilon());
    }
  }

  private void project(float[] output, float[] input, int rows, DebertaV2Weights.Matrix matrix) {
    forEachRow(
        rows,
        (row, worker) ->
            projectSingle(output, row * matrix.rows(), input, row * matrix.columns(), matrix));
  }

  private static void projectSingle(
      float[] output, float[] input, int inputOffset, DebertaV2Weights.Matrix matrix) {
    projectSingle(output, 0, input, inputOffset, matrix);
  }

  private static void projectSingle(
      float[] output,
      int outputOffset,
      float[] input,
      int inputOffset,
      DebertaV2Weights.Matrix matrix) {
    VectorUtil.batchDotProductExact(
        input,
        inputOffset,
        matrix.values(),
        0,
        matrix.columns(),
        matrix.rows(),
        matrix.columns(),
        output,
        outputOffset);
  }

  private static void addBias(float[] output, int rows, int columns, float[] bias) {
    for (int row = 0; row < rows; row++) {
      int offset = row * columns;
      for (int column = 0; column < columns; column++) {
        output[offset + column] += bias[column];
      }
    }
  }

  private void ensureCapacity(int sequenceLength) {
    if (capacity >= sequenceLength) {
      return;
    }
    int dim = config.hiddenSize();
    int stateElements = Math.multiplyExact(sequenceLength, dim);
    hidden = new float[stateElements];
    queries = new float[stateElements];
    keys = new float[stateElements];
    values = new float[stateElements];
    attention = new float[stateElements];
    attentionProjected = new float[stateElements];
    normalized = new float[stateElements];
    feedForward = new float[Math.multiplyExact(sequenceLength, config.intermediateSize())];
    feedForwardProjected = new float[stateElements];
    capacity = sequenceLength;
  }

  private static int resolveParallelism() {
    String configured = System.getProperty(PARALLELISM_PROPERTY);
    if (configured != null && !configured.isBlank()) {
      int requested = Integer.parseInt(configured.trim());
      if (requested < 1) {
        throw new IllegalArgumentException(PARALLELISM_PROPERTY + " must be >= 1: " + requested);
      }
      return requested;
    }
    // Leave two processors for the host once enough are present. The controlled x86_64 sweep
    // found that saturating every logical processor reduced both pair and batch throughput.
    int processors = Math.max(1, Runtime.getRuntime().availableProcessors());
    return processors > 4 ? processors - 2 : processors;
  }

  private void forEachRow(int rows, RowWork body) {
    int parallelism = Math.min(workers.length, rows);
    if (parallelism == 1) {
      for (int row = 0; row < rows; row++) {
        body.run(row, workers[0]);
      }
      return;
    }
    Thread[] threads = new Thread[parallelism];
    AtomicReference<RuntimeException> failure = new AtomicReference<>();
    for (int worker = 0; worker < parallelism; worker++) {
      int index = worker;
      threads[worker] =
          Thread.ofVirtual()
              .start(
                  () -> {
                    try {
                      for (int row = index; row < rows; row += parallelism) {
                        body.run(row, workers[index]);
                      }
                    } catch (RuntimeException thrown) {
                      failure.compareAndSet(null, thrown);
                    }
                  });
    }
    for (Thread thread : threads) {
      try {
        thread.join();
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("interrupted while reranking", interrupted);
      }
    }
    RuntimeException thrown = failure.get();
    if (thrown != null) {
      throw thrown;
    }
  }

  @FunctionalInterface
  private interface RowWork {
    void run(int row, Worker worker);
  }
}
