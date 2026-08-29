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
package com.integrallis.models.accelerator;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Java-authored fixed-shape grouped-query attention experiment with a device-resident KV cache. */
final class CausalAttentionKernel {
  private static final int START_POSITION_INDEX = 0;
  private static final int ACTUAL_BATCH_INDEX = 1;

  private CausalAttentionKernel() {}

  /** Stores the current prompt chunk in a persistent, position-major KV cache. */
  static void store(
      IntArray state,
      FloatArray key,
      FloatArray value,
      FloatArray keyCache,
      FloatArray valueCache,
      int keyDim,
      int valueDim) {
    int executionBatchSize = key.getSize() / keyDim;
    int entriesPerBatch = keyDim + valueDim;
    for (@Parallel int entry = 0; entry < executionBatchSize * entriesPerBatch; entry++) {
      int batch = entry / entriesPerBatch;
      int component = entry - batch * entriesPerBatch;
      if (batch < state.get(ACTUAL_BATCH_INDEX)) {
        int position = state.get(START_POSITION_INDEX) + batch;
        if (component < keyDim) {
          keyCache.set(position * keyDim + component, key.get(batch * keyDim + component));
        } else {
          int valueComponent = component - keyDim;
          valueCache.set(
              position * valueDim + valueComponent, value.get(batch * valueDim + valueComponent));
        }
      }
    }
  }

  /** Computes causal grouped-query attention with one independent work item per token/head pair. */
  static void attend(
      IntArray state,
      FloatArray query,
      FloatArray keyCache,
      FloatArray valueCache,
      FloatArray scores,
      FloatArray output,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int keyDim,
      int valueDim,
      int maxSequenceLength,
      int slidingWindow) {
    int queryDim = numHeads * keyLength;
    int outputDim = numHeads * valueLength;
    int executionBatchSize = query.getSize() / queryDim;
    int groupSize = numHeads / numKvHeads;
    float scale = 1.0f / TornadoMath.sqrt(keyLength);
    for (@Parallel int task = 0; task < executionBatchSize * numHeads; task++) {
      int batch = task / numHeads;
      int head = task - batch * numHeads;
      int outputOffset = batch * outputDim + head * valueLength;
      if (batch < state.get(ACTUAL_BATCH_INDEX)) {
        int position = state.get(START_POSITION_INDEX) + batch;
        int firstPosition = slidingWindow > 0 ? Math.max(0, position - slidingWindow + 1) : 0;
        int kvHead = head / groupSize;
        int queryOffset = batch * queryDim + head * keyLength;
        int scoreOffset = task * maxSequenceLength;
        float maximum = Float.NEGATIVE_INFINITY;
        for (int cachedPosition = firstPosition; cachedPosition <= position; cachedPosition++) {
          int keyOffset = cachedPosition * keyDim + kvHead * keyLength;
          float score = 0.0f;
          for (int column = 0; column < keyLength; column++) {
            score += query.get(queryOffset + column) * keyCache.get(keyOffset + column);
          }
          score *= scale;
          scores.set(scoreOffset + cachedPosition, score);
          maximum = Math.max(maximum, score);
        }

        float denominator = 0.0f;
        for (int cachedPosition = firstPosition; cachedPosition <= position; cachedPosition++) {
          float probability = TornadoMath.exp(scores.get(scoreOffset + cachedPosition) - maximum);
          scores.set(scoreOffset + cachedPosition, probability);
          denominator += probability;
        }

        for (int column = 0; column < valueLength; column++) {
          float weightedValue = 0.0f;
          for (int cachedPosition = firstPosition; cachedPosition <= position; cachedPosition++) {
            int valueOffset = cachedPosition * valueDim + kvHead * valueLength;
            weightedValue +=
                scores.get(scoreOffset + cachedPosition) * valueCache.get(valueOffset + column);
          }
          output.set(outputOffset + column, weightedValue / denominator);
        }
      } else {
        for (int column = 0; column < valueLength; column++) {
          output.set(outputOffset + column, 0.0f);
        }
      }
    }
  }

  /**
   * Workgroup-tiled attention adapted from the local GPULlama3 reference implementation.
   *
   * <p>One workgroup owns a token/head pair. K/V rows are staged through local memory and softmax
   * is accumulated online across 16-position tiles.
   */
  static void attendTiled(
      KernelContext context,
      IntArray state,
      FloatArray query,
      FloatArray keyCache,
      FloatArray valueCache,
      FloatArray output,
      int numHeads,
      int keyLength,
      int keyDim,
      int groupSize,
      int queryDim) {
    int localThread = context.localIdx;
    int localSize = context.localGroupSizeX;
    int task = context.groupIdx;
    int batch = task / numHeads;
    int head = task - batch * numHeads;
    int outputOffset = batch * queryDim + head * keyLength;
    int tileSize = 16;

    float[] sharedQuery = context.allocateFloatLocalArray(keyLength);
    float[] keyTile = context.allocateFloatLocalArray(tileSize * keyLength);
    float[] valueTile = context.allocateFloatLocalArray(tileSize * keyLength);
    float[] scoreTile = context.allocateFloatLocalArray(tileSize);
    float[] maximumHolder = context.allocateFloatLocalArray(1);

    if (batch < state.get(ACTUAL_BATCH_INDEX)) {
      int position = state.get(START_POSITION_INDEX) + batch;
      int kvHead = head / groupSize;
      int queryOffset = batch * queryDim + head * keyLength;
      for (int column = localThread; column < keyLength; column += localSize) {
        sharedQuery[column] = query.get(queryOffset + column);
      }
      context.localBarrier();

      float maximum = Float.NEGATIVE_INFINITY;
      float denominator = 0.0f;
      float[] accumulator = new float[keyLength];
      for (int column = 0; column < keyLength; column++) {
        accumulator[column] = 0.0f;
      }

      for (int tileStart = 0; tileStart <= position; tileStart += tileSize) {
        int tileEnd = Math.min(tileStart + tileSize - 1, position);
        for (int cachedPosition = tileStart + localThread;
            cachedPosition <= tileEnd;
            cachedPosition += localSize) {
          int tilePosition = cachedPosition - tileStart;
          int tileOffset = tilePosition * keyLength;
          int cacheOffset = cachedPosition * keyDim + kvHead * keyLength;
          for (int column = 0; column < keyLength; column++) {
            keyTile[tileOffset + column] = keyCache.get(cacheOffset + column);
            valueTile[tileOffset + column] = valueCache.get(cacheOffset + column);
          }
        }
        context.localBarrier();

        for (int cachedPosition = tileStart + localThread;
            cachedPosition <= tileEnd;
            cachedPosition += localSize) {
          int tilePosition = cachedPosition - tileStart;
          float score = 0.0f;
          for (int column = 0; column < keyLength; column++) {
            score += sharedQuery[column] * keyTile[tilePosition * keyLength + column];
          }
          scoreTile[tilePosition] = score / TornadoMath.sqrt(keyLength);
        }
        context.localBarrier();

        float tileMaximum = Float.NEGATIVE_INFINITY;
        for (int tilePosition = 0; tilePosition <= tileEnd - tileStart; tilePosition++) {
          tileMaximum = Math.max(tileMaximum, scoreTile[tilePosition]);
        }
        if (localThread == 0) {
          maximumHolder[0] = tileMaximum;
        }
        context.localBarrier();

        float newMaximum = Math.max(maximum, maximumHolder[0]);
        if (maximum != Float.NEGATIVE_INFINITY) {
          float previousScale = TornadoMath.exp(maximum - newMaximum);
          denominator *= previousScale;
          for (int column = 0; column < keyLength; column++) {
            accumulator[column] *= previousScale;
          }
        }
        maximum = newMaximum;
        for (int tilePosition = 0; tilePosition <= tileEnd - tileStart; tilePosition++) {
          float probability = TornadoMath.exp(scoreTile[tilePosition] - maximum);
          denominator += probability;
          for (int column = 0; column < keyLength; column++) {
            accumulator[column] += probability * valueTile[tilePosition * keyLength + column];
          }
        }
        context.localBarrier();
      }

      float inverseDenominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;
      for (int column = localThread; column < keyLength; column += localSize) {
        output.set(outputOffset + column, accumulator[column] * inverseDenominator);
      }
    } else {
      for (int column = localThread; column < keyLength; column += localSize) {
        output.set(outputOffset + column, 0.0f);
      }
    }
  }
}
