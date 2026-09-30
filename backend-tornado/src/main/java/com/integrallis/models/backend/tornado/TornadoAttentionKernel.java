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
package com.integrallis.models.backend.tornado;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Java-authored TornadoVM kernel for causal grouped-query attention over a device-resident KV
 * mirror.
 *
 * <p>The arithmetic deliberately mirrors the Java path's head-by-head loop ({@code
 * LlamaForwardPass.attendGroupHeadByHead}) operation for operation: score every visible cached key,
 * shift by the row maximum, exponentiate, normalise the weights <em>before</em> accumulating
 * values, then accumulate value rows in position order. Normalising first rather than dividing the
 * accumulated output at the end is not a cosmetic difference — it is what the Java path does
 * ({@code TensorOps.softmax} scales the score row in place), and it is what lets the off-device
 * parity test hold this kernel to bit-exact agreement with a scalar reference instead of a
 * tolerance.
 *
 * <p>Only {@code @Parallel} loops are used, and no {@code KernelContext}. That costs the workgroup
 * tiling a hand-scheduled kernel could use, and it buys something worth more at this stage: these
 * exact methods execute as ordinary sequential Java when they are called directly, so the whole
 * kernel is tested on every CI host and only PTX code generation is left to the GPU gate. A tiled
 * variant is a later A/B against this one, on a device, not a replacement for it.
 *
 * <p>{@code state} carries the two values that change between executions of one compiled plan:
 * {@code state[0]} is the sequence position of batch row zero and {@code state[1]} is how many of
 * the plan's fixed batch rows are real. Everything else — the shape and the attention scale — is a
 * task constant, because none of it varies within a loaded model.
 */
public final class TornadoAttentionKernel {

  /** Sequence position that staged key/value row zero belongs to. */
  public static final int STORE_POSITION_INDEX = 0;

  /** How many staged key/value rows to write into the mirror; zero writes none. */
  public static final int STORE_ROWS_INDEX = 1;

  /** Sequence position that query row zero attends from. */
  public static final int ATTEND_POSITION_INDEX = 2;

  /** How many query rows to attend; zero attends none and clears the whole output block. */
  public static final int ATTEND_ROWS_INDEX = 3;

  /** Number of entries in the execution state vector. */
  public static final int STATE_LENGTH = 4;

  private TornadoAttentionKernel() {}

  /**
   * Writes staged key and value rows into the position-major device mirror at their own positions.
   *
   * <p>Rows at or beyond {@code state[STORE_ROWS_INDEX]} are padding and are skipped rather than
   * written, so a short final prompt chunk cannot overwrite a mirrored position with zeros. The
   * same task serves two jobs that differ only in how the state vector is set: appending the rows
   * of the chunk about to be attended, and backfilling a run of rows the Java path already cached
   * before the device saw this sequence. Backfill runs through this task rather than a graph of its
   * own so that exactly one task graph ever owns the mirror buffers.
   */
  public static void store(
      IntArray state,
      FloatArray key,
      FloatArray value,
      FloatArray keyCache,
      FloatArray valueCache,
      int keyDim,
      int valueDim) {
    int executionBatchSize = key.getSize() / keyDim;
    int entriesPerRow = keyDim + valueDim;
    for (@Parallel int entry = 0; entry < executionBatchSize * entriesPerRow; entry++) {
      int batch = entry / entriesPerRow;
      int component = entry - batch * entriesPerRow;
      if (batch < state.get(STORE_ROWS_INDEX)) {
        int position = state.get(STORE_POSITION_INDEX) + batch;
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

  /**
   * Causal grouped-query attention with one independent work item per (batch row, query head).
   *
   * @param state {@code {storePosition, storeRows, attendPosition, attendRows}}
   * @param query batch-major query rows, {@code executionBatchSize * numHeads * keyLength}
   * @param keyCache position-major mirrored keys, {@code maxSequenceLength * keyDim}
   * @param valueCache position-major mirrored values, {@code maxSequenceLength * valueDim}
   * @param scores per-work-item score scratch, {@code executionBatchSize * numHeads *
   *     maxSequenceLength}
   * @param output batch-major attention output, {@code executionBatchSize * numHeads * valueLength}
   * @param groupSize query heads per key/value head
   * @param slidingWindow bounded backward reach, or zero for an unbounded causal prefix
   * @param scale the multiplier the Java path applies to every score; a task constant because a
   *     loaded model never changes it, and a parameter rather than {@code 1/sqrt(keyLength)}
   *     because Granite persists a scale that is not that
   */
  public static void attend(
      IntArray state,
      FloatArray query,
      FloatArray keyCache,
      FloatArray valueCache,
      FloatArray scores,
      FloatArray output,
      int numHeads,
      int groupSize,
      int keyLength,
      int valueLength,
      int keyDim,
      int valueDim,
      int maxSequenceLength,
      int slidingWindow,
      float scale) {
    int queryDim = numHeads * keyLength;
    int outputDim = numHeads * valueLength;
    int executionBatchSize = query.getSize() / queryDim;
    for (@Parallel int task = 0; task < executionBatchSize * numHeads; task++) {
      int batch = task / numHeads;
      int head = task - batch * numHeads;
      int outputOffset = batch * outputDim + head * valueLength;
      if (batch < state.get(ATTEND_ROWS_INDEX)) {
        int position = state.get(ATTEND_POSITION_INDEX) + batch;
        int firstPosition = 0;
        if (slidingWindow > 0 && position - slidingWindow + 1 > 0) {
          firstPosition = position - slidingWindow + 1;
        }
        int kvHead = head / groupSize;
        int queryOffset = batch * queryDim + head * keyLength;
        int scoreOffset = task * maxSequenceLength;

        float maximum = Float.NEGATIVE_INFINITY;
        for (int cached = firstPosition; cached <= position; cached++) {
          int keyOffset = cached * keyDim + kvHead * keyLength;
          float score = 0.0f;
          for (int column = 0; column < keyLength; column++) {
            score += query.get(queryOffset + column) * keyCache.get(keyOffset + column);
          }
          score = score * scale;
          scores.set(scoreOffset + cached, score);
          maximum = TornadoMath.max(maximum, score);
        }

        float sum = 0.0f;
        for (int cached = firstPosition; cached <= position; cached++) {
          float weight = TornadoMath.exp(scores.get(scoreOffset + cached) - maximum);
          scores.set(scoreOffset + cached, weight);
          sum += weight;
        }

        // Normalise the weights before accumulating, exactly as TensorOps.softmax does, rather
        // than dividing the accumulated output once at the end. The two differ in the last bits.
        float inverseSum = 1.0f / sum;
        for (int cached = firstPosition; cached <= position; cached++) {
          scores.set(scoreOffset + cached, scores.get(scoreOffset + cached) * inverseSum);
        }

        for (int column = 0; column < valueLength; column++) {
          output.set(outputOffset + column, 0.0f);
        }
        for (int cached = firstPosition; cached <= position; cached++) {
          float weight = scores.get(scoreOffset + cached);
          int valueOffset = cached * valueDim + kvHead * valueLength;
          for (int column = 0; column < valueLength; column++) {
            output.set(
                outputOffset + column,
                output.get(outputOffset + column) + weight * valueCache.get(valueOffset + column));
          }
        }
      } else {
        for (int column = 0; column < valueLength; column++) {
          output.set(outputOffset + column, 0.0f);
        }
      }
    }
  }

  /** Rejects storage that does not match the plan shape before a graph is ever built. */
  public static void validate(
      FloatArray query,
      FloatArray output,
      FloatArray scores,
      int executionBatchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength) {
    if (executionBatchSize < 1) {
      throw new IllegalArgumentException("executionBatchSize must be positive");
    }
    if (numHeads < 1 || numKvHeads < 1) {
      throw new IllegalArgumentException("numHeads and numKvHeads must be positive");
    }
    if (numHeads % numKvHeads != 0) {
      throw new IllegalArgumentException(
          "numHeads must be a multiple of numKvHeads: " + numHeads + " % " + numKvHeads);
    }
    if (keyLength < 1 || valueLength < 1) {
      throw new IllegalArgumentException("keyLength and valueLength must be positive");
    }
    if (maxSequenceLength < executionBatchSize) {
      throw new IllegalArgumentException("maxSequenceLength must cover the execution batch");
    }
    if (query.getSize()
        != Math.multiplyExact(Math.multiplyExact(executionBatchSize, numHeads), keyLength)) {
      throw new IllegalArgumentException("query storage does not match the attention shape");
    }
    if (output.getSize()
        != Math.multiplyExact(Math.multiplyExact(executionBatchSize, numHeads), valueLength)) {
      throw new IllegalArgumentException("output storage does not match the attention shape");
    }
    if (scores.getSize()
        != Math.multiplyExact(
            Math.multiplyExact(executionBatchSize, numHeads), maxSequenceLength)) {
      throw new IllegalArgumentException("scores storage does not match the attention shape");
    }
  }
}
