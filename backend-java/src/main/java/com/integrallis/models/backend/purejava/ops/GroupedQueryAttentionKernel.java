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
package com.integrallis.models.backend.purejava.ops;

import com.integrallis.vectors.core.MathUtil;
import com.integrallis.vectors.core.PanamaConstants;
import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Grouped-query attention arithmetic that reads each cached key and value row once for every query
 * head that shares it.
 *
 * <p>The per-head results are bit-identical to what {@code VectorUtil.matVecDotExact} (scores) and
 * {@code VectorUtil.addWeightedRowsInPlace} (values) produce head by head: the same species, the
 * same FMA selection, the same four-accumulator column unrolling, the same fixed lane reduction,
 * and the same sequential row chain. Only the memory traffic changes; a Granite 4.1 group of five
 * heads used to stream every K and V row five times.
 */
public final class GroupedQueryAttentionKernel {
  static final VectorSpecies<Float> SPECIES = species();

  private GroupedQueryAttentionKernel() {}

  private static VectorSpecies<Float> species() {
    VectorSpecies<Float> preferred = FloatVector.SPECIES_PREFERRED;
    if (preferred.vectorBitSize() <= PanamaConstants.MAX_BITS) {
      return preferred;
    }
    return VectorSpecies.of(float.class, VectorShape.forBitSize(PanamaConstants.MAX_BITS));
  }

  /**
   * Writes {@code scale * dot(query_h, key_row)} for every head {@code h} of one group and every
   * key row into {@code scores[scoresOffset + h * scoresHeadStride + row]}.
   */
  public static void scoreGroup(
      float[] query,
      int queryOffset,
      int queryHeadStride,
      int groupSize,
      float[] keys,
      int keyOffset,
      int keyRowStride,
      int rows,
      int columns,
      float scale,
      float[] scores,
      int scoresOffset,
      int scoresHeadStride) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(keys, "keys");
    Objects.requireNonNull(scores, "scores");
    if (groupSize < 1 || rows < 0 || columns < 1) {
      throw new IllegalArgumentException("groupSize and columns must be positive, rows >= 0");
    }
    Objects.checkFromIndexSize(
        queryOffset, (groupSize - 1) * queryHeadStride + columns, query.length);
    if (rows > 0) {
      Objects.checkFromIndexSize(keyOffset, (rows - 1) * keyRowStride + columns, keys.length);
      Objects.checkFromIndexSize(
          scoresOffset, (groupSize - 1) * scoresHeadStride + rows, scores.length);
    }
    int lanes = SPECIES.length();
    int vectorLimit = SPECIES.loopBound(columns);
    int pairLimit = vectorLimit - lanes;
    for (int row = 0; row < rows; row++) {
      int keyBase = keyOffset + row * keyRowStride;
      for (int head = 0; head < groupSize; head++) {
        int queryBase = queryOffset + head * queryHeadStride;
        FloatVector acc0 = FloatVector.zero(SPECIES);
        FloatVector acc1 = FloatVector.zero(SPECIES);
        int column = 0;
        for (; column < pairLimit; column += 2 * lanes) {
          acc0 = fma(load(query, queryBase + column), load(keys, keyBase + column), acc0);
          acc1 =
              fma(
                  load(query, queryBase + column + lanes),
                  load(keys, keyBase + column + lanes),
                  acc1);
        }
        for (; column < vectorLimit; column += lanes) {
          acc0 = fma(load(query, queryBase + column), load(keys, keyBase + column), acc0);
        }
        float sum = acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        for (; column < columns; column++) {
          sum = MathUtil.fma(query[queryBase + column], keys[keyBase + column], sum);
        }
        scores[scoresOffset + head * scoresHeadStride + row] = sum * scale;
      }
    }
  }

  /**
   * In-place stable softmax over {@code x[offset, offset + size)}; the same contract as {@code
   * TensorOps.softmax} (rejects NaN and +infinity, requires one finite input) with vector
   * exponentials.
   */
  public static void softmax(float[] x, int offset, int size) {
    Objects.requireNonNull(x, "x");
    if (size <= 0) {
      throw new IllegalArgumentException("size must be positive: " + size);
    }
    Objects.checkFromIndexSize(offset, size, x.length);
    int lanes = SPECIES.length();
    int vectorLimit = SPECIES.loopBound(size);
    FloatVector maxVector = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
    int index = 0;
    for (; index < vectorLimit; index += lanes) {
      maxVector = maxVector.max(load(x, offset + index));
    }
    float max = maxVector.reduceLanes(VectorOperators.MAX);
    for (; index < size; index++) {
      max = Math.max(max, x[offset + index]);
    }
    for (int probe = 0; probe < size; probe++) {
      float value = x[offset + probe];
      if (Float.isNaN(value)) {
        throw new IllegalArgumentException(
            "softmax input contains NaN at index " + (offset + probe));
      }
      if (value == Float.POSITIVE_INFINITY) {
        throw new IllegalArgumentException(
            "softmax input contains positive infinity at index " + (offset + probe));
      }
    }
    if (!Float.isFinite(max)) {
      throw new IllegalArgumentException("softmax requires at least one finite input");
    }
    FloatVector maxBroadcast = FloatVector.broadcast(SPECIES, max);
    FloatVector sumVector = FloatVector.zero(SPECIES);
    index = 0;
    for (; index < vectorLimit; index += lanes) {
      FloatVector value = load(x, offset + index).sub(maxBroadcast).lanewise(VectorOperators.EXP);
      value.intoArray(x, offset + index);
      sumVector = sumVector.add(value);
    }
    float sum = sumVector.reduceLanes(VectorOperators.ADD);
    for (; index < size; index++) {
      float value = (float) Math.exp(x[offset + index] - max);
      x[offset + index] = value;
      sum += value;
    }
    float inverseSum = 1.0f / sum;
    FloatVector inverse = FloatVector.broadcast(SPECIES, inverseSum);
    index = 0;
    for (; index < vectorLimit; index += lanes) {
      load(x, offset + index).mul(inverse).intoArray(x, offset + index);
    }
    for (; index < size; index++) {
      x[offset + index] *= inverseSum;
    }
  }

  /**
   * Adds {@code weights_h[row] * value_row} into {@code output[outputOffset + h * outputHeadStride
   * ..]} for every head of the group, row by row in order. The per-column FMA chain runs in row
   * order, so results match {@code VectorUtil.addWeightedRowsInPlace} bit for bit.
   */
  public static void accumulateGroup(
      float[] output,
      int outputOffset,
      int outputHeadStride,
      int groupSize,
      float[] values,
      int valueOffset,
      int valueRowStride,
      int rows,
      int columns,
      float[] weights,
      int weightsOffset,
      int weightsHeadStride) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(values, "values");
    Objects.requireNonNull(weights, "weights");
    if (groupSize < 1 || rows < 0 || columns < 1) {
      throw new IllegalArgumentException("groupSize and columns must be positive, rows >= 0");
    }
    Objects.checkFromIndexSize(
        outputOffset, (groupSize - 1) * outputHeadStride + columns, output.length);
    if (rows > 0) {
      Objects.checkFromIndexSize(valueOffset, (rows - 1) * valueRowStride + columns, values.length);
      Objects.checkFromIndexSize(
          weightsOffset, (groupSize - 1) * weightsHeadStride + rows, weights.length);
    }
    int lanes = SPECIES.length();
    int vectorLimit = SPECIES.loopBound(columns);
    int quadLimit = vectorLimit - 3 * lanes;
    for (int head = 0; head < groupSize; head++) {
      int outputBase = outputOffset + head * outputHeadStride;
      int weightBase = weightsOffset + head * weightsHeadStride;
      int column = 0;
      // Four column vectors stay in registers across every row: one weight broadcast and four
      // loads per row, no accumulator traffic to memory until the rows are exhausted.
      for (; column < quadLimit; column += 4 * lanes) {
        FloatVector acc0 = load(output, outputBase + column);
        FloatVector acc1 = load(output, outputBase + column + lanes);
        FloatVector acc2 = load(output, outputBase + column + 2 * lanes);
        FloatVector acc3 = load(output, outputBase + column + 3 * lanes);
        for (int row = 0; row < rows; row++) {
          int rowBase = valueOffset + row * valueRowStride + column;
          FloatVector weight = FloatVector.broadcast(SPECIES, weights[weightBase + row]);
          acc0 = fma(load(values, rowBase), weight, acc0);
          acc1 = fma(load(values, rowBase + lanes), weight, acc1);
          acc2 = fma(load(values, rowBase + 2 * lanes), weight, acc2);
          acc3 = fma(load(values, rowBase + 3 * lanes), weight, acc3);
        }
        acc0.intoArray(output, outputBase + column);
        acc1.intoArray(output, outputBase + column + lanes);
        acc2.intoArray(output, outputBase + column + 2 * lanes);
        acc3.intoArray(output, outputBase + column + 3 * lanes);
      }
      for (; column < vectorLimit; column += lanes) {
        FloatVector acc = load(output, outputBase + column);
        for (int row = 0; row < rows; row++) {
          acc =
              fma(
                  load(values, valueOffset + row * valueRowStride + column),
                  FloatVector.broadcast(SPECIES, weights[weightBase + row]),
                  acc);
        }
        acc.intoArray(output, outputBase + column);
      }
      for (; column < columns; column++) {
        float result = output[outputBase + column];
        for (int row = 0; row < rows; row++) {
          result =
              MathUtil.fma(
                  values[valueOffset + row * valueRowStride + column],
                  weights[weightBase + row],
                  result);
        }
        output[outputBase + column] = result;
      }
    }
  }

  private static FloatVector load(float[] array, int offset) {
    return FloatVector.fromArray(SPECIES, array, offset);
  }

  private static FloatVector fma(FloatVector a, FloatVector b, FloatVector c) {
    return PanamaConstants.HAS_FAST_VECTOR_FMA ? a.fma(b, c) : a.mul(b).add(c);
  }
}
