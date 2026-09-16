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
    boolean vectorized = columns > 2 * lanes;
    int vectorLimit = vectorized ? SPECIES.loopBound(columns) : 0;
    int unrolledLimit = vectorLimit - 3 * lanes;
    for (int row = 0; row < rows; row++) {
      int keyBase = keyOffset + row * keyRowStride;
      for (int head = 0; head < groupSize; head++) {
        int queryBase = queryOffset + head * queryHeadStride;
        float sum = 0.0f;
        int column = 0;
        if (vectorized) {
          FloatVector acc0 = FloatVector.zero(SPECIES);
          FloatVector acc1 = FloatVector.zero(SPECIES);
          FloatVector acc2 = FloatVector.zero(SPECIES);
          FloatVector acc3 = FloatVector.zero(SPECIES);
          for (; column < unrolledLimit; column += 4 * lanes) {
            acc0 = fma(load(query, queryBase + column), load(keys, keyBase + column), acc0);
            acc1 =
                fma(
                    load(query, queryBase + column + lanes),
                    load(keys, keyBase + column + lanes),
                    acc1);
            acc2 =
                fma(
                    load(query, queryBase + column + 2 * lanes),
                    load(keys, keyBase + column + 2 * lanes),
                    acc2);
            acc3 =
                fma(
                    load(query, queryBase + column + 3 * lanes),
                    load(keys, keyBase + column + 3 * lanes),
                    acc3);
          }
          for (; column < vectorLimit; column += lanes) {
            acc0 = fma(load(query, queryBase + column), load(keys, keyBase + column), acc0);
          }
          sum += reduceAdd(acc0.add(acc1).add(acc2.add(acc3)));
        }
        for (; column < columns; column++) {
          sum = MathUtil.fma(query[queryBase + column], keys[keyBase + column], sum);
        }
        scores[scoresOffset + head * scoresHeadStride + row] = sum * scale;
      }
    }
  }

  /**
   * Adds {@code weights_h[row] * value_row} into {@code output[outputOffset + h * outputHeadStride
   * ..]} for every head of the group, row by row in order.
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
    for (int column = 0; column < vectorLimit; column += lanes) {
      for (int head = 0; head < groupSize; head++) {
        int outputBase = outputOffset + head * outputHeadStride + column;
        int weightBase = weightsOffset + head * weightsHeadStride;
        FloatVector result = load(output, outputBase);
        for (int row = 0; row < rows; row++) {
          result =
              fma(
                  load(values, valueOffset + row * valueRowStride + column),
                  FloatVector.broadcast(SPECIES, weights[weightBase + row]),
                  result);
        }
        result.intoArray(output, outputBase);
      }
    }
    for (int column = vectorLimit; column < columns; column++) {
      for (int head = 0; head < groupSize; head++) {
        int outputIndex = outputOffset + head * outputHeadStride + column;
        int weightBase = weightsOffset + head * weightsHeadStride;
        float result = output[outputIndex];
        for (int row = 0; row < rows; row++) {
          result =
              MathUtil.fma(
                  values[valueOffset + row * valueRowStride + column],
                  weights[weightBase + row],
                  result);
        }
        output[outputIndex] = result;
      }
    }
  }

  private static FloatVector load(float[] array, int offset) {
    return FloatVector.fromArray(SPECIES, array, offset);
  }

  private static FloatVector fma(FloatVector a, FloatVector b, FloatVector c) {
    return PanamaConstants.HAS_FAST_VECTOR_FMA ? a.fma(b, c) : a.mul(b).add(c);
  }

  /** The fixed reduction order vectors-core uses, so sums match lane for lane. */
  static float reduceAdd(FloatVector vector) {
    return switch (vector.length()) {
      case 1 -> vector.lane(0);
      case 2 -> vector.lane(1) + vector.lane(0);
      case 4 -> (vector.lane(2) + vector.lane(0)) + (vector.lane(3) + vector.lane(1));
      case 8 -> {
        float even = (vector.lane(4) + vector.lane(0)) + (vector.lane(6) + vector.lane(2));
        float odd = (vector.lane(5) + vector.lane(1)) + (vector.lane(7) + vector.lane(3));
        yield even + odd;
      }
      case 16 -> {
        float lane0 = vector.lane(8) + vector.lane(0);
        float lane1 = vector.lane(9) + vector.lane(1);
        float lane2 = vector.lane(10) + vector.lane(2);
        float lane3 = vector.lane(11) + vector.lane(3);
        float lane4 = vector.lane(12) + vector.lane(4);
        float lane5 = vector.lane(13) + vector.lane(5);
        float lane6 = vector.lane(14) + vector.lane(6);
        float lane7 = vector.lane(15) + vector.lane(7);
        float even = (lane4 + lane0) + (lane6 + lane2);
        float odd = (lane5 + lane1) + (lane7 + lane3);
        yield even + odd;
      }
      default -> throw new AssertionError("unsupported float vector length: " + vector.length());
    };
  }
}
