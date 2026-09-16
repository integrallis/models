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
import jdk.incubator.vector.VectorShuffle;
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
  private static final VectorShuffle<Float> ROTATE_8 = rotation(8);
  private static final VectorShuffle<Float> ROTATE_4 = rotation(4);
  private static final VectorShuffle<Float> ROTATE_2 = rotation(2);
  private static final VectorShuffle<Float> ROTATE_1 = rotation(1);

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
    int blockedRows = rows & ~3;
    // Four rows per head at a time: four independent FMA chains hide the FMA latency that a single
    // dependent chain over 64 columns exposes, and each query vector load serves four rows.
    for (int row = 0; row < blockedRows; row += 4) {
      int key0 = keyOffset + row * keyRowStride;
      int key1 = key0 + keyRowStride;
      int key2 = key1 + keyRowStride;
      int key3 = key2 + keyRowStride;
      for (int head = 0; head < groupSize; head++) {
        int queryBase = queryOffset + head * queryHeadStride;
        FloatVector acc0 = FloatVector.zero(SPECIES);
        FloatVector acc1 = FloatVector.zero(SPECIES);
        FloatVector acc2 = FloatVector.zero(SPECIES);
        FloatVector acc3 = FloatVector.zero(SPECIES);
        int column = 0;
        for (; column < vectorLimit; column += lanes) {
          FloatVector q = load(query, queryBase + column);
          acc0 = fma(q, load(keys, key0 + column), acc0);
          acc1 = fma(q, load(keys, key1 + column), acc1);
          acc2 = fma(q, load(keys, key2 + column), acc2);
          acc3 = fma(q, load(keys, key3 + column), acc3);
        }
        float sum0 = reduceAddFixedTree(acc0);
        float sum1 = reduceAddFixedTree(acc1);
        float sum2 = reduceAddFixedTree(acc2);
        float sum3 = reduceAddFixedTree(acc3);
        for (; column < columns; column++) {
          float q = query[queryBase + column];
          sum0 = MathUtil.fma(q, keys[key0 + column], sum0);
          sum1 = MathUtil.fma(q, keys[key1 + column], sum1);
          sum2 = MathUtil.fma(q, keys[key2 + column], sum2);
          sum3 = MathUtil.fma(q, keys[key3 + column], sum3);
        }
        int scoreBase = scoresOffset + head * scoresHeadStride + row;
        scores[scoreBase] = sum0 * scale;
        scores[scoreBase + 1] = sum1 * scale;
        scores[scoreBase + 2] = sum2 * scale;
        scores[scoreBase + 3] = sum3 * scale;
      }
    }
    for (int row = blockedRows; row < rows; row++) {
      int keyBase = keyOffset + row * keyRowStride;
      for (int head = 0; head < groupSize; head++) {
        int queryBase = queryOffset + head * queryHeadStride;
        FloatVector acc = FloatVector.zero(SPECIES);
        int column = 0;
        for (; column < vectorLimit; column += lanes) {
          acc = fma(load(query, queryBase + column), load(keys, keyBase + column), acc);
        }
        float sum = reduceAddFixedTree(acc);
        for (; column < columns; column++) {
          sum = MathUtil.fma(query[queryBase + column], keys[keyBase + column], sum);
        }
        scores[scoresOffset + head * scoresHeadStride + row] = sum * scale;
      }
    }
  }

  /**
   * In-place stable softmax over {@code x[offset, offset + size)}; the same contract as {@code
   * TensorOps.softmax} (rejects NaN and +infinity, requires one finite input) with a vector maximum
   * and normalisation and a tier-stable scalar exponential.
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
    // The exponential stays scalar Math.exp on purpose: HotSpot gives Math.exp the same result in
    // the interpreter and in compiled code, whereas lanewise(EXP) switches to the vector math
    // library once C2 compiles this method and then differs from its pre-compilation fallback in
    // the last bits. Attention outputs must not depend on JIT tier.
    float sum = 0.0f;
    for (index = 0; index < size; index++) {
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
    int pairedHeads = groupSize & ~1;
    // Two heads at a time share every value load; four column vectors per head stay in registers
    // across all rows. The per-(head, column) FMA chain runs in row order, so results match the
    // head-by-head kernel bit for bit.
    for (int head = 0; head < pairedHeads; head += 2) {
      int outputA = outputOffset + head * outputHeadStride;
      int outputB = outputA + outputHeadStride;
      int weightA = weightsOffset + head * weightsHeadStride;
      int weightB = weightA + weightsHeadStride;
      int column = 0;
      for (; column < quadLimit; column += 4 * lanes) {
        FloatVector a0 = load(output, outputA + column);
        FloatVector a1 = load(output, outputA + column + lanes);
        FloatVector a2 = load(output, outputA + column + 2 * lanes);
        FloatVector a3 = load(output, outputA + column + 3 * lanes);
        FloatVector b0 = load(output, outputB + column);
        FloatVector b1 = load(output, outputB + column + lanes);
        FloatVector b2 = load(output, outputB + column + 2 * lanes);
        FloatVector b3 = load(output, outputB + column + 3 * lanes);
        for (int row = 0; row < rows; row++) {
          int rowBase = valueOffset + row * valueRowStride + column;
          FloatVector v0 = load(values, rowBase);
          FloatVector v1 = load(values, rowBase + lanes);
          FloatVector v2 = load(values, rowBase + 2 * lanes);
          FloatVector v3 = load(values, rowBase + 3 * lanes);
          FloatVector wa = FloatVector.broadcast(SPECIES, weights[weightA + row]);
          FloatVector wb = FloatVector.broadcast(SPECIES, weights[weightB + row]);
          a0 = fma(v0, wa, a0);
          a1 = fma(v1, wa, a1);
          a2 = fma(v2, wa, a2);
          a3 = fma(v3, wa, a3);
          b0 = fma(v0, wb, b0);
          b1 = fma(v1, wb, b1);
          b2 = fma(v2, wb, b2);
          b3 = fma(v3, wb, b3);
        }
        a0.intoArray(output, outputA + column);
        a1.intoArray(output, outputA + column + lanes);
        a2.intoArray(output, outputA + column + 2 * lanes);
        a3.intoArray(output, outputA + column + 3 * lanes);
        b0.intoArray(output, outputB + column);
        b1.intoArray(output, outputB + column + lanes);
        b2.intoArray(output, outputB + column + 2 * lanes);
        b3.intoArray(output, outputB + column + 3 * lanes);
      }
      for (; column < vectorLimit; column += lanes) {
        FloatVector a = load(output, outputA + column);
        FloatVector b = load(output, outputB + column);
        for (int row = 0; row < rows; row++) {
          FloatVector v = load(values, valueOffset + row * valueRowStride + column);
          a = fma(v, FloatVector.broadcast(SPECIES, weights[weightA + row]), a);
          b = fma(v, FloatVector.broadcast(SPECIES, weights[weightB + row]), b);
        }
        a.intoArray(output, outputA + column);
        b.intoArray(output, outputB + column);
      }
      for (; column < columns; column++) {
        float a = output[outputA + column];
        float b = output[outputB + column];
        for (int row = 0; row < rows; row++) {
          float v = values[valueOffset + row * valueRowStride + column];
          a = MathUtil.fma(v, weights[weightA + row], a);
          b = MathUtil.fma(v, weights[weightB + row], b);
        }
        output[outputA + column] = a;
        output[outputB + column] = b;
      }
    }
    for (int head = pairedHeads; head < groupSize; head++) {
      int outputBase = outputOffset + head * outputHeadStride;
      int weightBase = weightsOffset + head * weightsHeadStride;
      int column = 0;
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

  private static VectorShuffle<Float> rotation(int distance) {
    int mask = SPECIES.length() - 1;
    return VectorShuffle.fromOp(SPECIES, lane -> (lane + distance) & mask);
  }

  /**
   * Sums the lanes through an explicit rotate-and-add tree. {@code reduceLanes(ADD)} is not used
   * because its compiled form reduces as a tree while its pre-compilation fallback sums lanes in
   * sequence, so the same input would round differently by JIT tier; this tree is the same sequence
   * of lanewise adds in both.
   */
  private static float reduceAddFixedTree(FloatVector vector) {
    int lanes = vector.length();
    if (lanes == 16) {
      vector = vector.add(vector.rearrange(ROTATE_8));
    }
    if (lanes >= 8) {
      vector = vector.add(vector.rearrange(ROTATE_4));
    }
    if (lanes >= 4) {
      vector = vector.add(vector.rearrange(ROTATE_2));
    }
    if (lanes >= 2) {
      vector = vector.add(vector.rearrange(ROTATE_1));
    }
    return vector.lane(0);
  }

  private static FloatVector load(float[] array, int offset) {
    return FloatVector.fromArray(SPECIES, array, offset);
  }

  private static FloatVector fma(FloatVector a, FloatVector b, FloatVector c) {
    return PanamaConstants.HAS_FAST_VECTOR_FMA ? a.fma(b, c) : a.mul(b).add(c);
  }
}
