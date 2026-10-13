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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

class PinnedAttentionWidthContractTest {
  private static float fold(float[] a) {
    return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
  }

  @Test
  void scoreTailsRetainEightLogicalFmaChainsAndPinnedFold() {
    Random random = new Random(921);
    for (int columns : new int[] {1, 4, 7, 8, 9, 15, 16, 17, 64, 65}) {
      int rows = 7, heads = 5, stride = columns + 3;
      float[] q = new float[heads * stride + 2],
          k = new float[rows * stride + 2],
          actual = new float[heads * (rows + 2) + 3];
      for (int i = 0; i < q.length; i++) q[i] = random.nextFloat() * 2 - 1;
      for (int i = 0; i < k.length; i++) k[i] = random.nextFloat() * 2 - 1;
      GroupedQueryAttentionKernel.scoreGroup(
          q, 2, stride, heads, k, 2, stride, rows, columns, .125f, actual, 3, rows + 2);
      for (int head = 0; head < heads; head++)
        for (int row = 0; row < rows; row++) {
          float[] partial = new float[8];
          int column = 0;
          for (; column + 8 <= columns; column += 8)
            for (int lane = 0; lane < 8; lane++)
              partial[lane] =
                  Math.fma(
                      q[2 + head * stride + column + lane],
                      k[2 + row * stride + column + lane],
                      partial[lane]);
          float sum = fold(partial);
          for (; column < columns; column++)
            sum = Math.fma(q[2 + head * stride + column], k[2 + row * stride + column], sum);
          assertThat(Float.floatToRawIntBits(actual[3 + head * (rows + 2) + row]))
              .isEqualTo(Float.floatToRawIntBits(sum * .125f));
        }
    }
  }

  @Test
  void softmaxRetainsPolynomialOutputsAndEightLaneSumAcrossTails() {
    Random random = new Random(515);
    for (int size : new int[] {1, 4, 7, 8, 9, 15, 16, 17, 257, 515}) {
      float[] values = new float[size + 5];
      for (int i = 0; i < values.length; i++) values[i] = random.nextFloat() * 32 - 16;
      float[] expected = values.clone();
      float max = Float.NEGATIVE_INFINITY;
      for (int i = 0; i < size; i++) max = Math.max(max, expected[2 + i]);
      float[] sums = new float[8];
      int index = 0;
      for (; index + 8 <= size; index += 8)
        for (int lane = 0; lane < 8; lane++) {
          float value = GroupedQueryAttentionKernel.expScalar(expected[2 + index + lane] - max);
          expected[2 + index + lane] = value;
          sums[lane] += value;
        }
      float sum = fold(sums);
      for (; index < size; index++) {
        float value = GroupedQueryAttentionKernel.expScalar(expected[2 + index] - max);
        expected[2 + index] = value;
        sum += value;
      }
      float inverse = 1f / sum;
      for (int i = 0; i < size; i++) expected[2 + i] *= inverse;
      GroupedQueryAttentionKernel.softmax(values, 2, size);
      assertThat(Arrays.equals(raw(expected), raw(values))).isTrue();
    }
  }

  private static int[] raw(float[] values) {
    int[] bits = new int[values.length];
    for (int i = 0; i < values.length; i++) bits[i] = Float.floatToRawIntBits(values[i]);
    return bits;
  }
}
