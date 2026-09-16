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

import com.integrallis.vectors.core.VectorUtil;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * The fused kernels must reproduce the head-by-head vectors-core kernels bit for bit
 * (batchDotProductExact, addWeightedRowsInPlace).
 */
class GroupedQueryAttentionKernelTest {
  private static final int[][] SHAPES = {
    {64, 5, 301, 512}, {128, 8, 7, 256}, {16, 1, 3, 48}, {96, 3, 2, 96}, {64, 4, 0, 512}
  };

  @Test
  void scoresMatchMatVecDotExactHeadByHead() {
    for (int[] shape : SHAPES) {
      int columns = shape[0];
      int groupSize = shape[1];
      int rows = shape[2];
      int keyRowStride = shape[3];
      Random random = new Random(7L * columns + rows);
      float[] query = randomArray(random, 3 + groupSize * columns);
      float[] keys = randomArray(random, 11 + Math.max(1, rows) * keyRowStride);
      int keyOffset = 11 + (keyRowStride - columns) / 2;
      float scale = 0.015625f;
      int stride = rows + 5;
      float[] fused = new float[2 + groupSize * stride];
      GroupedQueryAttentionKernel.scoreGroup(
          query,
          3,
          columns,
          groupSize,
          keys,
          keyOffset,
          keyRowStride,
          rows,
          columns,
          scale,
          fused,
          2,
          stride);
      for (int head = 0; head < groupSize; head++) {
        float[] reference = new float[rows];
        if (rows > 0) {
          VectorUtil.batchDotProductExact(
              query,
              3 + head * columns,
              keys,
              keyOffset,
              keyRowStride,
              rows,
              columns,
              reference,
              0);
        }
        for (int row = 0; row < rows; row++) {
          assertThat(fused[2 + head * stride + row])
              .as("columns=%d head=%d row=%d", columns, head, row)
              .isEqualTo(reference[row] * scale);
        }
      }
    }
  }

  @Test
  void valuesMatchAddWeightedRowsInPlaceHeadByHead() {
    for (int[] shape : SHAPES) {
      int columns = shape[0];
      int groupSize = shape[1];
      int rows = shape[2];
      int valueRowStride = shape[3];
      Random random = new Random(13L * columns + rows);
      float[] values = randomArray(random, 5 + Math.max(1, rows) * valueRowStride);
      int valueOffset = 5 + (valueRowStride - columns) / 3;
      int stride = rows + 2;
      float[] weights = randomArray(random, 1 + groupSize * stride);
      float[] fused = randomArray(random, 4 + groupSize * columns);
      float[] reference = fused.clone();
      GroupedQueryAttentionKernel.accumulateGroup(
          fused,
          4,
          columns,
          groupSize,
          values,
          valueOffset,
          valueRowStride,
          rows,
          columns,
          weights,
          1,
          stride);
      for (int head = 0; head < groupSize; head++) {
        if (rows > 0) {
          VectorUtil.addWeightedRowsInPlace(
              reference,
              4 + head * columns,
              values,
              valueOffset,
              valueRowStride,
              weights,
              1 + head * stride,
              rows,
              columns);
        }
      }
      assertThat(fused)
          .as("columns=%d group=%d rows=%d", columns, groupSize, rows)
          .isEqualTo(reference);
    }
  }

  private static float[] randomArray(Random random, int length) {
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = (random.nextFloat() - 0.5f) * 4.0f;
    }
    return values;
  }
}
