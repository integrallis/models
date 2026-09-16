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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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
          float expected = reference[row] * scale;
          assertThat(fused[2 + head * stride + row])
              .as("columns=%d head=%d row=%d", columns, head, row)
              .isCloseTo(
                  expected,
                  org.assertj.core.data.Offset.offset(2e-6f + Math.abs(expected) * 2e-6f));
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

  @Test
  void softmaxMatchesTheScalarReferenceWithinLastBits() {
    for (int size : new int[] {1, 5, 8, 37, 301}) {
      Random random = new Random(size);
      float[] scalar = randomArray(random, size + 3);
      for (int index = 0; index < scalar.length; index++) {
        scalar[index] *= 6.0f;
      }
      float[] vector = scalar.clone();
      TensorOps.softmax(scalar, 2, size);
      GroupedQueryAttentionKernel.softmax(vector, 2, size);
      float total = 0.0f;
      for (int index = 0; index < size; index++) {
        assertThat(vector[2 + index])
            .as("size=%d index=%d", size, index)
            .isCloseTo(scalar[2 + index], org.assertj.core.data.Offset.offset(2e-6f));
        total += vector[2 + index];
      }
      assertThat(total).isCloseTo(1.0f, org.assertj.core.data.Offset.offset(1e-4f));
      assertThat(vector[0]).isEqualTo(scalar[0]);
      assertThat(vector[size + 2]).isEqualTo(scalar[size + 2]);
    }
    assertThatThrownBy(() -> GroupedQueryAttentionKernel.softmax(new float[] {1f, Float.NaN}, 0, 2))
        .hasMessageContaining("NaN");
  }

  @Test
  void vectorExponentialMatchesItsScalarTwinAndMathExp() {
    float[] inputs = new float[64];
    Random random = new Random(3);
    for (int index = 0; index < inputs.length; index++) {
      inputs[index] = -random.nextFloat() * 30.0f;
    }
    inputs[0] = 0.0f;
    inputs[1] = -1e-7f;
    inputs[2] = -0.34657359f;
    inputs[3] = -0.3465736f;
    inputs[4] = -90.0f;
    float[] fromVector = new float[inputs.length];
    int lanes = GroupedQueryAttentionKernel.SPECIES.length();
    for (int index = 0; index + lanes <= inputs.length; index += lanes) {
      GroupedQueryAttentionKernel.expVector(
              jdk.incubator.vector.FloatVector.fromArray(
                  GroupedQueryAttentionKernel.SPECIES, inputs, index))
          .intoArray(fromVector, index);
    }
    for (int index = 0; index < inputs.length; index++) {
      float scalar = GroupedQueryAttentionKernel.expScalar(inputs[index]);
      assertThat(fromVector[index]).as("lane %d", index).isEqualTo(scalar);
      // Inputs below -87 are clamped: their probability mass is below 1e-37 of the row and
      // the clamp keeps the exponent assembly in normal range.
      double expected = Math.exp(Math.max(inputs[index], -87.0f));
      assertThat((double) scalar)
          .as("x=%s", inputs[index])
          .isCloseTo(expected, org.assertj.core.data.Offset.offset(expected * 4e-7));
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
