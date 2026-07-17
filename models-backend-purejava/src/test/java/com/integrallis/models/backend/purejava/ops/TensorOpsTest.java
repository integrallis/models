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
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.function.IntUnaryOperator;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TensorOpsTest {

  @Nested
  class RmsNorm {

    @Test
    void identityWeightNormalizesCorrectly() {
      float[] x = {3.0f, 4.0f};
      float[] weight = {1.0f, 1.0f};
      float[] out = new float[2];

      TensorOps.rmsNorm(out, x, weight, 2, 1e-5f);

      // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
      float rms = (float) Math.sqrt(12.5f);
      assertThat(out[0]).isCloseTo(3.0f / rms, within(1e-4f));
      assertThat(out[1]).isCloseTo(4.0f / rms, within(1e-4f));
    }

    @Test
    void weightScalesOutput() {
      float[] x = {1.0f, 1.0f};
      float[] weight = {2.0f, 3.0f};
      float[] out = new float[2];

      TensorOps.rmsNorm(out, x, weight, 2, 1e-5f);

      // rms = sqrt((1+1)/2) = 1.0
      assertThat(out[0]).isCloseTo(2.0f, within(1e-4f));
      assertThat(out[1]).isCloseTo(3.0f, within(1e-4f));
    }
  }

  @Nested
  class Matmul {

    @Test
    void identityMatrix() {
      // 2x2 identity * [1, 2] = [1, 2]
      float[] weight = {1, 0, 0, 1};
      float[] x = {1.0f, 2.0f};
      float[] out = new float[2];

      TensorOps.matmul(out, x, weight, 2, 2);

      assertThat(out[0]).isEqualTo(1.0f);
      assertThat(out[1]).isEqualTo(2.0f);
    }

    @Test
    void known3x2Product() {
      // [[1,2],[3,4],[5,6]] * [1,2] = [5, 11, 17]
      float[] weight = {1, 2, 3, 4, 5, 6};
      float[] x = {1.0f, 2.0f};
      float[] out = new float[3];

      TensorOps.matmul(out, x, weight, 3, 2);

      assertThat(out[0]).isEqualTo(5.0f);
      assertThat(out[1]).isEqualTo(11.0f);
      assertThat(out[2]).isEqualTo(17.0f);
    }

    @Test
    void delegatesToVectorsRowMajorGemvSemantics() {
      int rows = 5;
      int cols = 7;
      float[] x = {0.25f, -1.0f, 2.5f, 0.0f, 3.0f, -0.5f, 1.25f};
      float[] weight = new float[rows * cols];
      for (int i = 0; i < weight.length; i++) {
        weight[i] = (i % 9 - 4) * 0.125f;
      }
      float[] expected = new float[rows];
      float[] actual = new float[rows];

      VectorUtil.batchDotProduct(x, weight, rows, cols, expected);
      TensorOps.matmul(actual, x, weight, rows, cols);

      for (int row = 0; row < rows; row++) {
        assertThat(actual[row]).isCloseTo(expected[row], within(1e-5f));
      }
    }

    @Test
    void ggufF32MatrixUsesMappedVectorsKernel() {
      float[] x = {1.0f, 2.0f};
      ByteBuffer bytes = ByteBuffer.allocate(6 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : new float[] {1, 2, 3, 4, 5, 6}) {
        bytes.putFloat(value);
      }

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment weight = arena.allocate(bytes.capacity());
        MemorySegment.copy(bytes.array(), 0, weight, ValueLayout.JAVA_BYTE, 0, bytes.capacity());
        float[] actual = new float[3];

        TensorOps.ggufMatmul(actual, x, weight, GgufTensorType.F32, 3, 2);

        assertThat(actual).containsExactly(5.0f, 11.0f, 17.0f);
      }
    }
  }

  @Nested
  class QuantizedMatmul {

    @Test
    void q4_0BatchedMatmulMatchesIndependentQueries() {
      int batchSize = 3;
      int cols = 32;
      float[] queries = new float[batchSize * cols];
      float[] expected = new float[batchSize];
      float[] actual = new float[batchSize];
      for (int batch = 0; batch < batchSize; batch++) {
        for (int col = 0; col < cols; col++) {
          queries[batch * cols + col] = ((batch + 1) * (col - 13)) / 17.0f;
        }
      }

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, q4Block(0.5f));
        for (int batch = 0; batch < batchSize; batch++) {
          float[] query = new float[cols];
          System.arraycopy(queries, batch * cols, query, 0, cols);
          float[] result = new float[1];
          TensorOps.ggufMatmul(result, query, qWeight, GgufTensorType.Q4_0, 1, cols);
          expected[batch] = result[0];
        }

        TensorOps.ggufBatchedMatmul(
            actual,
            queries,
            qWeight,
            GgufTensorType.Q4_0,
            batchSize,
            1,
            cols,
            new byte[batchSize * cols],
            new float[batchSize * (cols / 32)],
            new short[batchSize * (cols / 16)],
            new float[batchSize * 8]);
      }

      assertThat(actual).containsExactly(expected);
    }

    @Test
    void q4_KBatchedMatmulMatchesIndependentQueries() {
      int batchSize = 3;
      int cols = 256;
      float[] queries = new float[batchSize * cols];
      float[] expected = new float[batchSize];
      float[] actual = new float[batchSize];
      for (int batch = 0; batch < batchSize; batch++) {
        for (int col = 0; col < cols; col++) {
          queries[batch * cols + col] =
              (float) Math.cos((batch + 1.0) * (col + 0.5)) * (batch + 0.25f);
        }
      }

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, q4KBlock(0.125f, 0.0625f, 7));
        for (int batch = 0; batch < batchSize; batch++) {
          float[] query = new float[cols];
          System.arraycopy(queries, batch * cols, query, 0, cols);
          float[] result = new float[1];
          TensorOps.ggufMatmul(result, query, qWeight, GgufTensorType.Q4_K, 1, cols);
          expected[batch] = result[0];
        }

        TensorOps.ggufBatchedMatmul(
            actual,
            queries,
            qWeight,
            GgufTensorType.Q4_K,
            batchSize,
            1,
            cols,
            new byte[batchSize * cols],
            new float[batchSize * (cols / 256)],
            new short[batchSize * (cols / 16)],
            new float[0]);
      }

      assertThat(TensorOps.supportsBatchedMatmul(GgufTensorType.Q4_K)).isTrue();
      assertThat(actual).containsExactly(expected);
    }

    @Test
    void q6_KBatchedMatmulMatchesIndependentQueries() {
      int batchSize = 3;
      int cols = 256;
      float[] queries = new float[batchSize * cols];
      float[] expected = new float[batchSize];
      float[] actual = new float[batchSize];
      for (int batch = 0; batch < batchSize; batch++) {
        for (int col = 0; col < cols; col++) {
          queries[batch * cols + col] =
              (float) Math.sin((batch + 0.5) * (col + 1.0)) * (batch + 0.75f);
        }
      }

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight =
            copy(arena, q6KBlock(0.125f, i -> (i * 11) % 64 - 32, i -> (i % 9) - 4));
        for (int batch = 0; batch < batchSize; batch++) {
          float[] query = new float[cols];
          System.arraycopy(queries, batch * cols, query, 0, cols);
          float[] result = new float[1];
          TensorOps.ggufMatmul(result, query, qWeight, GgufTensorType.Q6_K, 1, cols);
          expected[batch] = result[0];
        }

        TensorOps.ggufBatchedMatmul(
            actual,
            queries,
            qWeight,
            GgufTensorType.Q6_K,
            batchSize,
            1,
            cols,
            new byte[batchSize * cols],
            new float[batchSize * (cols / 256)],
            new short[0],
            new float[0]);
      }

      assertThat(TensorOps.supportsBatchedMatmul(GgufTensorType.Q6_K)).isTrue();
      assertThat(actual).containsExactly(expected);
    }

    @Test
    void q4_0MatchesVectorsGgmlQ8_0ActivationSemantics() {
      float[] x = repeatingQuery(32);
      byte[] row = q4Block(0.5f);
      float[] expected = new float[1];
      float[] actual = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, row);

        VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
            x, qWeight, 1, 32, expected, new byte[32], new float[1]);
        TensorOps.quantizedMatmul(actual, x, qWeight, GgufTensorType.Q4_0, 1, 32);

        assertThat(actual[0]).isCloseTo(expected[0], within(1e-5f));
      }
    }

    @Test
    void q4_0DualMatmulMatchesSeparateProjectionsExactly() {
      int cols = 32;
      float[] input = repeatingQuery(cols);
      float[] expectedFirst = new float[1];
      float[] expectedSecond = new float[1];
      float[] actualFirst = new float[1];
      float[] actualSecond = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment firstWeight = copy(arena, q4Block(0.5f));
        MemorySegment secondWeight = copy(arena, q4Block(-0.25f));
        TensorOps.ggufMatmul(expectedFirst, input, firstWeight, GgufTensorType.Q4_0, 1, cols);
        TensorOps.ggufMatmul(expectedSecond, input, secondWeight, GgufTensorType.Q4_0, 1, cols);

        TensorOps.ggufDualMatmul(
            actualFirst,
            firstWeight,
            GgufTensorType.Q4_0,
            1,
            actualSecond,
            secondWeight,
            GgufTensorType.Q4_0,
            1,
            input,
            cols,
            new byte[cols],
            new float[cols / 32],
            new short[cols / 16]);
      }

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
    }

    @Test
    void q4_0TripleMatmulMatchesSeparateProjectionsExactly() {
      int cols = 32;
      float[] input = repeatingQuery(cols);
      float[] expectedFirst = new float[1];
      float[] expectedSecond = new float[1];
      float[] expectedThird = new float[1];
      float[] actualFirst = new float[1];
      float[] actualSecond = new float[1];
      float[] actualThird = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment firstWeight = copy(arena, q4Block(0.5f));
        MemorySegment secondWeight = copy(arena, q4Block(-0.25f));
        MemorySegment thirdWeight = copy(arena, q4Block(0.125f));
        TensorOps.ggufMatmul(expectedFirst, input, firstWeight, GgufTensorType.Q4_0, 1, cols);
        TensorOps.ggufMatmul(expectedSecond, input, secondWeight, GgufTensorType.Q4_0, 1, cols);
        TensorOps.ggufMatmul(expectedThird, input, thirdWeight, GgufTensorType.Q4_0, 1, cols);

        TensorOps.ggufTripleMatmul(
            actualFirst,
            firstWeight,
            GgufTensorType.Q4_0,
            1,
            actualSecond,
            secondWeight,
            GgufTensorType.Q4_0,
            1,
            actualThird,
            thirdWeight,
            GgufTensorType.Q4_0,
            1,
            input,
            cols,
            new byte[cols],
            new float[cols / 32],
            new short[cols / 16]);
      }

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
      assertThat(actualThird).containsExactly(expectedThird);
    }

    @Test
    void groupedMatmulSupportIncludesKQuantProjectionTypes() {
      assertThat(TensorOps.supportsGroupedMatmul(GgufTensorType.Q4_0)).isTrue();
      assertThat(TensorOps.supportsGroupedMatmul(GgufTensorType.Q8_0)).isTrue();
      assertThat(TensorOps.supportsGroupedMatmul(GgufTensorType.Q4_K)).isTrue();
      assertThat(TensorOps.supportsGroupedMatmul(GgufTensorType.Q5_K)).isTrue();
    }

    @Test
    void groupedTriplePlanRetainsMatchingPairInMixedKQuantLayers() {
      assertThat(
              TensorOps.groupedProjectionPlan(
                  GgufTensorType.Q4_K, GgufTensorType.Q4_K, GgufTensorType.Q4_K))
          .isEqualTo(TensorOps.GroupedProjectionPlan.ALL);
      assertThat(
              TensorOps.groupedProjectionPlan(
                  GgufTensorType.Q4_K, GgufTensorType.Q4_K, GgufTensorType.Q6_K))
          .isEqualTo(TensorOps.GroupedProjectionPlan.FIRST_SECOND);
      assertThat(
              TensorOps.groupedProjectionPlan(
                  GgufTensorType.Q4_K, GgufTensorType.Q6_K, GgufTensorType.Q4_K))
          .isEqualTo(TensorOps.GroupedProjectionPlan.FIRST_THIRD);
      assertThat(
              TensorOps.groupedProjectionPlan(
                  GgufTensorType.Q6_K, GgufTensorType.Q4_K, GgufTensorType.Q4_K))
          .isEqualTo(TensorOps.GroupedProjectionPlan.SECOND_THIRD);
      assertThat(
              TensorOps.groupedProjectionPlan(
                  GgufTensorType.Q6_K, GgufTensorType.Q6_K, GgufTensorType.Q6_K))
          .isEqualTo(TensorOps.GroupedProjectionPlan.NONE);
      assertThat(
              TensorOps.groupedProjectionPlan(
                  GgufTensorType.Q5_K, GgufTensorType.Q5_K, GgufTensorType.Q5_K))
          .isEqualTo(TensorOps.GroupedProjectionPlan.ALL);
      assertThat(
              TensorOps.groupedProjectionPlan(
                  GgufTensorType.Q5_K, GgufTensorType.Q5_K, GgufTensorType.Q6_K))
          .isEqualTo(TensorOps.GroupedProjectionPlan.FIRST_SECOND);
      assertThat(
              TensorOps.groupedProjectionPlan(
                  GgufTensorType.Q8_0, GgufTensorType.Q8_0, GgufTensorType.Q8_0))
          .as("Q8 triple grouping must remain disabled until its measured regression is resolved")
          .isEqualTo(TensorOps.GroupedProjectionPlan.NONE);
    }

    @Test
    void q4_KDualMatmulMatchesSeparateProjectionsExactly() {
      int cols = 256;
      float[] input = repeatingQuery(cols);
      float[] expectedFirst = new float[1];
      float[] expectedSecond = new float[1];
      float[] actualFirst = new float[1];
      float[] actualSecond = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment firstWeight = copy(arena, q4KBlock(0.125f, 0.0625f, 7));
        MemorySegment secondWeight = copy(arena, q4KBlock(-0.25f, 0.03125f, 3));
        TensorOps.ggufMatmul(expectedFirst, input, firstWeight, GgufTensorType.Q4_K, 1, cols);
        TensorOps.ggufMatmul(expectedSecond, input, secondWeight, GgufTensorType.Q4_K, 1, cols);

        TensorOps.ggufDualMatmul(
            actualFirst,
            firstWeight,
            GgufTensorType.Q4_K,
            1,
            actualSecond,
            secondWeight,
            GgufTensorType.Q4_K,
            1,
            input,
            cols,
            new byte[cols],
            new float[cols / 256],
            new short[cols / 16]);
      }

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
    }

    @Test
    void q4_KTripleMatmulMatchesSeparateProjectionsExactly() {
      int cols = 256;
      float[] input = repeatingQuery(cols);
      float[] expectedFirst = new float[1];
      float[] expectedSecond = new float[1];
      float[] expectedThird = new float[1];
      float[] actualFirst = new float[1];
      float[] actualSecond = new float[1];
      float[] actualThird = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment firstWeight = copy(arena, q4KBlock(0.125f, 0.0625f, 7));
        MemorySegment secondWeight = copy(arena, q4KBlock(-0.25f, 0.03125f, 3));
        MemorySegment thirdWeight = copy(arena, q4KBlock(0.5f, -0.015625f, 11));
        TensorOps.ggufMatmul(expectedFirst, input, firstWeight, GgufTensorType.Q4_K, 1, cols);
        TensorOps.ggufMatmul(expectedSecond, input, secondWeight, GgufTensorType.Q4_K, 1, cols);
        TensorOps.ggufMatmul(expectedThird, input, thirdWeight, GgufTensorType.Q4_K, 1, cols);

        TensorOps.ggufTripleMatmul(
            actualFirst,
            firstWeight,
            GgufTensorType.Q4_K,
            1,
            actualSecond,
            secondWeight,
            GgufTensorType.Q4_K,
            1,
            actualThird,
            thirdWeight,
            GgufTensorType.Q4_K,
            1,
            input,
            cols,
            new byte[cols],
            new float[cols / 256],
            new short[cols / 16]);
      }

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
      assertThat(actualThird).containsExactly(expectedThird);
    }

    @Test
    void q5_KDualMatmulMatchesSeparateProjectionsExactly() {
      int cols = 256;
      float[] input = repeatingQuery(cols);
      int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
      int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
      float[] expectedFirst = new float[1];
      float[] expectedSecond = new float[1];
      float[] actualFirst = new float[1];
      float[] actualSecond = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment firstWeight =
            copy(arena, q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins));
        MemorySegment secondWeight =
            copy(arena, q5KBlock(-0.25f, 0.03125f, i -> 31 - i % 32, scales, mins));
        TensorOps.ggufMatmul(expectedFirst, input, firstWeight, GgufTensorType.Q5_K, 1, cols);
        TensorOps.ggufMatmul(expectedSecond, input, secondWeight, GgufTensorType.Q5_K, 1, cols);

        TensorOps.ggufDualMatmul(
            actualFirst,
            firstWeight,
            GgufTensorType.Q5_K,
            1,
            actualSecond,
            secondWeight,
            GgufTensorType.Q5_K,
            1,
            input,
            cols,
            new byte[cols],
            new float[cols / 256],
            new short[cols / 16]);
      }

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
    }

    @Test
    void q5_KTripleMatmulMatchesSeparateProjectionsExactly() {
      int cols = 256;
      float[] input = repeatingQuery(cols);
      int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
      int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
      float[] expectedFirst = new float[1];
      float[] expectedSecond = new float[1];
      float[] expectedThird = new float[1];
      float[] actualFirst = new float[1];
      float[] actualSecond = new float[1];
      float[] actualThird = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment firstWeight =
            copy(arena, q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins));
        MemorySegment secondWeight =
            copy(arena, q5KBlock(-0.25f, 0.03125f, i -> 31 - i % 32, scales, mins));
        MemorySegment thirdWeight =
            copy(arena, q5KBlock(0.5f, -0.015625f, i -> i * 11 % 32, scales, mins));
        TensorOps.ggufMatmul(expectedFirst, input, firstWeight, GgufTensorType.Q5_K, 1, cols);
        TensorOps.ggufMatmul(expectedSecond, input, secondWeight, GgufTensorType.Q5_K, 1, cols);
        TensorOps.ggufMatmul(expectedThird, input, thirdWeight, GgufTensorType.Q5_K, 1, cols);

        TensorOps.ggufTripleMatmul(
            actualFirst,
            firstWeight,
            GgufTensorType.Q5_K,
            1,
            actualSecond,
            secondWeight,
            GgufTensorType.Q5_K,
            1,
            actualThird,
            thirdWeight,
            GgufTensorType.Q5_K,
            1,
            input,
            cols,
            new byte[cols],
            new float[cols / 256],
            new short[cols / 16]);
      }

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
      assertThat(actualThird).containsExactly(expectedThird);
    }

    @Test
    void q8_0MatchesVectorsGgmlQ8_0ActivationSemantics() {
      float[] x = repeatingQuery(32);
      byte[] row = q8Block(0.25f);
      float[] expected = new float[1];
      float[] actual = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, row);

        VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
            x, qWeight, 1, 32, expected, new byte[32], new float[1]);
        TensorOps.quantizedMatmul(actual, x, qWeight, GgufTensorType.Q8_0, 1, 32);

        assertThat(actual[0]).isCloseTo(expected[0], within(1e-5f));
      }
    }

    @Test
    void q8_0DualMatmulMatchesSeparateProjectionsExactly() {
      int cols = 32;
      float[] input = repeatingQuery(cols);
      float[] expectedFirst = new float[1];
      float[] expectedSecond = new float[1];
      float[] actualFirst = new float[1];
      float[] actualSecond = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment firstWeight = copy(arena, q8Block(0.25f));
        MemorySegment secondWeight = copy(arena, q8Block(-0.125f));
        TensorOps.ggufMatmul(expectedFirst, input, firstWeight, GgufTensorType.Q8_0, 1, cols);
        TensorOps.ggufMatmul(expectedSecond, input, secondWeight, GgufTensorType.Q8_0, 1, cols);

        TensorOps.ggufDualMatmul(
            actualFirst,
            firstWeight,
            GgufTensorType.Q8_0,
            1,
            actualSecond,
            secondWeight,
            GgufTensorType.Q8_0,
            1,
            input,
            cols,
            new byte[cols],
            new float[cols / 32],
            new short[cols / 16]);
      }

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
    }

    @Test
    void q6_KMatchesVectorsGgmlQ8_KActivationSemantics() {
      float[] x = repeatingQuery(256);
      byte[] row = q6KBlock(0.125f, i -> (i % 64) - 32, i -> (i % 7) - 3);
      float[] expected = new float[1];
      float[] actual = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, row);

        VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
            x, qWeight, 1, 256, expected, new byte[256], new float[1]);
        TensorOps.quantizedMatmul(actual, x, qWeight, GgufTensorType.Q6_K, 1, 256);

        assertThat(actual[0]).isCloseTo(expected[0], within(1e-5f));
      }
    }

    @Test
    void q4_KMatchesVectorsGgmlQ8_KActivationSemantics() {
      float[] x = repeatingQuery(256);
      byte[] row = q4KBlock(0.125f, 0.0625f, 7);
      float[] expected = new float[1];
      float[] actual = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, row);

        VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
            x, qWeight, 1, 256, expected, new byte[256], new float[1], new short[16]);
        TensorOps.quantizedMatmul(actual, x, qWeight, GgufTensorType.Q4_K, 1, 256);

        assertThat(actual[0]).isCloseTo(expected[0], within(1e-5f));
      }
    }

    @Test
    void q5_KMatchesVectorsGgmlQ8_KActivationSemantics() {
      float[] x = repeatingQuery(256);
      int[] scales = {1, 2, 3, 4, 5, 6, 7, 8};
      int[] mins = {8, 7, 6, 5, 4, 3, 2, 1};
      byte[] row = q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins);
      float[] expected = new float[1];
      float[] actual = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, row);

        VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
            x, qWeight, 1, 256, expected, new byte[256], new float[1], new short[16]);
        TensorOps.quantizedMatmul(actual, x, qWeight, GgufTensorType.Q5_K, 1, 256);

        assertThat(actual[0]).isCloseTo(expected[0], within(1e-5f));
      }
    }

    @Test
    void q5_0MatchesVectorsGgmlQ8_0ActivationSemantics() {
      float[] x = repeatingQuery(32);
      byte[] row = q5Block(0.25f, i -> (i * 7) % 32 - 16);
      float[] expected = new float[1];
      float[] actual = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, row);

        VectorUtil.ggufQ5_0Q8_0BatchDotProduct(
            x, qWeight, 1, 32, expected, new byte[32], new float[1]);
        TensorOps.quantizedMatmul(actual, x, qWeight, GgufTensorType.Q5_0, 1, 32);

        assertThat(actual[0]).isCloseTo(expected[0], within(1e-5f));
      }
    }

    @Test
    void rejectsNonBlockAlignedQ4_0Dimensions() {
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, q4Block(1.0f));

        assertThatThrownBy(
                () ->
                    TensorOps.quantizedMatmul(
                        new float[1], repeatingQuery(31), qWeight, GgufTensorType.Q4_0, 1, 31))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("multiple of 32");
      }
    }

    @Test
    void rejectsNonBlockAlignedQ8_0Dimensions() {
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, q8Block(1.0f));

        assertThatThrownBy(
                () ->
                    TensorOps.quantizedMatmul(
                        new float[1], repeatingQuery(31), qWeight, GgufTensorType.Q8_0, 1, 31))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("multiple of 32");
      }
    }

    private static float[] repeatingQuery(int length) {
      float[] out = new float[length];
      for (int i = 0; i < length; i++) {
        out[i] = (i % 7) - 3.0f;
      }
      return out;
    }

    private static byte[] q4Block(float scale) {
      byte[] block = new byte[18];
      ByteBuffer.wrap(block)
          .order(ByteOrder.LITTLE_ENDIAN)
          .putShort(0, Float.floatToFloat16(scale));
      for (int i = 0; i < 16; i++) {
        int lo = i & 0x0F;
        int hi = 15 - i;
        block[2 + i] = (byte) (lo | (hi << 4));
      }
      return block;
    }

    private static byte[] q8Block(float scale) {
      byte[] block = new byte[34];
      ByteBuffer.wrap(block)
          .order(ByteOrder.LITTLE_ENDIAN)
          .putShort(0, Float.floatToFloat16(scale));
      for (int i = 0; i < 32; i++) {
        block[2 + i] = (byte) (i - 16);
      }
      return block;
    }

    private static byte[] q4KBlock(float scale, float minScale, int quant) {
      byte[] block = new byte[144];
      ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
      buffer.putShort(0, Float.floatToFloat16(scale));
      buffer.putShort(2, Float.floatToFloat16(minScale));
      for (int group = 0; group < 4; group++) {
        block[4 + group] = 1;
      }
      for (int group = 4; group < 8; group++) {
        block[4 + group + 4] = 1;
      }
      java.util.Arrays.fill(block, 16, block.length, (byte) (quant | (quant << 4)));
      return block;
    }

    private static byte[] q5Block(float scale, IntUnaryOperator quantFactory) {
      byte[] block = new byte[22];
      ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
      buffer.putShort(0, Float.floatToFloat16(scale));
      int highBits = 0;
      for (int index = 0; index < 16; index++) {
        int lowQuant = quantFactory.applyAsInt(index) + 16;
        int highQuant = quantFactory.applyAsInt(index + 16) + 16;
        block[6 + index] = (byte) ((lowQuant & 0x0F) | ((highQuant & 0x0F) << 4));
        highBits |= ((lowQuant >>> 4) & 1) << index;
        highBits |= ((highQuant >>> 4) & 1) << (index + 16);
      }
      buffer.putInt(2, highBits);
      return block;
    }

    private static byte[] q5KBlock(
        float scale, float minScale, IntUnaryOperator quantFactory, int[] scales, int[] mins) {
      byte[] block = new byte[176];
      ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
      buffer.putShort(0, Float.floatToFloat16(scale));
      buffer.putShort(2, Float.floatToFloat16(minScale));

      for (int group = 0; group < 4; group++) {
        block[4 + group] = (byte) scales[group];
        block[8 + group] = (byte) mins[group];
      }
      for (int group = 4; group < 8; group++) {
        block[8 + group] = (byte) ((scales[group] & 0x0F) | ((mins[group] & 0x0F) << 4));
        block[group] |= (byte) ((scales[group] >>> 4) << 6);
        block[4 + group] |= (byte) ((mins[group] >>> 4) << 6);
      }

      for (int group = 0; group < 8; group++) {
        int byteOffset = 48 + (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        for (int index = 0; index < 32; index++) {
          int quant = quantFactory.applyAsInt(group * 32 + index);
          block[byteOffset + index] |= (byte) ((quant & 0x0F) << shift);
          block[16 + index] |= (byte) (((quant >>> 4) & 1) << group);
        }
      }
      return block;
    }

    private static byte[] q6KBlock(
        float scale, IntUnaryOperator quantFactory, IntUnaryOperator scaleFactory) {
      byte[] block = new byte[210];
      for (int i = 0; i < 16; i++) {
        block[192 + i] = (byte) scaleFactory.applyAsInt(i);
      }
      ByteBuffer.wrap(block)
          .order(ByteOrder.LITTLE_ENDIAN)
          .putShort(208, Float.floatToFloat16(scale));

      for (int superBlock = 0; superBlock < 2; superBlock++) {
        int positionBase = superBlock * 128;
        int qlBase = superBlock * 64;
        int qhBase = 128 + superBlock * 32;
        for (int l = 0; l < 32; l++) {
          int q1 = quantFactory.applyAsInt(positionBase + l) + 32;
          int q2 = quantFactory.applyAsInt(positionBase + l + 32) + 32;
          int q3 = quantFactory.applyAsInt(positionBase + l + 64) + 32;
          int q4 = quantFactory.applyAsInt(positionBase + l + 96) + 32;
          block[qlBase + l] = (byte) ((q1 & 0x0F) | ((q3 & 0x0F) << 4));
          block[qlBase + 32 + l] = (byte) ((q2 & 0x0F) | ((q4 & 0x0F) << 4));
          block[qhBase + l] =
              (byte)
                  (((q1 >>> 4) & 0x03)
                      | (((q2 >>> 4) & 0x03) << 2)
                      | (((q3 >>> 4) & 0x03) << 4)
                      | (((q4 >>> 4) & 0x03) << 6));
        }
      }
      return block;
    }

    private static MemorySegment copy(Arena arena, byte[] bytes) {
      MemorySegment segment = arena.allocate(bytes.length);
      MemorySegment.copy(bytes, 0, segment, ValueLayout.JAVA_BYTE, 0, bytes.length);
      return segment;
    }
  }

  @Nested
  class Softmax {

    @Test
    void uniformInputProducesUniform() {
      float[] x = {1.0f, 1.0f, 1.0f, 1.0f};
      TensorOps.softmax(x, 0, 4);

      for (float v : x) {
        assertThat(v).isCloseTo(0.25f, within(1e-5f));
      }
    }

    @Test
    void sumsToOne() {
      float[] x = {1.0f, 2.0f, 3.0f};
      TensorOps.softmax(x, 0, 3);

      float sum = x[0] + x[1] + x[2];
      assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }

    @Test
    void numericallyStableWithLargeValues() {
      float[] x = {1000.0f, 1001.0f, 1002.0f};
      TensorOps.softmax(x, 0, 3);

      float sum = x[0] + x[1] + x[2];
      assertThat(sum).isCloseTo(1.0f, within(1e-5f));
      // All values should be finite
      for (float v : x) {
        assertThat(v).isFinite();
        assertThat(v).isGreaterThan(0.0f);
      }
    }

    @Test
    void respectsOffset() {
      float[] x = {99.0f, 1.0f, 2.0f, 3.0f, 99.0f};
      TensorOps.softmax(x, 1, 3);

      assertThat(x[0]).isEqualTo(99.0f); // untouched
      assertThat(x[4]).isEqualTo(99.0f); // untouched
      float sum = x[1] + x[2] + x[3];
      assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }
  }

  @Nested
  class Rope {

    @Test
    void positionZeroIsIdentity() {
      float[] q = {1.0f, 2.0f, 3.0f, 4.0f};
      float[] k = {5.0f, 6.0f, 7.0f, 8.0f};
      float[] qOrig = q.clone();
      float[] kOrig = k.clone();

      TensorOps.rope(q, k, 0, 4, 10000.0f);

      // At position 0, angle=0, cos=1, sin=0, so values unchanged
      for (int i = 0; i < 4; i++) {
        assertThat(q[i]).isCloseTo(qOrig[i], within(1e-5f));
        assertThat(k[i]).isCloseTo(kOrig[i], within(1e-5f));
      }
    }

    @Test
    void preservesNorm() {
      float[] q = {1.0f, 2.0f, 3.0f, 4.0f};
      float[] k = {5.0f, 6.0f, 7.0f, 8.0f};

      float qNormBefore = norm(q);
      float kNormBefore = norm(k);

      TensorOps.rope(q, k, 7, 4, 10000.0f);

      assertThat(norm(q)).isCloseTo(qNormBefore, within(1e-4f));
      assertThat(norm(k)).isCloseTo(kNormBefore, within(1e-4f));
    }

    @Test
    void offsetAwareRopeMatchesCopiedHeadSemantics() {
      float[] q = {99.0f, 1.0f, 2.0f, 3.0f, 4.0f, 98.0f};
      float[] k = {97.0f, 5.0f, 6.0f, 7.0f, 8.0f, 96.0f};
      float[] expectedQ = {1.0f, 2.0f, 3.0f, 4.0f};
      float[] expectedK = {5.0f, 6.0f, 7.0f, 8.0f};

      TensorOps.rope(expectedQ, expectedK, 7, 4, 10000.0f);
      TensorOps.rope(q, 1, k, 1, 7, 4, 10000.0f);

      assertThat(q[0]).isEqualTo(99.0f);
      assertThat(q[5]).isEqualTo(98.0f);
      assertThat(k[0]).isEqualTo(97.0f);
      assertThat(k[5]).isEqualTo(96.0f);
      for (int i = 0; i < expectedQ.length; i++) {
        assertThat(q[1 + i]).isCloseTo(expectedQ[i], within(1e-5f));
        assertThat(k[1 + i]).isCloseTo(expectedK[i], within(1e-5f));
      }
    }

    @Test
    void offsetAwareSingleVectorRopeMatchesCopiedHeadSemantics() {
      float[] q = {99.0f, 98.0f, 1.0f, 2.0f, 3.0f, 4.0f, 97.0f};
      float[] expectedQ = {1.0f, 2.0f, 3.0f, 4.0f};
      float[] ignoredK = new float[4];

      TensorOps.rope(expectedQ, ignoredK, 9, 4, 10000.0f);
      TensorOps.rope(q, 2, 9, 4, 10000.0f);

      assertThat(q[0]).isEqualTo(99.0f);
      assertThat(q[1]).isEqualTo(98.0f);
      assertThat(q[6]).isEqualTo(97.0f);
      for (int i = 0; i < expectedQ.length; i++) {
        assertThat(q[2 + i]).isCloseTo(expectedQ[i], within(1e-5f));
      }
    }

    @Test
    void neoxLayoutRotatesPairsSeparatedByHalfAHead() {
      float[] vector = {1.0f, 2.0f, 3.0f, 4.0f};
      float cos = (float) Math.cos(1.0);
      float sin = (float) Math.sin(1.0);

      TensorOps.ropeNeox(vector, 0, 1, 4, 10_000.0f);

      assertThat(vector[0]).isCloseTo(1.0f * cos - 3.0f * sin, within(1e-6f));
      assertThat(vector[2]).isCloseTo(1.0f * sin + 3.0f * cos, within(1e-6f));
      float secondCos = (float) Math.cos(0.01);
      float secondSin = (float) Math.sin(0.01);
      assertThat(vector[1]).isCloseTo(2.0f * secondCos - 4.0f * secondSin, within(1e-6f));
      assertThat(vector[3]).isCloseTo(2.0f * secondSin + 4.0f * secondCos, within(1e-6f));
    }

    @Test
    void appliesFrequencyScaleToNormalAndNeoxRope() {
      float[] normal = {1.0f, 2.0f};
      float[] neox = {1.0f, 2.0f};
      float cos = (float) Math.cos(0.5f);
      float sin = (float) Math.sin(0.5f);

      TensorOps.rope(normal, 0, 2, 2, 10_000.0f, 0.25f);
      TensorOps.ropeNeox(neox, 0, 2, 2, 10_000.0f, 0.25f);

      assertThat(normal[0]).isCloseTo(cos - 2.0f * sin, within(1e-6f));
      assertThat(normal[1]).isCloseTo(sin + 2.0f * cos, within(1e-6f));
      assertThat(neox).containsExactly(normal);
    }

    private float norm(float[] v) {
      float sum = 0;
      for (float x : v) sum += x * x;
      return (float) Math.sqrt(sum);
    }
  }

  @Nested
  class SwiGlu {

    @Test
    void knownValues() {
      // silu(x) = x * sigmoid(x)
      // For x=0: silu(0) = 0, so out = 0*up = 0
      float[] gate = {0.0f};
      float[] up = {5.0f};
      float[] out = new float[1];

      TensorOps.swiGlu(out, gate, up, 1);
      assertThat(out[0]).isCloseTo(0.0f, within(1e-5f));
    }

    @Test
    void positiveGateScalesUp() {
      // For large positive gate, silu(x) ≈ x
      float[] gate = {10.0f};
      float[] up = {2.0f};
      float[] out = new float[1];

      TensorOps.swiGlu(out, gate, up, 1);
      // silu(10) ≈ 10, so out ≈ 20
      assertThat(out[0]).isCloseTo(20.0f, within(0.01f));
    }

    @Test
    void operatesOnBatchBufferOffsets() {
      float[] gate = {99.0f, 0.0f, 10.0f, 98.0f};
      float[] up = {97.0f, 5.0f, 2.0f, 96.0f};
      float[] out = {95.0f, 0.0f, 0.0f, 94.0f};

      TensorOps.swiGlu(out, 1, gate, 1, up, 1, 2);

      assertThat(out[0]).isEqualTo(95.0f);
      assertThat(out[1]).isZero();
      assertThat(out[2]).isCloseTo(20.0f, within(0.01f));
      assertThat(out[3]).isEqualTo(94.0f);
    }
  }
}
