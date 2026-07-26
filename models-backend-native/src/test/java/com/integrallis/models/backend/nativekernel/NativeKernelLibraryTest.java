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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class NativeKernelLibraryTest {
  private static final ValueLayout.OfInt LE_INT =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Test
  void exposesVersionedQuantizedKernelCapabilities() {
    try (NativeKernelLibrary kernels = NativeKernelLibrary.open(libraryPath())) {
      assertThat(kernels.abiVersion()).isEqualTo(NativeKernelLibrary.ABI_VERSION);
      assertThat(kernels.supports(NativeKernelCapability.Q4_0_F32_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q5_0_F32_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q5_0_F32_GROUPED_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q8_0_F32_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q8_0_F32_GROUPED_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q4_K_F32_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q4_K_F32_GROUPED_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q5_K_F32_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q5_K_F32_GROUPED_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q6_K_F32_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q6_K_F32_GROUPED_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.MIXED_K_F32_GROUPED_BATCHED_MATMUL))
          .isTrue();
      assertThat(kernels.supports(NativeKernelCapability.PERSISTENT_WORKER_CONTEXT)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.K_QUANT_BATCH_WEIGHT_REUSE)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.Q4_K_BATCH_VECTOR_ACCUMULATION)).isTrue();
    }
  }

  @Test
  void computesQ4_0BatchedMatrixMultiplication() {
    int batchSize = 2;
    int rows = 3;
    int cols = 64;
    float[] input = inputs(batchSize, cols);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];

    try (Arena arena = Arena.ofConfined();
        NativeKernelLibrary kernels = NativeKernelLibrary.open(libraryPath())) {
      MemorySegment weights = arena.allocate(rows * cols / 32L * 18L);
      fillWeights(weights, rows, cols);
      referenceQ4_0F32BatchedMatmul(weights, input, batchSize, rows, cols, expected);

      kernels.q4_0F32BatchedMatmul(weights, input, batchSize, rows, cols, actual);

      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void rejectsInvalidShapesBeforeCrossingTheNativeBoundary() {
    try (Arena arena = Arena.ofConfined();
        NativeKernelLibrary kernels = NativeKernelLibrary.open(libraryPath())) {
      MemorySegment weights = arena.allocate(18);

      assertThatThrownBy(
              () -> kernels.q4_0F32BatchedMatmul(weights, new float[31], 1, 1, 31, new float[1]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
    }
  }

  @Test
  void reusableGgufKernelComputesAcrossWorkspaceGrowth() {
    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      for (int[] shape : new int[][] {{1, 2}, {3, 128}}) {
        int batchSize = shape[0];
        int rows = shape[1];
        int cols = 64;
        float[] input = inputs(batchSize, cols);
        float[] expected = new float[batchSize * rows];
        float[] actual = new float[batchSize * rows];
        MemorySegment weights = arena.allocate(rows * cols / 32L * 18L);
        fillWeights(weights, rows, cols);
        referenceQ4_0F32BatchedMatmul(weights, input, batchSize, rows, cols, expected);

        kernel.multiply(actual, input, weights, GgufTensorType.Q4_0, batchSize, rows, cols);

        assertThat(actual).containsExactly(expected);
      }
      assertThat(kernel.implementation()).isEqualTo("rust-ffm-quantized-v10");
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 1, 2, 64)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 2, 2, 64)).isTrue();
      assertThat(kernel.planRecommendations())
          .containsEntry(PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY, "true")
          .containsEntry(PureJavaPlanConfiguration.MIXED_K_PROJECTIONS_PROPERTY, "true")
          .containsEntry(PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY, "false");
    }
  }

  @Test
  void profiledNativeDecodeMakesSingleTokenProjectionsEligible() {
    try (RustGgufBatchedMatrixKernel kernel =
        RustGgufBatchedMatrixKernel.open(libraryPath(), true)) {
      assertThat(kernel.nativeDecodeEnabled()).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 1, 128, 64)).isTrue();
      assertThat(kernel.isDualEligible(GgufTensorType.Q4_0, 128, GgufTensorType.Q4_0, 128, 1, 64))
          .isTrue();
      assertThat(
              kernel.isTripleEligible(
                  GgufTensorType.Q4_0,
                  128,
                  GgufTensorType.Q4_0,
                  128,
                  GgufTensorType.Q4_0,
                  128,
                  1,
                  64))
          .isTrue();
    }
  }

  @Test
  void profiledSettingsConfigureDecodeGroupingAndWorkerCountTogether() {
    NativeKernelSettings settings = new NativeKernelSettings(true, true, 4);

    try (RustGgufBatchedMatrixKernel kernel =
        RustGgufBatchedMatrixKernel.open(libraryPath(), settings)) {
      assertThat(kernel.nativeDecodeEnabled()).isTrue();
      assertThat(kernel.q5_0GroupedEnabled()).isTrue();
      assertThat(kernel.threadCount()).isEqualTo(4);
    }
  }

  @Test
  void unprofiledQ5_0GroupingIsNotEligible() {
    try (RustGgufBatchedMatrixKernel kernel =
        RustGgufBatchedMatrixKernel.open(libraryPath(), true)) {
      assertThat(kernel.supports(GgufTensorType.Q5_0)).isTrue();
      assertThat(kernel.supportsDual(GgufTensorType.Q5_0, GgufTensorType.Q5_0)).isFalse();
      assertThat(
              kernel.supportsTriple(GgufTensorType.Q5_0, GgufTensorType.Q5_0, GgufTensorType.Q5_0))
          .isFalse();
    }
  }

  @Test
  void reusableGgufKernelComputesExactQ8_0BatchedMatrixMultiplication() {
    int batchSize = 3;
    int rows = 5;
    int cols = 64;
    float[] input = inputs(batchSize, cols);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      MemorySegment weights = arena.allocate(rows * cols / 32L * 34L);
      fillQ8_0Weights(weights, rows, cols);
      referenceQ8_0F32BatchedMatmul(weights, input, batchSize, rows, cols, expected);

      assertThat(kernel.supports(GgufTensorType.Q8_0)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q8_0, batchSize, rows, cols)).isTrue();
      kernel.multiply(actual, input, weights, GgufTensorType.Q8_0, batchSize, rows, cols);

      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void reusableGgufKernelComputesExactQ5_0BatchedMatrixMultiplication() {
    int batchSize = 3;
    int rows = 5;
    int cols = 64;
    float[] input = inputs(batchSize, cols);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      MemorySegment weights = arena.allocate(rows * cols / 32L * 22L);
      fillQ5_0Weights(weights, rows, cols);
      referenceQ5_0F32BatchedMatmul(weights, input, batchSize, rows, cols, expected);

      assertThat(kernel.supports(GgufTensorType.Q5_0)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q5_0, batchSize, rows, cols)).isTrue();
      kernel.multiply(actual, input, weights, GgufTensorType.Q5_0, batchSize, rows, cols);

      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void reusableGgufKernelComputesQ5_0SingleTokenProjectionWithinQuantizedTolerance() {
    int batchSize = 1;
    int rows = 5;
    int cols = 256;
    float[] input = inputs(batchSize, cols);
    float[] expected = new float[rows];
    float[] actual = new float[rows];

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel =
            RustGgufBatchedMatrixKernel.open(libraryPath(), true)) {
      MemorySegment weights = arena.allocate(rows * cols / 32L * 22L);
      fillQ5_0Weights(weights, rows, cols);
      referenceQ5_0F32BatchedMatmul(weights, input, batchSize, rows, cols, expected);

      kernel.multiply(actual, input, weights, GgufTensorType.Q5_0, batchSize, rows, cols);

      for (int index = 0; index < rows; index++) {
        assertThat(actual[index]).as("output[%s]", index).isCloseTo(expected[index], within(1e-3f));
      }
    }
  }

  @Test
  void reusableGgufKernelMatchesVectorSemanticsForKQuantizedBatchedMultiplication() {
    int batchSize = 3;
    int rows = 5;
    int cols = 512;
    float[] input = inputs(batchSize, cols);

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      MemorySegment q4Weights = arena.allocate((long) rows * cols / 256 * 144);
      MemorySegment q5Weights = arena.allocate((long) rows * cols / 256 * 176);
      MemorySegment q6Weights = arena.allocate((long) rows * cols / 256 * 210);
      fillQ4KWeights(q4Weights, rows, cols);
      fillQ5KWeights(q5Weights, rows, cols);
      fillQ6KWeights(q6Weights, rows, cols);

      assertThat(kernel.supports(GgufTensorType.Q4_K)).isTrue();
      assertThat(kernel.supports(GgufTensorType.Q5_K)).isTrue();
      assertThat(kernel.supports(GgufTensorType.Q6_K)).isTrue();
      assertKQuantizedProjectionMatches(
          kernel, q4Weights, GgufTensorType.Q4_K, input, batchSize, rows, cols);
      assertKQuantizedProjectionMatches(
          kernel, q5Weights, GgufTensorType.Q5_K, input, batchSize, rows, cols);
      assertKQuantizedProjectionMatches(
          kernel, q6Weights, GgufTensorType.Q6_K, input, batchSize, rows, cols);
    }
  }

  @Test
  void groupedKernelSharesInputAcrossTwoExactQ4Projections() {
    int batchSize = 3;
    int cols = 64;
    int firstRows = 5;
    int secondRows = 7;
    float[] input = inputs(batchSize, cols);
    float[] expectedFirst = new float[batchSize * firstRows];
    float[] expectedSecond = new float[batchSize * secondRows];
    float[] actualFirst = new float[expectedFirst.length];
    float[] actualSecond = new float[expectedSecond.length];

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      MemorySegment firstWeights = arena.allocate(firstRows * cols / 32L * 18L);
      MemorySegment secondWeights = arena.allocate(secondRows * cols / 32L * 18L);
      fillWeights(firstWeights, firstRows, cols);
      fillWeights(secondWeights, secondRows, cols);
      referenceQ4_0F32BatchedMatmul(firstWeights, input, batchSize, firstRows, cols, expectedFirst);
      referenceQ4_0F32BatchedMatmul(
          secondWeights, input, batchSize, secondRows, cols, expectedSecond);

      assertThat(
              kernel.isDualEligible(
                  GgufTensorType.Q4_0, firstRows, GgufTensorType.Q4_0, secondRows, batchSize, cols))
          .isTrue();
      kernel.multiplyDual(
          actualFirst,
          firstWeights,
          GgufTensorType.Q4_0,
          firstRows,
          actualSecond,
          secondWeights,
          GgufTensorType.Q4_0,
          secondRows,
          input,
          batchSize,
          cols);

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
    }
  }

  @Test
  void groupedKernelSharesInputAcrossTwoExactQ8Projections() {
    int batchSize = 3;
    int cols = 64;
    int firstRows = 5;
    int secondRows = 7;
    float[] input = inputs(batchSize, cols);
    float[] expectedFirst = new float[batchSize * firstRows];
    float[] expectedSecond = new float[batchSize * secondRows];
    float[] actualFirst = new float[expectedFirst.length];
    float[] actualSecond = new float[expectedSecond.length];

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      MemorySegment firstWeights = arena.allocate(firstRows * cols / 32L * 34L);
      MemorySegment secondWeights = arena.allocate(secondRows * cols / 32L * 34L);
      fillQ8_0Weights(firstWeights, firstRows, cols);
      fillQ8_0Weights(secondWeights, secondRows, cols);
      referenceQ8_0F32BatchedMatmul(firstWeights, input, batchSize, firstRows, cols, expectedFirst);
      referenceQ8_0F32BatchedMatmul(
          secondWeights, input, batchSize, secondRows, cols, expectedSecond);

      assertThat(
              kernel.isDualEligible(
                  GgufTensorType.Q8_0, firstRows, GgufTensorType.Q8_0, secondRows, batchSize, cols))
          .isTrue();
      assertThat(
              kernel.isDualEligible(
                  GgufTensorType.Q4_0, firstRows, GgufTensorType.Q8_0, secondRows, batchSize, cols))
          .isFalse();
      kernel.multiplyDual(
          actualFirst,
          firstWeights,
          GgufTensorType.Q8_0,
          firstRows,
          actualSecond,
          secondWeights,
          GgufTensorType.Q8_0,
          secondRows,
          input,
          batchSize,
          cols);

      assertThat(actualFirst).containsExactly(expectedFirst);
      assertThat(actualSecond).containsExactly(expectedSecond);
    }
  }

  @Test
  void groupedKernelSharesInputAcrossThreeExactQ4Projections() {
    int batchSize = 4;
    int cols = 64;
    int[] rowCounts = {3, 5, 7};
    float[] input = inputs(batchSize, cols);

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      MemorySegment[] weights = new MemorySegment[rowCounts.length];
      float[][] expected = new float[rowCounts.length][];
      float[][] actual = new float[rowCounts.length][];
      for (int index = 0; index < rowCounts.length; index++) {
        int rows = rowCounts[index];
        weights[index] = arena.allocate(rows * cols / 32L * 18L);
        fillWeights(weights[index], rows, cols);
        expected[index] = new float[batchSize * rows];
        actual[index] = new float[batchSize * rows];
        referenceQ4_0F32BatchedMatmul(
            weights[index], input, batchSize, rows, cols, expected[index]);
      }

      assertThat(
              kernel.isTripleEligible(
                  GgufTensorType.Q4_0,
                  rowCounts[0],
                  GgufTensorType.Q4_0,
                  rowCounts[1],
                  GgufTensorType.Q4_0,
                  rowCounts[2],
                  batchSize,
                  cols))
          .isTrue();
      kernel.multiplyTriple(
          actual[0],
          weights[0],
          GgufTensorType.Q4_0,
          rowCounts[0],
          actual[1],
          weights[1],
          GgufTensorType.Q4_0,
          rowCounts[1],
          actual[2],
          weights[2],
          GgufTensorType.Q4_0,
          rowCounts[2],
          input,
          batchSize,
          cols);

      assertThat(actual[0]).containsExactly(expected[0]);
      assertThat(actual[1]).containsExactly(expected[1]);
      assertThat(actual[2]).containsExactly(expected[2]);
    }
  }

  @Test
  void groupedKernelSharesInputAcrossThreeExactQ5_0Projections() {
    int batchSize = 4;
    int cols = 64;
    int[] rowCounts = {3, 5, 7};
    float[] input = inputs(batchSize, cols);

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel =
            RustGgufBatchedMatrixKernel.open(libraryPath(), false, true)) {
      MemorySegment[] weights = new MemorySegment[rowCounts.length];
      float[][] expected = new float[rowCounts.length][];
      float[][] actual = new float[rowCounts.length][];
      for (int index = 0; index < rowCounts.length; index++) {
        int rows = rowCounts[index];
        weights[index] = arena.allocate(rows * cols / 32L * 22L);
        fillQ5_0Weights(weights[index], rows, cols);
        expected[index] = new float[batchSize * rows];
        actual[index] = new float[batchSize * rows];
        referenceQ5_0F32BatchedMatmul(
            weights[index], input, batchSize, rows, cols, expected[index]);
      }

      assertThat(
              kernel.isTripleEligible(
                  GgufTensorType.Q5_0,
                  rowCounts[0],
                  GgufTensorType.Q5_0,
                  rowCounts[1],
                  GgufTensorType.Q5_0,
                  rowCounts[2],
                  batchSize,
                  cols))
          .isTrue();
      kernel.multiplyTriple(
          actual[0],
          weights[0],
          GgufTensorType.Q5_0,
          rowCounts[0],
          actual[1],
          weights[1],
          GgufTensorType.Q5_0,
          rowCounts[1],
          actual[2],
          weights[2],
          GgufTensorType.Q5_0,
          rowCounts[2],
          input,
          batchSize,
          cols);

      assertThat(actual[0]).containsExactly(expected[0]);
      assertThat(actual[1]).containsExactly(expected[1]);
      assertThat(actual[2]).containsExactly(expected[2]);
    }
  }

  @Test
  void groupedKernelMatchesVectorSemanticsForMixedQ4_KQ5_KQ6_KProjections() {
    int batchSize = 3;
    int cols = 512;
    int[] rowCounts = {3, 5, 7};
    float[] input = inputs(batchSize, cols);

    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      MemorySegment firstWeights = arena.allocate((long) rowCounts[0] * cols / 256 * 144);
      MemorySegment secondWeights = arena.allocate((long) rowCounts[1] * cols / 256 * 176);
      MemorySegment thirdWeights = arena.allocate((long) rowCounts[2] * cols / 256 * 210);
      fillQ4KWeights(firstWeights, rowCounts[0], cols);
      fillQ5KWeights(secondWeights, rowCounts[1], cols);
      fillQ6KWeights(thirdWeights, rowCounts[2], cols);

      float[] expectedFirst =
          tensorOpsReference(
              firstWeights, GgufTensorType.Q4_K, input, batchSize, rowCounts[0], cols);
      float[] expectedSecond =
          tensorOpsReference(
              secondWeights, GgufTensorType.Q5_K, input, batchSize, rowCounts[1], cols);
      float[] expectedThird =
          tensorOpsReference(
              thirdWeights, GgufTensorType.Q6_K, input, batchSize, rowCounts[2], cols);
      float[] actualFirst = new float[expectedFirst.length];
      float[] actualSecond = new float[expectedSecond.length];
      float[] actualThird = new float[expectedThird.length];

      assertThat(
              kernel.isTripleEligible(
                  GgufTensorType.Q4_K,
                  rowCounts[0],
                  GgufTensorType.Q5_K,
                  rowCounts[1],
                  GgufTensorType.Q6_K,
                  rowCounts[2],
                  batchSize,
                  cols))
          .isTrue();
      kernel.multiplyTriple(
          actualFirst,
          firstWeights,
          GgufTensorType.Q4_K,
          rowCounts[0],
          actualSecond,
          secondWeights,
          GgufTensorType.Q5_K,
          rowCounts[1],
          actualThird,
          thirdWeights,
          GgufTensorType.Q6_K,
          rowCounts[2],
          input,
          batchSize,
          cols);

      assertClose(actualFirst, expectedFirst);
      assertClose(actualSecond, expectedSecond);
      assertClose(actualThird, expectedThird);
    }
  }

  private static Path libraryPath() {
    return Path.of(System.getProperty("models.native.kernels.library"));
  }

  private static float[] inputs(int batchSize, int cols) {
    float[] input = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        input[batch * cols + col] = ((col * 17 + batch * 11) % 29 - 14) * 0.125f;
      }
    }
    return input;
  }

  private static void fillWeights(MemorySegment weights, int rows, int cols) {
    int blocksPerRow = cols / 32;
    for (int row = 0; row < rows; row++) {
      for (int block = 0; block < blocksPerRow; block++) {
        long offset = ((long) row * blocksPerRow + block) * 18;
        float scale = (row + block + 1) * 0.125f;
        weights.set(LE_SHORT, offset, Float.floatToFloat16(scale));
        for (int lane = 0; lane < 16; lane++) {
          int low = (row * 3 + block * 5 + lane) & 0xf;
          int high = (row * 7 + block * 2 + lane * 3) & 0xf;
          weights.set(ValueLayout.JAVA_BYTE, offset + 2 + lane, (byte) (low | high << 4));
        }
      }
    }
  }

  private static void fillQ8_0Weights(MemorySegment weights, int rows, int cols) {
    int blocksPerRow = cols / 32;
    for (int row = 0; row < rows; row++) {
      for (int block = 0; block < blocksPerRow; block++) {
        long offset = ((long) row * blocksPerRow + block) * 34;
        float scale = (row + block + 1) * 0.03125f;
        weights.set(LE_SHORT, offset, Float.floatToFloat16(scale));
        for (int lane = 0; lane < 32; lane++) {
          int quantized = ((row * 37 + block * 19 + lane * 11) & 0xff) - 128;
          weights.set(ValueLayout.JAVA_BYTE, offset + 2 + lane, (byte) quantized);
        }
      }
    }
  }

  private static void fillQ5_0Weights(MemorySegment weights, int rows, int cols) {
    int blocksPerRow = cols / 32;
    for (int row = 0; row < rows; row++) {
      for (int block = 0; block < blocksPerRow; block++) {
        long offset = ((long) row * blocksPerRow + block) * 22;
        float scale = (row + block + 1) * 0.0625f;
        int highBits = 0;
        weights.set(LE_SHORT, offset, Float.floatToFloat16(scale));
        for (int lane = 0; lane < 16; lane++) {
          int low = (row * 17 + block * 11 + lane * 5) & 0x1f;
          int high = (row * 13 + block * 7 + lane * 9) & 0x1f;
          weights.set(
              ValueLayout.JAVA_BYTE,
              offset + 6 + lane,
              (byte) ((low & 0x0f) | ((high & 0x0f) << 4)));
          highBits |= ((low >>> 4) & 1) << lane;
          highBits |= ((high >>> 4) & 1) << (lane + 16);
        }
        weights.set(LE_INT, offset + 2, highBits);
      }
    }
  }

  private static void fillQ4KWeights(MemorySegment weights, int rows, int cols) {
    int blocksPerRow = cols / 256;
    for (int row = 0; row < rows; row++) {
      for (int block = 0; block < blocksPerRow; block++) {
        byte[] bytes = q4KBlock((row + 1) * 0.03125f, (block + 1) * 0.015625f, row, block);
        MemorySegment.copy(
            bytes,
            0,
            weights,
            ValueLayout.JAVA_BYTE,
            ((long) row * blocksPerRow + block) * bytes.length,
            bytes.length);
      }
    }
  }

  private static byte[] q4KBlock(float scale, float minScale, int row, int block) {
    byte[] bytes = new byte[144];
    ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putShort(0, Float.floatToFloat16(scale));
    buffer.putShort(2, Float.floatToFloat16(minScale));
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    for (int group = 0; group < 4; group++) {
      bytes[4 + group] = (byte) scales[group];
      bytes[8 + group] = (byte) mins[group];
    }
    for (int group = 4; group < 8; group++) {
      bytes[8 + group] = (byte) ((scales[group] & 0x0f) | ((mins[group] & 0x0f) << 4));
      bytes[group] |= (byte) ((scales[group] >>> 4) << 6);
      bytes[4 + group] |= (byte) ((mins[group] >>> 4) << 6);
    }
    for (int group = 0; group < 8; group++) {
      int packedOffset = 16 + (group >>> 1) * 32;
      int shift = (group & 1) * 4;
      for (int index = 0; index < 32; index++) {
        int quant = (row * 13 + block * 7 + group * 5 + index * 3) & 0x0f;
        bytes[packedOffset + index] |= (byte) (quant << shift);
      }
    }
    return bytes;
  }

  private static void fillQ6KWeights(MemorySegment weights, int rows, int cols) {
    int blocksPerRow = cols / 256;
    for (int row = 0; row < rows; row++) {
      for (int block = 0; block < blocksPerRow; block++) {
        byte[] bytes = q6KBlock((row + block + 1) * 0.015625f, row, block);
        MemorySegment.copy(
            bytes,
            0,
            weights,
            ValueLayout.JAVA_BYTE,
            ((long) row * blocksPerRow + block) * bytes.length,
            bytes.length);
      }
    }
  }

  private static void fillQ5KWeights(MemorySegment weights, int rows, int cols) {
    int blocksPerRow = cols / 256;
    for (int row = 0; row < rows; row++) {
      for (int block = 0; block < blocksPerRow; block++) {
        byte[] bytes = q5KBlock((row + 1) * 0.03125f, (block + 1) * 0.015625f, row, block);
        MemorySegment.copy(
            bytes,
            0,
            weights,
            ValueLayout.JAVA_BYTE,
            ((long) row * blocksPerRow + block) * bytes.length,
            bytes.length);
      }
    }
  }

  private static byte[] q5KBlock(float scale, float minScale, int row, int block) {
    byte[] bytes = new byte[176];
    ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putShort(0, Float.floatToFloat16(scale));
    buffer.putShort(2, Float.floatToFloat16(minScale));
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    for (int group = 0; group < 4; group++) {
      bytes[4 + group] = (byte) scales[group];
      bytes[8 + group] = (byte) mins[group];
    }
    for (int group = 4; group < 8; group++) {
      bytes[8 + group] = (byte) ((scales[group] & 0x0f) | ((mins[group] & 0x0f) << 4));
      bytes[group] |= (byte) ((scales[group] >>> 4) << 6);
      bytes[4 + group] |= (byte) ((mins[group] >>> 4) << 6);
    }
    for (int group = 0; group < 8; group++) {
      int packedOffset = 48 + (group >>> 1) * 32;
      int shift = (group & 1) * 4;
      int highBit = 1 << group;
      for (int index = 0; index < 32; index++) {
        int quant = (row * 17 + block * 11 + group * 7 + index * 5) & 0x1f;
        bytes[packedOffset + index] |= (byte) ((quant & 0x0f) << shift);
        if ((quant & 0x10) != 0) {
          bytes[16 + index] |= (byte) highBit;
        }
      }
    }
    return bytes;
  }

  private static byte[] q6KBlock(float scale, int row, int block) {
    byte[] bytes = new byte[210];
    for (int index = 0; index < 16; index++) {
      bytes[192 + index] = (byte) ((row * 5 + block * 3 + index * 7) % 17 - 8);
    }
    ByteBuffer.wrap(bytes)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putShort(208, Float.floatToFloat16(scale));
    for (int superBlock = 0; superBlock < 2; superBlock++) {
      int positionBase = superBlock * 128;
      int qlBase = superBlock * 64;
      int qhBase = 128 + superBlock * 32;
      for (int index = 0; index < 32; index++) {
        int q1 = q6Quant(row, block, positionBase + index) + 32;
        int q2 = q6Quant(row, block, positionBase + index + 32) + 32;
        int q3 = q6Quant(row, block, positionBase + index + 64) + 32;
        int q4 = q6Quant(row, block, positionBase + index + 96) + 32;
        bytes[qlBase + index] = (byte) ((q1 & 0x0f) | ((q3 & 0x0f) << 4));
        bytes[qlBase + 32 + index] = (byte) ((q2 & 0x0f) | ((q4 & 0x0f) << 4));
        bytes[qhBase + index] =
            (byte)
                (((q1 >>> 4) & 0x03)
                    | (((q2 >>> 4) & 0x03) << 2)
                    | (((q3 >>> 4) & 0x03) << 4)
                    | (((q4 >>> 4) & 0x03) << 6));
      }
    }
    return bytes;
  }

  private static int q6Quant(int row, int block, int index) {
    return (row * 19 + block * 11 + index * 7) % 64 - 32;
  }

  private static void assertKQuantizedProjectionMatches(
      RustGgufBatchedMatrixKernel kernel,
      MemorySegment weights,
      GgufTensorType type,
      float[] input,
      int batchSize,
      int rows,
      int cols) {
    float[] expected = tensorOpsReference(weights, type, input, batchSize, rows, cols);
    float[] actual = new float[expected.length];
    assertThat(kernel.isEligible(type, batchSize, rows, cols)).isTrue();
    kernel.multiply(actual, input, weights, type, batchSize, rows, cols);
    assertClose(actual, expected);
  }

  private static float[] tensorOpsReference(
      MemorySegment weights,
      GgufTensorType type,
      float[] input,
      int batchSize,
      int rows,
      int cols) {
    float[] output = new float[batchSize * rows];
    for (int batch = 0; batch < batchSize; batch++) {
      float[] query = Arrays.copyOfRange(input, batch * cols, (batch + 1) * cols);
      float[] projection = new float[rows];
      TensorOps.ggufMatmul(projection, query, weights, type, rows, cols);
      System.arraycopy(projection, 0, output, batch * rows, rows);
    }
    return output;
  }

  private static void assertClose(float[] actual, float[] expected) {
    assertThat(actual).hasSameSizeAs(expected);
    for (int index = 0; index < actual.length; index++) {
      assertThat(actual[index]).as("output[%s]", index).isCloseTo(expected[index], within(1e-4f));
    }
  }

  private static void referenceQ4_0F32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    int blocksPerRow = cols / 32;
    byte[] quantized = new byte[batchSize * cols];
    float[] scales = new float[batchSize * blocksPerRow];
    quantizeInputs(input, batchSize, cols, quantized, scales);

    for (int batch = 0; batch < batchSize; batch++) {
      for (int row = 0; row < rows; row++) {
        float sum = 0;
        for (int block = 0; block < blocksPerRow; block++) {
          long weightOffset = ((long) row * blocksPerRow + block) * 18;
          float weightScale = Float.float16ToFloat(weights.get(LE_SHORT, weightOffset));
          int inputOffset = batch * cols + block * 32;
          int integerSum = 0;
          for (int lane = 0; lane < 16; lane++) {
            int packed =
                Byte.toUnsignedInt(weights.get(ValueLayout.JAVA_BYTE, weightOffset + 2 + lane));
            integerSum += ((packed & 0xf) - 8) * quantized[inputOffset + lane];
            integerSum += ((packed >>> 4) - 8) * quantized[inputOffset + lane + 16];
          }
          float scale = weightScale * scales[batch * blocksPerRow + block];
          sum = Math.fma(scale, integerSum, sum);
        }
        output[batch * rows + row] = sum;
      }
    }
  }

  private static void referenceQ8_0F32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    int blocksPerRow = cols / 32;
    byte[] quantized = new byte[batchSize * cols];
    float[] scales = new float[batchSize * blocksPerRow];
    quantizeInputs(input, batchSize, cols, quantized, scales);

    for (int batch = 0; batch < batchSize; batch++) {
      for (int row = 0; row < rows; row++) {
        float sum = 0;
        for (int block = 0; block < blocksPerRow; block++) {
          long weightOffset = ((long) row * blocksPerRow + block) * 34;
          float weightScale = Float.float16ToFloat(weights.get(LE_SHORT, weightOffset));
          int inputOffset = batch * cols + block * 32;
          int integerSum = 0;
          for (int lane = 0; lane < 32; lane++) {
            integerSum +=
                weights.get(ValueLayout.JAVA_BYTE, weightOffset + 2 + lane)
                    * quantized[inputOffset + lane];
          }
          float scale = weightScale * scales[batch * blocksPerRow + block];
          sum = Math.fma(scale, integerSum, sum);
        }
        output[batch * rows + row] = sum;
      }
    }
  }

  private static void referenceQ5_0F32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    int blocksPerRow = cols / 32;
    byte[] quantized = new byte[batchSize * cols];
    float[] scales = new float[batchSize * blocksPerRow];
    quantizeInputs(input, batchSize, cols, quantized, scales);

    for (int batch = 0; batch < batchSize; batch++) {
      for (int row = 0; row < rows; row++) {
        float sum = 0;
        for (int block = 0; block < blocksPerRow; block++) {
          long weightOffset = ((long) row * blocksPerRow + block) * 22;
          float weightScale = Float.float16ToFloat(weights.get(LE_SHORT, weightOffset));
          int highBits = weights.get(LE_INT, weightOffset + 2);
          int inputOffset = batch * cols + block * 32;
          int integerSum = 0;
          for (int lane = 0; lane < 16; lane++) {
            int packed =
                Byte.toUnsignedInt(weights.get(ValueLayout.JAVA_BYTE, weightOffset + 6 + lane));
            int low = ((packed & 0x0f) | (((highBits >>> lane) & 1) << 4)) - 16;
            int high = ((packed >>> 4) | (((highBits >>> (lane + 16)) & 1) << 4)) - 16;
            integerSum += low * quantized[inputOffset + lane];
            integerSum += high * quantized[inputOffset + lane + 16];
          }
          float scale = weightScale * scales[batch * blocksPerRow + block];
          sum = Math.fma(scale, integerSum, sum);
        }
        output[batch * rows + row] = sum;
      }
    }
  }

  private static void quantizeInputs(
      float[] input, int batchSize, int cols, byte[] quantized, float[] scales) {
    int blocksPerRow = cols / 32;
    for (int batch = 0; batch < batchSize; batch++) {
      for (int block = 0; block < blocksPerRow; block++) {
        int inputOffset = batch * cols + block * 32;
        float absoluteMax = 0;
        for (int lane = 0; lane < 32; lane++) {
          absoluteMax = Math.max(absoluteMax, Math.abs(input[inputOffset + lane]));
        }
        float inverseScale = absoluteMax == 0 ? 0 : 127.0f / absoluteMax;
        scales[batch * blocksPerRow + block] =
            Float.float16ToFloat(Float.floatToFloat16(absoluteMax / 127.0f));
        for (int lane = 0; lane < 32; lane++) {
          quantized[inputOffset + lane] =
              (byte) ggmlNearestInt(input[inputOffset + lane] * inverseScale);
        }
      }
    }
  }

  private static int ggmlNearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007f_ffff) - 0x0040_0000;
  }
}
