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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class NativeKernelLibraryTest {
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Test
  void exposesVersionedQ4KernelCapability() {
    try (NativeKernelLibrary kernels = NativeKernelLibrary.open(libraryPath())) {
      assertThat(kernels.abiVersion()).isEqualTo(NativeKernelLibrary.ABI_VERSION);
      assertThat(kernels.supports(NativeKernelCapability.Q4_0_F32_BATCHED_MATMUL)).isTrue();
      assertThat(kernels.supports(NativeKernelCapability.PERSISTENT_WORKER_CONTEXT)).isTrue();
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
      assertThat(kernel.implementation()).isEqualTo("rust-ffm-q4_0-v1");
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 1, 2, 64)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 2, 2, 64)).isTrue();
      assertThat(kernel.planRecommendations())
          .containsEntry(PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY, "true")
          .containsEntry(PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY, "false");
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

  private static void referenceQ4_0F32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    int blocksPerRow = cols / 32;
    byte[] quantized = new byte[batchSize * cols];
    float[] scales = new float[batchSize * blocksPerRow];
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

  private static int ggmlNearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007f_ffff) - 0x0040_0000;
  }
}
