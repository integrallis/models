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
import static org.mockito.Mockito.mockStatic;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.BFloat16Matrix;
import com.integrallis.vectors.core.VectorRuntimeCapabilities;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

class BFloat16NativeKernelTest {
  private static final int ROWS = 48;
  private static final int COLS = 5120;
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Test
  void actualProviderMatchesJavaAndPreservesCallerBuffers() {
    Random random = new Random(987223);
    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      MemorySegment weights = arena.allocate((long) ROWS * COLS * 2, 8);
      for (int i = 0; i < ROWS * COLS; i++) {
        weights.set(
            LE_SHORT,
            (long) i * 2,
            (short) (Float.floatToRawIntBits(random.nextFloat() * .2f - .1f) >>> 16));
      }
      byte[] originalWeights = weights.toArray(ValueLayout.JAVA_BYTE);
      BFloat16Matrix reference = BFloat16Matrix.of(weights, ROWS, COLS);
      assertThat(kernel.isEligible(GgufTensorType.BF16, 1, ROWS, COLS)).isTrue();
      for (int batch : new int[] {1, 2, 3, 8, 26, 32}) {
        float[] input = new float[batch * COLS + 3];
        for (int i = 0; i < input.length; i++) input[i] = random.nextFloat() * 4 - 2;
        float[] originalInput = input.clone();
        float[] expected = new float[batch * ROWS + 5];
        float[] actual = new float[expected.length];
        Arrays.fill(expected, 42);
        Arrays.fill(actual, 42);
        reference.multiplyBatch(input, 0, batch, expected, 0);
        // Cold and repeated calls exercise the actual SPI downcall, not a mock.
        for (int repeat = 0; repeat < 4; repeat++) {
          kernel.multiply(actual, input, weights, GgufTensorType.BF16, batch, ROWS, COLS);
          for (int i = 0; i < actual.length; i++)
            assertThat(Float.floatToRawIntBits(actual[i]))
                .as("batch %s repeat %s output %s", batch, repeat, i)
                .isEqualTo(Float.floatToRawIntBits(expected[i]));
        }
        assertThat(input).containsExactly(originalInput);
      }
      assertThat(weights.toArray(ValueLayout.JAVA_BYTE)).containsExactly(originalWeights);
      assertThat(kernel.isEligible(GgufTensorType.BF16, 33, ROWS, COLS)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.BF16, 1, ROWS + 1, COLS)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.BF16, 1, ROWS, COLS - 1)).isFalse();
    }
  }

  @Test
  void rejectsInvalidBuffersAndClosedLibraryBeforeCallingNativeCode() {
    try (Arena arena = Arena.ofConfined();
        NativeKernelLibrary library = NativeKernelLibrary.open(libraryPath())) {
      MemorySegment weights = arena.allocate((long) ROWS * COLS * 2, 8);
      float[] input = new float[COLS];
      float[] output = new float[ROWS];
      assertThatThrownBy(
              () -> library.bfloat16F32BatchedMatmul(weights, input, input, 1, ROWS, COLS, 4, true))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(
              () ->
                  library.bfloat16F32BatchedMatmul(
                      weights, new float[COLS - 1], output, 1, ROWS, COLS, 4, true))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(
              () ->
                  library.bfloat16F32BatchedMatmul(
                      weights.asSlice(0, weights.byteSize() - 1),
                      input,
                      output,
                      1,
                      ROWS,
                      COLS,
                      4,
                      true))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(
              () ->
                  library.bfloat16F32BatchedMatmul(
                      MemorySegment.ofArray(new byte[ROWS * COLS * 2]),
                      input,
                      output,
                      1,
                      ROWS,
                      COLS,
                      4,
                      true))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(
              () ->
                  library.bfloat16F32BatchedMatmul(weights, input, output, 1, ROWS, COLS, 16, true))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void rejectsUseAfterClose() {
    NativeKernelLibrary library = NativeKernelLibrary.open(libraryPath());
    library.close();
    assertThatThrownBy(
            () ->
                library.bfloat16F32BatchedMatmul(
                    MemorySegment.NULL, new float[COLS], new float[ROWS], 1, ROWS, COLS, 4, true))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("closed");
  }

  @Test
  void scalarAndUnsupportedVectorWidthsStayOnJava() {
    for (int bits : new int[] {0, 64, 512}) {
      var capabilities =
          new VectorRuntimeCapabilities(
              "test", bits != 0, 256, 256, bits, true, true, false, false, false, false, "test", 1,
              1);
      try (var vectors = mockStatic(VectorUtil.class)) {
        vectors.when(VectorUtil::runtimeCapabilities).thenReturn(capabilities);
        try (var kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
          assertThat(kernel.supports(GgufTensorType.BF16)).isFalse();
          assertThat(kernel.isEligible(GgufTensorType.BF16, 1, ROWS, COLS)).isFalse();
        }
      }
    }
  }

  private static Path libraryPath() {
    return Path.of(System.getProperty("models.native.kernels.library"));
  }
}
