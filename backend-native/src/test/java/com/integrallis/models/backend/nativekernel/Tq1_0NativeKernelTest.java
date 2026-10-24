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
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The TQ1_0 x Q8_K projection, against an independent Java transcription of ggml's.
 *
 * <p>This exists because our previous TQ1_0 route kept activations in F32 while ggml quantizes them
 * to Q8_K, so the arithmetic was never the reference's and maple's output drifted from llama.cpp
 * from the first layer. Matching the reference is the requirement here, which makes "close" a
 * failure: the assertions are exact.
 */
@Tag("unit")
class Tq1_0NativeKernelTest {

  private static final int BLOCK_BYTES = 54;
  private static final int BLOCK_ELEMENTS = 256;

  @Test
  void ternaryProjectionMatchesTheGgmlReferenceExactly() {
    Path library = Path.of(System.getProperty("models.native.kernels.library"));
    // Two implementations that both return zeros agree perfectly and prove nothing. Tracked across
    // the whole sweep rather than per case, because the scale list below deliberately includes an
    // f16 zero, and a single-block row carrying it is legitimately all zeros.
    boolean producedNonZero = false;
    // Both worker counts: a batched dispatch must not depend on how rows were partitioned.
    for (int threads : new int[] {1, 4}) {
      try (var kernel = RustGgufBatchedMatrixKernel.open(library, threads);
          var arena = Arena.ofShared()) {
        assertThat(kernel.supports(GgufTensorType.TQ1_0)).isTrue();

        var random = new Random(34_000);
        for (int cols : new int[] {256, 2048}) {
          // Large then one then large again, so every worker has to wake after a single-row call.
          for (int rows : new int[] {17, 1, 33}) {
            byte[] packed = new byte[rows * cols / BLOCK_ELEMENTS * BLOCK_BYTES];
            random.nextBytes(packed);
            // Base-3 bytes only reach 242; above that the trits are meaningless, and a scale of
            // zero, one, and both signs all have to be exercised.
            int[] scaleBits = {0, 1, 0x3000, 0x3c00, 0xbc00, 0x1400};
            for (int block = 0; block < packed.length / BLOCK_BYTES; block++) {
              int base = block * BLOCK_BYTES;
              for (int byteIndex = 0; byteIndex < 48; byteIndex++) {
                packed[base + byteIndex] = (byte) ((packed[base + byteIndex] & 0xFF) % 243);
              }
              for (int byteIndex = 48; byteIndex < 52; byteIndex++) {
                packed[base + byteIndex] = (byte) ((packed[base + byteIndex] & 0xFF) % 81);
              }
              int scale = scaleBits[block % scaleBits.length];
              packed[base + 52] = (byte) scale;
              packed[base + 53] = (byte) (scale >>> 8);
            }
            // Offset slice: the kernel must not assume the weights start at a segment boundary.
            MemorySegment weights = arena.allocate(packed.length + 13).asSlice(13);
            weights.copyFrom(MemorySegment.ofArray(packed));

            for (int batch : new int[] {1, 3, 8}) {
              float[] input = new float[batch * cols];
              for (int index = 0; index < input.length; index++) {
                input[index] =
                    switch (index % 101) {
                      // Q8_K picks its scale from the signed extremum, so a block whose largest
                      // magnitude is negative takes a different sign path than one whose is
                      // positive. Both have to appear, along with zeros and a flat block.
                      case 0 -> 0.0f;
                      case 1 -> -0.0f;
                      case 2 -> 7.5f;
                      case 3 -> -7.5f;
                      case 4 -> Float.MIN_NORMAL;
                      default -> random.nextFloat() * 2.0f - 1.0f;
                    };
              }

              float[] actual = new float[batch * rows];
              kernel.multiply(actual, input, weights, GgufTensorType.TQ1_0, batch, rows, cols);
              float[] expected = Tq1_0Q8KOracle.multiply(weights, input, batch, rows, cols);

              for (float value : expected) {
                if (value != 0.0f) {
                  producedNonZero = true;
                  break;
                }
              }

              for (int index = 0; index < actual.length; index++) {
                assertThat(actual[index])
                    .as(
                        "threads=%d cols=%d rows=%d batch=%d index=%d",
                        threads, cols, rows, batch, index)
                    .isEqualTo(expected[index]);
              }
            }
          }
        }
      }
    }
    assertThat(producedNonZero)
        .as("every case produced zeros, so the comparison proved nothing")
        .isTrue();
  }

  @Test
  void anAllZeroActivationBlockProducesZero() {
    // Q8_K writes d = 0 and zero quants when the block has no magnitude, so the product must be
    // exactly zero rather than a denormal left over from the scale reciprocal.
    Path library = Path.of(System.getProperty("models.native.kernels.library"));
    try (var kernel = RustGgufBatchedMatrixKernel.open(library, 1);
        var arena = Arena.ofShared()) {
      int cols = 256;
      int rows = 4;
      byte[] packed = new byte[rows * cols / BLOCK_ELEMENTS * BLOCK_BYTES];
      new Random(7).nextBytes(packed);
      for (int block = 0; block < packed.length / BLOCK_BYTES; block++) {
        packed[block * BLOCK_BYTES + 52] = 0x00;
        packed[block * BLOCK_BYTES + 53] = 0x3c; // 1.0
      }
      MemorySegment weights = arena.allocate(packed.length);
      weights.copyFrom(MemorySegment.ofArray(packed));

      float[] actual = new float[rows];
      kernel.multiply(actual, new float[cols], weights, GgufTensorType.TQ1_0, 1, rows, cols);

      assertThat(actual).containsOnly(0.0f);
    }
  }

  @Test
  void theHeapOverloadAgreesWithTheReferenceToo() {
    // kernel.multiply() reaches the MemorySegment overload. The public float[] entry point is a
    // separate code path with its own staging, so it gets its own comparison rather than being
    // assumed equivalent.
    int cols = 256;
    int rows = 5;
    try (NativeKernelLibrary kernels = NativeKernelLibrary.open(libraryPath(), 2);
        var arena = Arena.ofShared()) {
      assertThat(kernels.supports(NativeKernelCapability.TQ1_0_Q8_K_BATCHED_MATMUL)).isTrue();

      byte[] packed = new byte[rows * cols / BLOCK_ELEMENTS * BLOCK_BYTES];
      var random = new Random(991);
      random.nextBytes(packed);
      for (int block = 0; block < packed.length / BLOCK_BYTES; block++) {
        int base = block * BLOCK_BYTES;
        for (int byteIndex = 0; byteIndex < 48; byteIndex++) {
          packed[base + byteIndex] = (byte) ((packed[base + byteIndex] & 0xFF) % 243);
        }
        for (int byteIndex = 48; byteIndex < 52; byteIndex++) {
          packed[base + byteIndex] = (byte) ((packed[base + byteIndex] & 0xFF) % 81);
        }
        packed[base + 52] = 0x00;
        packed[base + 53] = 0x3c; // 1.0
      }
      MemorySegment weights = arena.allocate(packed.length);
      weights.copyFrom(MemorySegment.ofArray(packed));

      int batch = 3;
      float[] input = new float[batch * cols];
      for (int index = 0; index < input.length; index++) {
        input[index] = random.nextFloat() * 2.0f - 1.0f;
      }
      float[] actual = new float[batch * rows];
      kernels.tq1_0F32BatchedMatmul(weights, input, batch, rows, cols, actual);
      float[] expected = Tq1_0Q8KOracle.multiply(weights, input, batch, rows, cols);

      assertThat(actual).containsExactly(expected);
      boolean producedNonZero = false;
      for (float value : actual) {
        if (value != 0.0f) {
          producedNonZero = true;
          break;
        }
      }
      assertThat(producedNonZero).as("all-zero output would make this comparison empty").isTrue();
    }
  }

  @Test
  void refusesWeightsTooSmallForTheDeclaredShape() {
    // 54 bytes per 256-element block. A segment short of that must be rejected rather than read
    // past its end: the kernel reads the weights as raw memory, so a wrong block size here is an
    // out-of-bounds read and not a wrong answer.
    try (var kernel = RustGgufBatchedMatrixKernel.open(libraryPath(), 1);
        var arena = Arena.ofShared()) {
      int cols = 256;
      int rows = 4;
      long needed = (long) rows * cols / BLOCK_ELEMENTS * BLOCK_BYTES;
      MemorySegment tooSmall = arena.allocate(needed - 1);

      assertThatThrownBy(
              () ->
                  kernel.multiply(
                      new float[rows],
                      new float[cols],
                      tooSmall,
                      GgufTensorType.TQ1_0,
                      1,
                      rows,
                      cols))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  private static Path libraryPath() {
    return Path.of(System.getProperty("models.native.kernels.library"));
  }

  @Test
  void tq1_0IsNotPtq1_0() {
    // Different formats: group 256 in 54 bytes against Q8_K, versus group 128 in 28 against Q8_0.
    // Asserted together so a future change cannot quietly route one through the other's kernel.
    Path library = Path.of(System.getProperty("models.native.kernels.library"));
    try (var kernel = RustGgufBatchedMatrixKernel.open(library, 1)) {
      assertThat(kernel.supports(GgufTensorType.TQ1_0)).isTrue();
      assertThat(kernel.supports(GgufTensorType.PTQ1_0)).isTrue();
      assertThat(GgufTensorType.TQ1_0.blockSize()).isEqualTo(256);
      assertThat(GgufTensorType.PTQ1_0.blockSize()).isEqualTo(128);
    }
  }
}
