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
package com.integrallis.models.backend.purejava.quant;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * GGUF {@code MXFP4} (tensor type 39): 32 weights per block as 4-bit E2M1 values with one E8M0
 * scale byte, the format GPT-OSS ships its expert weights in.
 *
 * <p><b>A transcription of the reference dequantizer, not a reimplementation from the struct
 * layout.</b> Same discipline as {@link TernaryDequantizer} and for the same reason: a low-bit
 * unpacker written from a plausible reading of the packing produces numbers that look reasonable
 * and are wrong, and the failure surfaces as slightly worse model output rather than as an error.
 * Three details here are not guessable from {@code {uint8_t e; uint8_t qs[16];}} alone.
 *
 * <ul>
 *   <li><b>The two nibbles of a byte are 16 elements apart, not adjacent.</b> Byte {@code j} holds
 *       element {@code j} in its low nibble and element {@code j + 16} in its high nibble, so the
 *       block is written as two halves rather than as consecutive pairs. Reading adjacent pairs
 *       transposes every block. Note this differs from the separate-tensor safetensors layout that
 *       {@code vectors-core}'s {@code Mxfp4Matrix} serves, where a byte does hold adjacent values.
 *   <li><b>The value table is doubled.</b> {@code {0, 1, 2, 3, 4, 6, 8, 12}} and negatives, which
 *       is twice the true E2M1 magnitudes {@code {0, 0.5, 1, 1.5, 2, 3, 4, 6}}. The scale
 *       compensates by being halved, so using the true magnitudes with an unhalved scale also works
 *       -- but mixing one convention's table with the other's scale is silently off by 2x.
 *   <li><b>The E8M0 scale decode is bit arithmetic with a denormal special case.</b> For {@code e <
 *       2} the result is a denormal built as {@code 0x00200000 << e}; otherwise it is {@code (e -
 *       1) << 23} reinterpreted as a float, which is {@code 2^(e - 128)} -- the halved form of
 *       {@code 2^(e - 127)}. Computing {@code Math.pow(2, e - 127) / 2} instead loses the
 *       denormals.
 * </ul>
 *
 * <p>Scalar on purpose, for the same reason as the ternary formats: bit-exact first, fast second. A
 * Panama version has to be proven identical to this one, and that comparison needs this one to
 * exist and be trusted.
 */
public final class Mxfp4Dequantizer {

  /** Weights per block. */
  public static final int BLOCK_SIZE = 32;

  /** Bytes per block: one E8M0 scale byte then {@code BLOCK_SIZE / 2} packed bytes. */
  public static final int BLOCK_BYTES = 1 + BLOCK_SIZE / 2;

  /**
   * E2M1 codes as twice their true magnitude, paired with the halved scale below.
   *
   * <p>Index is the raw 4-bit code: sign in the high bit, then two exponent bits and one mantissa
   * bit.
   */
  private static final int[] E2M1_DOUBLED = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
  };

  private Mxfp4Dequantizer() {}

  /**
   * Dequantizes {@code count} MXFP4 weights.
   *
   * @param source the mapped tensor data
   * @param sourceOffset byte offset of the first block
   * @param output destination
   * @param outputOffset first index to write
   * @param count number of weights, a multiple of {@value #BLOCK_SIZE}
   */
  public static void dequantize(
      MemorySegment source, long sourceOffset, float[] output, int outputOffset, int count) {
    if (count % BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "count must be a multiple of the "
              + BLOCK_SIZE
              + "-weight MXFP4 block, but was "
              + count);
    }
    int blocks = count / BLOCK_SIZE;
    int half = BLOCK_SIZE / 2;
    for (int block = 0; block < blocks; block++) {
      long base = sourceOffset + (long) block * BLOCK_BYTES;
      float scale = e8m0ToFloatHalf(Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, base)));
      int out = outputOffset + block * BLOCK_SIZE;
      for (int j = 0; j < half; j++) {
        int packed = Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, base + 1 + j));
        // Low nibble is element j; high nibble is element j + 16. Not a consecutive pair.
        output[out + j] = E2M1_DOUBLED[packed & 0x0F] * scale;
        output[out + j + half] = E2M1_DOUBLED[packed >>> 4] * scale;
      }
    }
  }

  /**
   * Dot product of a packed MXFP4 run against a float vector, without materialising the weights.
   *
   * <p>Used by the GGUF matmul so an MXFP4 tensor can be multiplied rather than only decoded. It
   * reads the same bytes in the same order as {@link #dequantize} -- the split-halves nibble order,
   * where byte {@code j} carries elements {@code j} and {@code j + 16} -- so the two cannot drift
   * apart in how they interpret a block. A test asserts they agree.
   *
   * <p>Accumulation is per block: the scale multiplies the block's summed contribution once instead
   * of every element, which is both fewer multiplies and closer to how the reference kernels group
   * the arithmetic.
   *
   * @param source the mapped tensor data
   * @param sourceOffset byte offset of the first block
   * @param vector the vector to multiply against
   * @param vectorOffset first index to read from {@code vector}
   * @param count number of weights, a multiple of {@value #BLOCK_SIZE}
   * @return the dot product
   */
  public static float dotProduct(
      MemorySegment source, long sourceOffset, float[] vector, int vectorOffset, int count) {
    if (count % BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "count must be a multiple of the "
              + BLOCK_SIZE
              + "-weight MXFP4 block, but was "
              + count);
    }
    Objects.checkFromIndexSize(vectorOffset, count, vector.length);
    int blocks = count / BLOCK_SIZE;
    int half = BLOCK_SIZE / 2;
    float total = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long base = sourceOffset + (long) block * BLOCK_BYTES;
      float scale = e8m0ToFloatHalf(Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, base)));
      int at = vectorOffset + block * BLOCK_SIZE;
      float blockTotal = 0.0f;
      for (int j = 0; j < half; j++) {
        int packed = Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, base + 1 + j));
        blockTotal += E2M1_DOUBLED[packed & 0x0F] * vector[at + j];
        blockTotal += E2M1_DOUBLED[packed >>> 4] * vector[at + j + half];
      }
      total += blockTotal * scale;
    }
    return total;
  }

  /**
   * Half of the E8M0 scale, matching the doubled value table.
   *
   * @param code the unsigned scale byte
   * @return {@code 2^(code - 128)}, with denormals for {@code code < 2}
   */
  private static float e8m0ToFloatHalf(int code) {
    int bits;
    if (code < 2) {
      // 0x00200000 is 2^-128 as a denormal; shifting gives 2^-127 for code 1.
      bits = 0x00200000 << code;
    } else {
      // 0.5 * 2^(code - 127) = 2^(code - 128): a normalized float with biased exponent code - 1.
      bits = (code - 1) << 23;
    }
    return Float.intBitsToFloat(bits);
  }
}
