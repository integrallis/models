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

/**
 * Ternary and low-bit GGUF block formats: upstream {@code TQ1_0}/{@code TQ2_0} and the
 * PrismML-private {@code PTQ1_0}/{@code PQ2_0}.
 *
 * <p>All four are the same two ideas at two group sizes. The 1-bit-ish formats pack <b>five trits
 * per byte</b> in base 3 ({@code 3^5 = 243 < 256}) and recover each one by multiplying by a power
 * of three and taking the top bits: {@code xi = ((q * pow3[n]) * 3) >> 8}, then {@code (xi - 1) *
 * d}. The 2-bit formats take two bits at a time and apply the same {@code (q - 1) * d}.
 *
 * <p><b>These are transcriptions of the reference dequantizers, not reimplementations from the
 * struct layouts.</b> That distinction is deliberate and was learned expensively: a low-bit
 * unpacker written from a plausible reading of the packing produces numbers that look reasonable
 * and are wrong, and the failure surfaces as slightly worse model output rather than as an error.
 * Three details here are not guessable from the struct definitions alone:
 *
 * <ul>
 *   <li><b>Output order is interleaved, not sequential.</b> Within a chunk of {@code c} bytes the
 *       reference emits the base-3 position {@code n} in the <i>outer</i> loop and the byte {@code
 *       m} in the inner one, so the value at output index {@code n * c + m} comes from byte {@code
 *       m}. Reading bytes in order and unpacking five trits from each would transpose the tensor.
 *   <li><b>{@code PTQ1_0} chunks its 24 {@code qs} bytes as 16 then 8</b>, from the reference's
 *       {@code {32, 16, 8}} stage list: 32 does not fit in 24, 16 does, and 8 covers the remainder.
 *       The chunk width is the stride of that interleave, so getting it wrong reorders the output.
 *   <li><b>{@code PQ2_0} is not ternary.</b> Its four codes map to {@code -1, 0, +1, +2}; the
 *       fourth state is used. It also stores its scale as the <b>first</b> field, where {@code
 *       TQ2_0} puts the scale last.
 * </ul>
 *
 * <p>Scalar on purpose for now. The base-3 recovery is a multiply and a shift per value and
 * vectorises cleanly over the Vector API, but a dequantizer that is bit-exact first and fast second
 * is the right order: the Panama version has to be proven identical to this one, and that
 * comparison needs this one to exist and be trusted.
 */
public final class TernaryDequantizer {

  /** Powers of three used to select a trit out of a base-3 packed byte. */
  private static final int[] POW3 = {1, 3, 9, 27, 81, 243};

  /** Chunk widths the {@code PTQ1_0}/{@code TQ1_0} reference walks {@code qs} with, in order. */
  private static final int[] STAGES = {32, 16, 8};

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN);

  private TernaryDequantizer() {}

  /**
   * Dequantizes upstream {@code TQ1_0}: 256 weights per block, 48 base-3 bytes, 4 high bytes, scale
   * last.
   *
   * @param source the mapped tensor data
   * @param sourceOffset byte offset of the first block
   * @param output destination
   * @param outputOffset first index to write
   * @param count number of weights, a multiple of 256
   */
  public static void dequantizeTq1_0(
      MemorySegment source, long sourceOffset, float[] output, int outputOffset, int count) {
    trits(source, sourceOffset, output, outputOffset, count, 256, 48, 4, /* scaleFirst= */ false);
  }

  /**
   * Dequantizes PrismML {@code PTQ1_0}: 128 weights per block, 24 base-3 bytes, 2 high bytes, scale
   * last.
   *
   * @param source the mapped tensor data
   * @param sourceOffset byte offset of the first block
   * @param output destination
   * @param outputOffset first index to write
   * @param count number of weights, a multiple of 128
   */
  public static void dequantizePtq1_0(
      MemorySegment source, long sourceOffset, float[] output, int outputOffset, int count) {
    trits(source, sourceOffset, output, outputOffset, count, 128, 24, 2, /* scaleFirst= */ false);
  }

  /**
   * Dequantizes upstream {@code TQ2_0}: 256 weights per block, 2 bits each, scale last.
   *
   * @param source the mapped tensor data
   * @param sourceOffset byte offset of the first block
   * @param output destination
   * @param outputOffset first index to write
   * @param count number of weights, a multiple of 256
   */
  public static void dequantizeTq2_0(
      MemorySegment source, long sourceOffset, float[] output, int outputOffset, int count) {
    twoBit(source, sourceOffset, output, outputOffset, count, 256, /* scaleFirst= */ false);
  }

  /**
   * Dequantizes PrismML {@code PQ2_0}: 128 weights per block, 2 bits each, scale <b>first</b>.
   *
   * @param source the mapped tensor data
   * @param sourceOffset byte offset of the first block
   * @param output destination
   * @param outputOffset first index to write
   * @param count number of weights, a multiple of 128
   */
  public static void dequantizePq2_0(
      MemorySegment source, long sourceOffset, float[] output, int outputOffset, int count) {
    twoBit(source, sourceOffset, output, outputOffset, count, 128, /* scaleFirst= */ true);
  }

  /** Base-3 trit formats. {@code qsBytes} carry five values each, {@code qhBytes} four. */
  private static void trits(
      MemorySegment source,
      long sourceOffset,
      float[] output,
      int outputOffset,
      int count,
      int groupSize,
      int qsBytes,
      int qhBytes,
      boolean scaleFirst) {
    require(count, groupSize);
    int blockBytes = qsBytes + qhBytes + Short.BYTES;
    int blocks = count / groupSize;
    int out = outputOffset;

    for (int block = 0; block < blocks; block++) {
      long base = sourceOffset + (long) block * blockBytes;
      long qs = scaleFirst ? base + Short.BYTES : base;
      long qh = qs + qsBytes;
      float d = halfToFloat(source.get(LE_SHORT, scaleFirst ? base : qh + qhBytes));

      // Walk qs in the reference's chunk widths, emitting base-3 position n outermost.
      int j = 0;
      for (int stage = 0; stage < STAGES.length; stage++) {
        int chunk = STAGES[stage];
        while (j + chunk <= qsBytes) {
          for (int n = 0; n < 5; n++) {
            for (int m = 0; m < chunk; m++) {
              // uint8_t q = qs[...] * pow3[n]  -- the reference truncates this multiply to 8 bits,
              // and the base-3 selection depends on that wrap. Without the mask the trits are wrong
              // for every position above the first.
              int q =
                  (Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, qs + j + m)) * POW3[n])
                      & 0xFF;
              int xi = (q * 3) >>> 8;
              output[out++] = (xi - 1) * d;
            }
          }
          j += chunk;
        }
      }
      // Then the high bytes, four values each, position outermost again.
      for (int n = 0; n < 4; n++) {
        for (int h = 0; h < qhBytes; h++) {
          int q = (Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, qh + h)) * POW3[n]) & 0xFF;
          int xi = (q * 3) >>> 8;
          output[out++] = (xi - 1) * d;
        }
      }
    }
  }

  /** Two-bit formats. Four codes per byte, mapped {@code (q - 1) * d}. */
  private static void twoBit(
      MemorySegment source,
      long sourceOffset,
      float[] output,
      int outputOffset,
      int count,
      int groupSize,
      boolean scaleFirst) {
    require(count, groupSize);
    int qsBytes = groupSize / 4;
    int blockBytes = qsBytes + Short.BYTES;
    int blocks = count / groupSize;
    int out = outputOffset;

    for (int block = 0; block < blocks; block++) {
      long base = sourceOffset + (long) block * blockBytes;
      long qs = scaleFirst ? base + Short.BYTES : base;
      float d = halfToFloat(source.get(LE_SHORT, scaleFirst ? base : qs + qsBytes));

      if (scaleFirst) {
        // PQ2_0 walks weights in index order: byte j/4, bit offset (j%4)*2.
        for (int j = 0; j < groupSize; j++) {
          int byteValue = Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, qs + (j >> 2)));
          int q = (byteValue >>> ((j & 3) * 2)) & 0x03;
          output[out++] = (q - 1) * d;
        }
      } else {
        // TQ2_0 emits the bit pair l outermost across each 32-byte chunk.
        for (int j = 0; j < qsBytes; j += 32) {
          for (int l = 0; l < 4; l++) {
            for (int m = 0; m < 32; m++) {
              int byteValue = Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, qs + j + m));
              int q = (byteValue >>> (l * 2)) & 3;
              output[out++] = (q - 1) * d;
            }
          }
        }
      }
    }
  }

  private static void require(int count, int groupSize) {
    if (count % groupSize != 0) {
      throw new IllegalArgumentException(
          "count must be a multiple of the " + groupSize + "-weight block size, but was " + count);
    }
  }

  /** IEEE half to float, matching {@code GGML_FP16_TO_FP32}. */
  private static float halfToFloat(short bits) {
    return Float.float16ToFloat(bits);
  }
}
