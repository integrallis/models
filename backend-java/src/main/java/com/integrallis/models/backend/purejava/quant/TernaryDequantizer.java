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
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

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
 * <p>The dequantizers retain scalar reference implementations. PTQ row products use the Vector API
 * with a fixed reduction contract; independent unpacking and quantization oracles check their
 * output against the native kernels.
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
   * Float-activation PTQ1_0 product without expanding a matrix to floats. Eight-lane blocks
   * preserve one reduction order on every host. The scalar decoder remains the independent
   * unpacking reference; tests compare randomized blocks, scales, and input offsets against it.
   */
  public static float dotProductPtq1_0(
      MemorySegment source, long sourceOffset, float[] input, int inputOffset, int count) {
    require(count, 128);
    java.util.Objects.checkFromIndexSize(inputOffset, count, input.length);
    if (FloatVector.SPECIES_PREFERRED.vectorBitSize() < 256) {
      return dotProductPtq1_0NarrowHost(source, sourceOffset, input, inputOffset, count);
    }
    float total = 0;
    for (int block = 0; block < count / 128; block++) {
      long base = sourceOffset + (long) block * 28;
      int at = inputOffset + block * 128;
      IntVector a = ptqBytes(source, base);
      IntVector b = ptqBytes(source, base + 8);
      IntVector c = ptqBytes(source, base + 16);
      FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);
      for (int n = 0; n < 5; n++) {
        sum =
            ptqTrits(a, POW3[n])
                .fma(FloatVector.fromArray(FloatVector.SPECIES_256, input, at + n * 16), sum);
        sum =
            ptqTrits(b, POW3[n])
                .fma(FloatVector.fromArray(FloatVector.SPECIES_256, input, at + n * 16 + 8), sum);
        sum =
            ptqTrits(c, POW3[n])
                .fma(FloatVector.fromArray(FloatVector.SPECIES_256, input, at + 80 + n * 8), sum);
      }
      float even = (sum.lane(4) + sum.lane(0)) + (sum.lane(6) + sum.lane(2));
      float odd = (sum.lane(5) + sum.lane(1)) + (sum.lane(7) + sum.lane(3));
      float blockSum = even + odd;
      for (int n = 0; n < 4; n++) {
        for (int h = 0; h < 2; h++) {
          int q =
              (Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, base + 24 + h)) * POW3[n])
                  & 255;
          blockSum = Math.fma(((q * 3) >>> 8) - 1, input[at + 120 + n * 2 + h], blockSum);
        }
      }
      float scale = halfToFloat(source.get(LE_SHORT, base + 26));
      total = Math.fma(blockSum, scale, total);
    }
    return total;
  }

  // Retain eight logical FMA chains with primitive values when 256-bit vectors are not native.
  // Splitting this decoder's widening operations also increased first-call allocation; primitive
  // chains avoid both unsupported-vector emulation and those conversion wrappers.
  private static float dotProductPtq1_0NarrowHost(
      MemorySegment source, long sourceOffset, float[] input, int inputOffset, int count) {
    float total = 0;
    for (int block = 0; block < count / 128; block++) {
      long base = sourceOffset + block * 28L;
      int at = inputOffset + block * 128;
      float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
      for (int n = 0; n < 5; n++) {
        int power = POW3[n];
        a0 = Math.fma(ptqTrit(source, base + 0, power), input[at + n * 16 + 0], a0);
        a0 = Math.fma(ptqTrit(source, base + 8, power), input[at + n * 16 + 8], a0);
        a0 = Math.fma(ptqTrit(source, base + 16, power), input[at + 80 + n * 8 + 0], a0);
        a1 = Math.fma(ptqTrit(source, base + 1, power), input[at + n * 16 + 1], a1);
        a1 = Math.fma(ptqTrit(source, base + 9, power), input[at + n * 16 + 9], a1);
        a1 = Math.fma(ptqTrit(source, base + 17, power), input[at + 80 + n * 8 + 1], a1);
        a2 = Math.fma(ptqTrit(source, base + 2, power), input[at + n * 16 + 2], a2);
        a2 = Math.fma(ptqTrit(source, base + 10, power), input[at + n * 16 + 10], a2);
        a2 = Math.fma(ptqTrit(source, base + 18, power), input[at + 80 + n * 8 + 2], a2);
        a3 = Math.fma(ptqTrit(source, base + 3, power), input[at + n * 16 + 3], a3);
        a3 = Math.fma(ptqTrit(source, base + 11, power), input[at + n * 16 + 11], a3);
        a3 = Math.fma(ptqTrit(source, base + 19, power), input[at + 80 + n * 8 + 3], a3);
        a4 = Math.fma(ptqTrit(source, base + 4, power), input[at + n * 16 + 4], a4);
        a4 = Math.fma(ptqTrit(source, base + 12, power), input[at + n * 16 + 12], a4);
        a4 = Math.fma(ptqTrit(source, base + 20, power), input[at + 80 + n * 8 + 4], a4);
        a5 = Math.fma(ptqTrit(source, base + 5, power), input[at + n * 16 + 5], a5);
        a5 = Math.fma(ptqTrit(source, base + 13, power), input[at + n * 16 + 13], a5);
        a5 = Math.fma(ptqTrit(source, base + 21, power), input[at + 80 + n * 8 + 5], a5);
        a6 = Math.fma(ptqTrit(source, base + 6, power), input[at + n * 16 + 6], a6);
        a6 = Math.fma(ptqTrit(source, base + 14, power), input[at + n * 16 + 14], a6);
        a6 = Math.fma(ptqTrit(source, base + 22, power), input[at + 80 + n * 8 + 6], a6);
        a7 = Math.fma(ptqTrit(source, base + 7, power), input[at + n * 16 + 7], a7);
        a7 = Math.fma(ptqTrit(source, base + 15, power), input[at + n * 16 + 15], a7);
        a7 = Math.fma(ptqTrit(source, base + 23, power), input[at + 80 + n * 8 + 7], a7);
      }
      float blockSum = ((a4 + a0) + (a6 + a2)) + ((a5 + a1) + (a7 + a3));
      for (int n = 0; n < 4; n++) {
        for (int h = 0; h < 2; h++) {
          blockSum =
              Math.fma(
                  ptqTrit(source, base + 24 + h, POW3[n]), input[at + 120 + n * 2 + h], blockSum);
        }
      }
      total = Math.fma(blockSum, halfToFloat(source.get(LE_SHORT, base + 26)), total);
    }
    return total;
  }

  private static int ptqTrit(MemorySegment source, long offset, int power) {
    int packed = (Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, offset)) * power) & 255;
    return ((packed * 3) >>> 8) - 1;
  }

  /**
   * Quantizes complete 32-value blocks for the PTQ1_0 x Q8_0 contract. Scales are rounded to
   * binary16; quants use nearest-even rounding before narrowing, matching the native kernel.
   */
  public static void quantizePtqActivations(
      float[] input, int count, byte[] quants, float[] scales) {
    if (count < 0 || count % 32 != 0)
      throw new IllegalArgumentException(
          "Q8_0 activation count must be a nonnegative multiple of 32");
    java.util.Objects.checkFromIndexSize(0, count, input.length);
    java.util.Objects.checkFromIndexSize(0, count, quants.length);
    java.util.Objects.checkFromIndexSize(0, count / 32, scales.length);
    for (int at = 0; at < count; at += 32) {
      float maximum = 0;
      for (int j = 0; j < 32; j++) maximum = Math.max(maximum, Math.abs(input[at + j]));
      float inverse = maximum == 0 ? 0 : 127f / maximum;
      scales[at / 32] = Float.float16ToFloat(Float.floatToFloat16(maximum / 127f));
      for (int j = 0; j < 32; j++) quants[at + j] = (byte) Math.rint(input[at + j] * inverse);
    }
  }

  /**
   * PTQ1_0 row product with prequantized Q8_0 activations. Integer sums are exact; four activation
   * scales and then the weight scale fold in a fixed FMA order on every backend. The
   * float-activation method above remains an explicit accuracy reference.
   */
  public static float dotProductPtq1_0Q8_0(
      MemorySegment source,
      long sourceOffset,
      byte[] input,
      float[] scales,
      int inputOffset,
      int count) {
    require(count, 128);
    if (count < 0 || inputOffset % 32 != 0)
      throw new IllegalArgumentException("PTQ/Q8 input must use aligned nonnegative dimensions");
    java.util.Objects.checkFromIndexSize(inputOffset, count, input.length);
    java.util.Objects.checkFromIndexSize(inputOffset / 32, count / 32, scales.length);
    if (IntVector.SPECIES_PREFERRED.vectorBitSize() < 256) {
      return dotProductPtq1_0Q8_0NarrowHost(
          source, sourceOffset, input, scales, inputOffset, count);
    }
    float total = 0;
    for (int block = 0; block < count / 128; block++) {
      long base = sourceOffset + (long) block * 28;
      int at = inputOffset + block * 128;
      float blockSum = 0;
      for (int group = 0; group < 4; group++) {
        IntVector vectorSum = IntVector.zero(IntVector.SPECIES_256);
        int scalarTail = 0;
        for (int lane = 0; lane < 4; lane++) {
          int index = group * 32 + lane * 8;
          if (index == 120) {
            for (int j = 0; j < 8; j++) {
              int packed =
                  (Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, base + 24 + j % 2))
                          * POW3[j / 2])
                      & 255;
              scalarTail += ((packed * 3 >>> 8) - 1) * input[at + index + j];
            }
          } else {
            int packedIndex = index < 80 ? index % 16 : 16;
            int power = index < 80 ? index / 16 : (index - 80) / 8;
            IntVector trits =
                ptqBytes(source, base + packedIndex)
                    .mul(POW3[power])
                    .and(255)
                    .mul(3)
                    .lanewise(VectorOperators.LSHR, 8)
                    .sub(1);
            IntVector activations =
                (IntVector)
                    ByteVector.fromArray(ByteVector.SPECIES_64, input, at + index)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
            vectorSum = vectorSum.add(trits.mul(activations));
          }
        }
        int integerSum = vectorSum.reduceLanes(VectorOperators.ADD) + scalarTail;
        blockSum = Math.fma(integerSum, scales[at / 32 + group], blockSum);
      }
      total = Math.fma(blockSum, halfToFloat(source.get(LE_SHORT, base + 26)), total);
    }
    return total;
  }

  /**
   * Integer sums remain exact; primitive accumulation avoids unsupported-vector emulation. Scale
   * FMAs retain their order.
   */
  private static float dotProductPtq1_0Q8_0NarrowHost(
      MemorySegment source,
      long sourceOffset,
      byte[] input,
      float[] scales,
      int inputOffset,
      int count) {
    float total = 0;
    for (int block = 0; block < count / 128; block++) {
      long base = sourceOffset + block * 28L;
      int at = inputOffset + block * 128;
      float blockSum = 0;
      for (int group = 0; group < 4; group++) {
        int integerSum = 0;
        for (int segment = 0; segment < 4; segment++) {
          int index = group * 32 + segment * 8;
          if (index == 120) {
            for (int j = 0; j < 8; j++) {
              int packed =
                  (Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, base + 24 + j % 2))
                          * POW3[j / 2])
                      & 255;
              integerSum += ((packed * 3 >>> 8) - 1) * input[at + index + j];
            }
          } else {
            int packedIndex = index < 80 ? index % 16 : 16;
            int power = POW3[index < 80 ? index / 16 : (index - 80) / 8];
            for (int j = 0; j < 8; j++) {
              int packed =
                  (Byte.toUnsignedInt(source.get(ValueLayout.JAVA_BYTE, base + packedIndex + j))
                          * power)
                      & 255;
              integerSum += ((packed * 3 >>> 8) - 1) * input[at + index + j];
            }
          }
        }
        blockSum = Math.fma(integerSum, scales[at / 32 + group], blockSum);
      }
      total = Math.fma(blockSum, halfToFloat(source.get(LE_SHORT, base + 26)), total);
    }
    return total;
  }

  private static IntVector ptqBytes(MemorySegment source, long offset) {
    return ((IntVector)
            ByteVector.fromMemorySegment(
                    ByteVector.SPECIES_64, source, offset, ByteOrder.LITTLE_ENDIAN)
                .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0))
        .and(255);
  }

  private static FloatVector ptqTrits(IntVector bytes, int power) {
    return (FloatVector)
        bytes
            .mul(power)
            .and(255)
            .mul(3)
            .lanewise(VectorOperators.LSHR, 8)
            .sub(1)
            .convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
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
