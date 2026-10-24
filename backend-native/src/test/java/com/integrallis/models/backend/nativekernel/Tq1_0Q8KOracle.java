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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * An independent Java implementation of ggml's TQ1_0 x Q8_K projection, for the native kernel to be
 * checked against.
 *
 * <p>Transcribed from {@code quantize_row_q8_K_ref} and {@code ggml_vec_dot_tq1_0_q8_K} at
 * ggml-org/llama.cpp {@code 3f03b1772}. Written out longhand on purpose: an oracle that shares code
 * with the implementation it checks cannot catch a transcription error, because both would make it.
 *
 * <p>Three parts of the reference are not guessable from the block layouts:
 *
 * <ul>
 *   <li>Q8_K's scale is {@code -127 / max} where {@code max} is the <b>signed</b> value sitting at
 *       the largest magnitude, not the magnitude itself. The extremal element therefore always
 *       quantizes to -127 and the stored {@code d} is negative whenever that element was positive.
 *   <li>{@code qs} is clamped on the <b>upper side only</b> -- {@code MIN(127, v)} with no lower
 *       bound.
 *   <li>The weight trit falls out of a <b>wrapping</b> 8-bit multiply, and the activation index
 *       strides by the base-3 plane: {@code y.qs[j * 5 + l * 32 + m]}.
 * </ul>
 */
final class Tq1_0Q8KOracle {

  private static final int QK_K = 256;
  private static final int BLOCK_BYTES = 54;
  private static final int QS_BYTES = 48;
  private static final int QH_BYTES = 4;
  private static final int[] POW3 = {1, 3, 9, 27, 81};

  private Tq1_0Q8KOracle() {}

  static float[] multiply(MemorySegment weights, float[] input, int batchSize, int rows, int cols) {
    int blocksPerRow = cols / QK_K;
    byte[] quantized = new byte[batchSize * cols];
    float[] scales = new float[batchSize * blocksPerRow];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int block = 0; block < blocksPerRow; block++) {
        quantizeQ8KBlock(
            input, batch * cols + block * QK_K, quantized, scales, batch * blocksPerRow + block);
      }
    }

    float[] output = new float[batchSize * rows];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (int block = 0; block < blocksPerRow; block++) {
          long weightOffset = (long) (row * blocksPerRow + block) * BLOCK_BYTES;
          int activationOffset = batch * cols + block * QK_K;
          int scaleBits =
              (weights.get(ValueLayout.JAVA_BYTE, weightOffset + QS_BYTES + QH_BYTES) & 0xFF)
                  | ((weights.get(ValueLayout.JAVA_BYTE, weightOffset + QS_BYTES + QH_BYTES + 1)
                          & 0xFF)
                      << 8);
          float d = f16ToF32(scaleBits) * scales[batch * blocksPerRow + block];

          int total = 0;
          int byteIndex = 0;
          for (int chunk : new int[] {32, 16}) {
            while (byteIndex + chunk <= QS_BYTES) {
              for (int plane = 0; plane < POW3.length; plane++) {
                int activationBase = activationOffset + byteIndex * 5 + plane * chunk;
                for (int lane = 0; lane < chunk; lane++) {
                  int packed =
                      weights.get(ValueLayout.JAVA_BYTE, weightOffset + byteIndex + lane) & 0xFF;
                  int q = (packed * POW3[plane]) & 0xFF;
                  int xi = (q * 3) >>> 8;
                  total += (xi - 1) * quantized[activationBase + lane];
                }
              }
              byteIndex += chunk;
            }
          }
          for (int plane = 0; plane < 4; plane++) {
            int activationBase = activationOffset + QS_BYTES * 5 + plane * QH_BYTES;
            for (int lane = 0; lane < QH_BYTES; lane++) {
              int packed =
                  weights.get(ValueLayout.JAVA_BYTE, weightOffset + QS_BYTES + lane) & 0xFF;
              int q = (packed * POW3[plane]) & 0xFF;
              int xi = (q * 3) >>> 8;
              total += (xi - 1) * quantized[activationBase + lane];
            }
          }
          // fma, to match the kernel's one fused multiply-add per block.
          sum = Math.fma(d, (float) total, sum);
        }
        output[batch * rows + row] = sum;
      }
    }
    return output;
  }

  private static void quantizeQ8KBlock(
      float[] input, int offset, byte[] quantized, float[] scales, int scaleIndex) {
    float max = 0.0f;
    float absoluteMax = 0.0f;
    for (int index = 0; index < QK_K; index++) {
      float absolute = Math.abs(input[offset + index]);
      if (absolute > absoluteMax) {
        absoluteMax = absolute;
        max = input[offset + index];
      }
    }
    if (absoluteMax == 0.0f) {
      scales[scaleIndex] = 0.0f;
      for (int index = 0; index < QK_K; index++) {
        quantized[offset + index] = 0;
      }
      return;
    }
    float inverseScale = -127.0f / max;
    for (int index = 0; index < QK_K; index++) {
      int value = nearestInt(inverseScale * input[offset + index]);
      quantized[offset + index] = (byte) Math.min(127, value);
    }
    scales[scaleIndex] = 1.0f / inverseScale;
  }

  /** ggml's {@code nearest_int}: the add-and-mask trick, not {@code Math.round}. */
  private static int nearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007fffff) - 0x00400000;
  }

  private static float f16ToF32(int bits) {
    int sign = (bits & 0x8000) << 16;
    int exponent = (bits >> 10) & 0x1f;
    int significand = bits & 0x03ff;
    if (exponent == 0) {
      if (significand == 0) {
        return Float.intBitsToFloat(sign);
      }
      float magnitude = significand * 0x1.0p-24f;
      return sign == 0 ? magnitude : -magnitude;
    }
    if (exponent == 0x1f) {
      return Float.intBitsToFloat(sign | 0x7f800000 | (significand << 13));
    }
    return Float.intBitsToFloat(sign | ((exponent + 112) << 23) | (significand << 13));
  }
}
