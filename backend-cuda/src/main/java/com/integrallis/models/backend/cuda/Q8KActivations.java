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
package com.integrallis.models.backend.cuda;

/**
 * Q8_K activation quantisation, on the host.
 *
 * <p>This stays in Java deliberately. It is {@code O(cols)} per token against the projections'
 * {@code O(rows * cols)}, so moving it to the device would buy nothing measurable while adding a
 * fourth kernel and a way for the two sides to disagree about what the activations were. Keeping
 * one quantiser means the device and the CPU fallback always see identical inputs, which is what
 * makes a token-level parity comparison meaningful.
 *
 * <p>Two details are easy to get wrong and are both load-bearing. The scale is derived from the
 * <strong>signed</strong> value at the absolute maximum, so it is frequently negative; and it is
 * <em>not</em> rounded through binary16 the way Q8_0's is. The sibling TornadoVM branch's first
 * test assumed a positive, fp16-rounded scale and failed on it.
 *
 * <p>Transcribed from {@code GgufQuantizationSupport.quantizeQ8_K} in {@code vectors-core} 0.1.22
 * and from its Rust twin {@code quantize_q8_k_batch}. {@code Q8KActivationsTest} asserts the
 * transcription is faithful by routing a full projection through {@code VectorUtil} and requiring
 * bit-identical output, so a drift in either direction fails the build.
 */
final class Q8KActivations {

  /** Weights per K-quant super-block. */
  static final int SUPER_BLOCK_VALUES = CudaKernelAbi.SUPER_BLOCK_VALUES;

  /** Activation values covered by one partial sum. */
  static final int SUM_BLOCK_VALUES = CudaKernelAbi.SUM_BLOCK_VALUES;

  private Q8KActivations() {}

  /** Signed 8-bit quants required for a batch-major activation plane. */
  static int quantCount(int batchSize, int cols) {
    return Math.multiplyExact(batchSize, cols);
  }

  /** Per-super-block scales required for a batch-major activation plane. */
  static int scaleCount(int batchSize, int cols) {
    return Math.multiplyExact(batchSize, cols / SUPER_BLOCK_VALUES);
  }

  /** Per-sub-block partial sums required for a batch-major activation plane. */
  static int sumCount(int batchSize, int cols) {
    return Math.multiplyExact(batchSize, cols / SUM_BLOCK_VALUES);
  }

  /**
   * Quantises {@code input} into {@code quants}, {@code scales} and {@code sums}.
   *
   * <p>{@code input} is batch-major: batch {@code b} occupies {@code [b * cols, (b + 1) * cols)}.
   */
  static void quantize(
      float[] input, int batchSize, int cols, byte[] quants, float[] scales, short[] sums) {
    int blocksPerRow = cols / SUPER_BLOCK_VALUES;
    for (int batch = 0; batch < batchSize; batch++) {
      for (int block = 0; block < blocksPerRow; block++) {
        int offset = batch * cols + block * SUPER_BLOCK_VALUES;
        float absoluteMaximum = 0.0f;
        float extremum = 0.0f;
        for (int index = 0; index < SUPER_BLOCK_VALUES; index++) {
          float value = input[offset + index];
          float magnitude = Math.abs(value);
          if (magnitude > absoluteMaximum) {
            absoluteMaximum = magnitude;
            extremum = value;
          }
        }
        int scaleIndex = batch * blocksPerRow + block;
        int sumBase =
            batch * cols / SUM_BLOCK_VALUES + block * SUPER_BLOCK_VALUES / SUM_BLOCK_VALUES;
        if (absoluteMaximum == 0.0f) {
          scales[scaleIndex] = 0.0f;
          java.util.Arrays.fill(quants, offset, offset + SUPER_BLOCK_VALUES, (byte) 0);
          java.util.Arrays.fill(
              sums, sumBase, sumBase + SUPER_BLOCK_VALUES / SUM_BLOCK_VALUES, (short) 0);
          continue;
        }
        float inverseScale = -127.0f / extremum;
        for (int index = 0; index < SUPER_BLOCK_VALUES; index++) {
          int quant = nearestInt(inverseScale * input[offset + index]);
          quants[offset + index] = (byte) Math.min(127, quant);
        }
        for (int sub = 0; sub < SUPER_BLOCK_VALUES / SUM_BLOCK_VALUES; sub++) {
          int total = 0;
          for (int index = 0; index < SUM_BLOCK_VALUES; index++) {
            total += quants[offset + sub * SUM_BLOCK_VALUES + index];
          }
          sums[sumBase + sub] = (short) total;
        }
        scales[scaleIndex] = 1.0f / inverseScale;
      }
    }
  }

  /**
   * GGML's round-to-nearest bit trick.
   *
   * <p>Adding {@code 1.5 * 2^23} parks the rounded integer in the mantissa; the mask and bias
   * recover it. Any other rounding rule changes the quantised activation and therefore the result,
   * so this is reproduced rather than replaced with {@link Math#round}.
   */
  static int nearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007F_FFFF) - 0x0040_0000;
  }
}
