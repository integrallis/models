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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Closes the chain between the shipped CPU path and the kernels that run on the device.
 *
 * <p>The Rust tests prove that the arithmetic compiled into the PTX is bit-exact with the CPU
 * kernel crate. This proves the remaining link: that the Java side quantises activations exactly as
 * {@code vectors-core} does, and folds the per-super-block scales in the same order. Together they
 * mean a device result and a Vector API result differ in no bit, which is what gate G1 asks of a
 * real model.
 *
 * <p>The comparison runs against {@link VectorUtil#ggufQ4_KQ8_KBatchedMatmul}, the production
 * Vector API entry point, on a fixture of real Q4_K super-blocks. It quantises internally, so
 * agreeing with it bit-for-bit means our quantiser agrees with its quantiser.
 */
class Q8KActivationsTest {

  private static final int SUPER_BLOCK = 256;
  private static final int BLOCK_BYTES = 144;

  @Test
  @DisplayName("the Java projection is bit-identical to the Vector API path on real Q4_K blocks")
  void theJavaProjectionIsBitIdenticalToTheVectorApiPath() {
    int rows = 24;
    int cols = 1_024;
    byte[] weightBytes = q4kTensor(rows, cols, 8_191L);
    float[] input = activations(cols, 65_537L);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment weights = arena.allocate(weightBytes.length);
      MemorySegment.copy(weightBytes, 0, weights, ValueLayout.JAVA_BYTE, 0, weightBytes.length);

      float[] expected = new float[rows];
      VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
          input,
          weights,
          1,
          rows,
          cols,
          expected,
          new byte[cols],
          new float[cols / SUPER_BLOCK],
          new short[cols / Q8KActivations.SUM_BLOCK_VALUES]);

      float[] actual = projectOnHost(weightBytes, input, rows, cols);
      for (int row = 0; row < rows; row++) {
        assertEquals(
            Float.floatToRawIntBits(expected[row]),
            Float.floatToRawIntBits(actual[row]),
            "row " + row + ": " + expected[row] + " != " + actual[row]);
      }
    }
  }

  @Test
  @DisplayName("the scale is signed, and the fixture actually produces negative ones")
  void theScaleIsSignedAndTheFixtureExercisesIt() {
    // Q8_K derives its scale from the SIGNED value at the absolute maximum, so it is negative
    // whenever that extremum is. A fixture that never produced a negative scale would let a
    // wrong sign convention pass; the sibling TornadoVM branch hit exactly that.
    float[] input = activations(SUPER_BLOCK * 8, 4_099L);
    float[] scales = new float[8];
    Q8KActivations.quantize(
        input, 1, SUPER_BLOCK * 8, new byte[SUPER_BLOCK * 8], scales, new short[SUPER_BLOCK / 2]);
    boolean sawNegative = false;
    boolean sawPositive = false;
    for (float scale : scales) {
      sawNegative |= scale < 0.0f;
      sawPositive |= scale > 0.0f;
    }
    assertTrue(sawNegative, "no negative scale in the fixture; the sign convention is untested");
    assertTrue(sawPositive, "no positive scale in the fixture");
  }

  @Test
  @DisplayName("the scale is not rounded through binary16, unlike Q8_0")
  void theScaleIsNotRoundedThroughBinary16() {
    float[] input = new float[SUPER_BLOCK];
    for (int index = 0; index < SUPER_BLOCK; index++) {
      input[index] = index == 0 ? 1.234_567_9f : 0.001f * index;
    }
    float[] scales = new float[1];
    Q8KActivations.quantize(
        input, 1, SUPER_BLOCK, new byte[SUPER_BLOCK], scales, new short[SUPER_BLOCK / 16]);
    float roundedThroughHalf = Float.float16ToFloat(Float.floatToFloat16(scales[0]));
    assertTrue(
        Float.floatToRawIntBits(scales[0]) != Float.floatToRawIntBits(roundedThroughHalf),
        "the fixture's scale survives an fp16 round trip, so this test cannot tell the two "
            + "conventions apart; pick a different value");
  }

  @Test
  @DisplayName("partial sums match an explicit sum of the quants")
  void partialSumsMatchAnExplicitSum() {
    int cols = SUPER_BLOCK * 4;
    float[] input = activations(cols, 7_919L);
    byte[] quants = new byte[cols];
    float[] scales = new float[cols / SUPER_BLOCK];
    short[] sums = new short[cols / Q8KActivations.SUM_BLOCK_VALUES];
    Q8KActivations.quantize(input, 1, cols, quants, scales, sums);
    for (int sub = 0; sub < sums.length; sub++) {
      int expected = 0;
      for (int index = 0; index < Q8KActivations.SUM_BLOCK_VALUES; index++) {
        expected += quants[sub * Q8KActivations.SUM_BLOCK_VALUES + index];
      }
      assertEquals((short) expected, sums[sub], "sub-block " + sub);
    }
  }

  @Test
  @DisplayName("an all-zero block yields a zero scale rather than a division by zero")
  void anAllZeroBlockYieldsAZeroScale() {
    float[] scales = new float[1];
    byte[] quants = new byte[SUPER_BLOCK];
    Q8KActivations.quantize(
        new float[SUPER_BLOCK], 1, SUPER_BLOCK, quants, scales, new short[SUPER_BLOCK / 16]);
    assertEquals(0.0f, scales[0]);
    for (byte quant : quants) {
      assertEquals((byte) 0, quant);
    }
  }

  // -------------------------------------------------------------------------------------------
  // The device algorithm, replayed on the host.
  // -------------------------------------------------------------------------------------------

  /**
   * The exact decomposition the PTX kernel uses: exact integer partials per super-block, then the
   * float scales folded in ascending block order.
   */
  private static float[] projectOnHost(byte[] weights, float[] input, int rows, int cols) {
    int blocksPerRow = cols / SUPER_BLOCK;
    byte[] quants = new byte[cols];
    float[] scales = new float[blocksPerRow];
    short[] sums = new short[cols / Q8KActivations.SUM_BLOCK_VALUES];
    Q8KActivations.quantize(input, 1, cols, quants, scales, sums);

    float[] output = new float[rows];
    for (int row = 0; row < rows; row++) {
      float sum = 0.0f;
      for (int block = 0; block < blocksPerRow; block++) {
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        float activationScale = scales[block];
        float d = Float.float16ToFloat(readShort(weights, base)) * activationScale;
        float dMin = Float.float16ToFloat(readShort(weights, base + 2)) * activationScale;
        int quantizedSum = 0;
        int minimumSum = 0;
        for (int group = 0; group < 8; group++) {
          int groupScale = qkScale(weights, base + 4, group);
          int groupMin = qkMin(weights, base + 4, group);
          int packedOffset = base + 16 + (group >>> 1) * 32;
          int shift = (group & 1) * 4;
          int activationOffset = block * SUPER_BLOCK + group * 32;
          int groupDot = 0;
          for (int index = 0; index < 32; index++) {
            int quant = (weights[packedOffset + index] >> shift) & 0x0F;
            groupDot += quant * quants[activationOffset + index];
          }
          quantizedSum += groupScale * groupDot;
          int sumOffset = block * (SUPER_BLOCK / 16) + group * 2;
          minimumSum += groupMin * (sums[sumOffset] + sums[sumOffset + 1]);
        }
        sum = Math.fma(d, quantizedSum, sum);
        sum = Math.fma(-dMin, minimumSum, sum);
      }
      output[row] = sum;
    }
    return output;
  }

  private static short readShort(byte[] bytes, int offset) {
    return (short) ((bytes[offset] & 0xFF) | ((bytes[offset + 1] & 0xFF) << 8));
  }

  private static int qkScale(byte[] weights, int scalesOffset, int group) {
    if (group < 4) {
      return weights[scalesOffset + group] & 0x3F;
    }
    int low = weights[scalesOffset + group + 4] & 0x0F;
    int high = (weights[scalesOffset + group - 4] & 0xFF) >>> 6;
    return low | (high << 4);
  }

  private static int qkMin(byte[] weights, int scalesOffset, int group) {
    if (group < 4) {
      return weights[scalesOffset + group + 4] & 0x3F;
    }
    int low = (weights[scalesOffset + group + 4] & 0xFF) >>> 4;
    int high = (weights[scalesOffset + group] & 0xFF) >>> 6;
    return low | (high << 4);
  }

  // -------------------------------------------------------------------------------------------
  // Fixtures: a fixed generator, so the bytes are identical on every host and in CI.
  // -------------------------------------------------------------------------------------------

  private static byte[] q4kTensor(int rows, int cols, long seed) {
    int blocks = rows * (cols / SUPER_BLOCK);
    byte[] bytes = new byte[blocks * BLOCK_BYTES];
    long state = seed | 1L;
    for (int block = 0; block < blocks; block++) {
      int base = block * BLOCK_BYTES;
      state = next(state);
      short d = Float.floatToFloat16(0.02f + (int) ((state >>> 33) % 64) * 0.001f);
      state = next(state);
      short dMin = Float.floatToFloat16(0.005f + (int) ((state >>> 33) % 32) * 0.000_5f);
      bytes[base] = (byte) (d & 0xFF);
      bytes[base + 1] = (byte) ((d >> 8) & 0xFF);
      bytes[base + 2] = (byte) (dMin & 0xFF);
      bytes[base + 3] = (byte) ((dMin >> 8) & 0xFF);

      int[] scales = new int[8];
      int[] mins = new int[8];
      for (int group = 0; group < 8; group++) {
        state = next(state);
        scales[group] = (int) ((state >>> 33) % 64);
        state = next(state);
        mins[group] = (int) ((state >>> 33) % 64);
      }
      for (int group = 0; group < 4; group++) {
        bytes[base + 4 + group] = (byte) scales[group];
        bytes[base + 8 + group] = (byte) mins[group];
      }
      for (int group = 4; group < 8; group++) {
        bytes[base + 4 + group] = (byte) ((scales[group] & 0x0F) | ((mins[group] & 0x0F) << 4));
        bytes[base + group] |= (byte) ((scales[group] >>> 4) << 6);
        bytes[base + 4 + group] |= (byte) ((mins[group] >>> 4) << 6);
      }
      for (int index = 0; index < 128; index++) {
        state = next(state);
        bytes[base + 16 + index] = (byte) (state >>> 33);
      }
    }
    return bytes;
  }

  private static float[] activations(int cols, long seed) {
    float[] values = new float[cols];
    long state = seed | 1L;
    for (int index = 0; index < cols; index++) {
      state = next(state);
      values[index] = (((state >>> 33) / (float) (Integer.MAX_VALUE)) * 2.0f - 1.0f) * 3.0f;
    }
    return values;
  }

  private static long next(long state) {
    return state * 6_364_136_223_846_793_005L + 1_442_695_040_888_963_407L;
  }
}
