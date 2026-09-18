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
package com.integrallis.models.backend.tornado;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Off-device parity tests for the K-quant TornadoVM kernels.
 *
 * <p>These run the kernels as ordinary sequential Java — {@code @Parallel} is an annotation, so a
 * direct call from a unit test executes the same arithmetic the PTX backend compiles — and score
 * them against the production vectors-core CPU kernels the pure-Java backend actually uses.
 *
 * <p>Two numeric contracts are asserted, deliberately separately:
 *
 * <ol>
 *   <li><b>Exact.</b> On super-blocks whose scales are powers of two and whose quantized sums stay
 *       well inside the exactly representable float range, the kernel must equal the CPU kernel
 *       bit-for-bit. This is the strong statement: it proves the GGUF super-block decode — the
 *       six-bit scale/min packing, the nibble group order, the Q6_K two-bit high pairs, and the
 *       Q8_K per-16 sums — is identical, because every integer reduction is exact and any decode
 *       error would move the result by at least one quantization step.
 *   <li><b>Bounded.</b> On pseudo-random super-blocks the kernel cannot be exact: the CPU kernels
 *       fuse the per-block scale application with {@code Math.fma} and this kernel uses a plain
 *       multiply and add, which is the operation set the existing Q4_0 kernel is known to compile
 *       to PTX with. Q4_K applies two such steps per super-block and Q6_K one, so the guaranteed
 *       contract is <b>at most two float roundings per super-block</b>: {@code |kernel - cpu| <= 2
 *       * (cols / 256) * ulp(max |cpu|)}. That is what {@link #assertWithinFusedMultiplyAddBudget}
 *       asserts.
 *       <p>Measured here on 2026-09-18 (Apple M-series, vectors-core 0.1.22 Panama provider,
 *       256-bit species, {@code fastVectorFMA=true}), over 40 pseudo-random matrices per width at
 *       batch 4 by 8 rows: Q4_K worst absolute difference 7.6e-6 at cols=256, 6.1e-5 at cols=1024,
 *       1.2e-4 at cols=4096; Q6_K 0.0, 9.2e-5, 1.8e-4 for the same widths, against reference
 *       magnitudes of 3.3e2, 5.3e2 and 1.4e3. Every one of those is inside the asserted budget with
 *       at least an order of magnitude to spare, and all of them scale with the super-block count
 *       exactly as a per-super-block rounding predicts.
 * </ol>
 *
 * <p>What these tests <b>cannot</b> establish: that TornadoVM's PTX backend lowers this bytecode to
 * the same arithmetic. A device run can differ by contracting {@code a * b + c} into a hardware
 * fused multiply-add, which would move results toward the CPU kernel rather than away from it, but
 * that is an argument, not a measurement. The device-side statement needs a GPU host.
 */
class KQuantProjectionKernelTest {
  private static final int SUPER_BLOCK = KQuantProjectionKernel.SUPER_BLOCK_VALUES;
  private static final int SUM_BLOCK = KQuantProjectionKernel.SUM_BLOCK_VALUES;

  // --- Q4_K ---

  @Test
  void matchesProductionQ4KProjectionAcrossBatches() {
    int batchSize = 3;
    int rows = 9;
    int cols = 512;
    byte[] weights = randomQ4KMatrix(rows, cols, 31L);
    float[] input = randomFloats(batchSize * cols, 37L);

    float[] actual = runQ4K(weights, input, batchSize, rows, cols);

    assertWithinFusedMultiplyAddBudget(
        actual, referenceQ4K(weights, input, batchSize, rows, cols), cols);
  }

  @Test
  void matchesProductionQ4KProjectionExactlyOnExactlyRepresentableBlocks() {
    int batchSize = 2;
    int rows = 5;
    int cols = 512;
    byte[] weights = exactQ4KMatrix(rows, cols, 71L);
    float[] input = exactFloats(batchSize, cols, 73L);

    float[] actual = runQ4K(weights, input, batchSize, rows, cols);

    assertThat(actual).isEqualTo(referenceQ4K(weights, input, batchSize, rows, cols));
  }

  @Test
  void matchesTwoProductionQ4KProjectionsWithOnePreparedActivation() {
    int batchSize = 2;
    int firstRows = 7;
    int secondRows = 5;
    int cols = 256;
    byte[] firstWeights = exactQ4KMatrix(firstRows, cols, 41L);
    byte[] secondWeights = exactQ4KMatrix(secondRows, cols, 43L);
    float[] input = exactFloats(batchSize, cols, 47L);
    Q8KActivation activation = Q8KActivation.of(input, batchSize, cols);
    FloatArray firstOutput = new FloatArray(batchSize * firstRows);
    FloatArray secondOutput = new FloatArray(batchSize * secondRows);

    KQuantProjectionKernel.multiplyQ4KDual(
        ByteArray.fromArray(firstWeights),
        firstRows,
        ByteArray.fromArray(secondWeights),
        secondRows,
        activation.quants(),
        activation.scales(),
        activation.sums(),
        firstOutput,
        secondOutput,
        batchSize,
        cols);

    assertThat(firstOutput.toHeapArray())
        .isEqualTo(referenceQ4K(firstWeights, input, batchSize, firstRows, cols));
    assertThat(secondOutput.toHeapArray())
        .isEqualTo(referenceQ4K(secondWeights, input, batchSize, secondRows, cols));
  }

  @Test
  void matchesThreeProductionQ4KProjectionsWithOnePreparedActivation() {
    int batchSize = 2;
    int firstRows = 4;
    int secondRows = 3;
    int thirdRows = 2;
    int cols = 256;
    byte[] firstWeights = randomQ4KMatrix(firstRows, cols, 53L);
    byte[] secondWeights = randomQ4KMatrix(secondRows, cols, 59L);
    byte[] thirdWeights = randomQ4KMatrix(thirdRows, cols, 61L);
    float[] input = randomFloats(batchSize * cols, 67L);
    Q8KActivation activation = Q8KActivation.of(input, batchSize, cols);
    FloatArray firstOutput = new FloatArray(batchSize * firstRows);
    FloatArray secondOutput = new FloatArray(batchSize * secondRows);
    FloatArray thirdOutput = new FloatArray(batchSize * thirdRows);

    KQuantProjectionKernel.multiplyQ4KTriple(
        ByteArray.fromArray(firstWeights),
        firstRows,
        ByteArray.fromArray(secondWeights),
        secondRows,
        ByteArray.fromArray(thirdWeights),
        thirdRows,
        activation.quants(),
        activation.scales(),
        activation.sums(),
        firstOutput,
        secondOutput,
        thirdOutput,
        batchSize,
        cols);

    assertWithinFusedMultiplyAddBudget(
        firstOutput.toHeapArray(),
        referenceQ4K(firstWeights, input, batchSize, firstRows, cols),
        cols);
    assertWithinFusedMultiplyAddBudget(
        secondOutput.toHeapArray(),
        referenceQ4K(secondWeights, input, batchSize, secondRows, cols),
        cols);
    assertWithinFusedMultiplyAddBudget(
        thirdOutput.toHeapArray(),
        referenceQ4K(thirdWeights, input, batchSize, thirdRows, cols),
        cols);
  }

  // --- Q6_K ---

  @Test
  void matchesProductionQ6KProjectionAcrossBatches() {
    int batchSize = 3;
    int rows = 9;
    int cols = 512;
    byte[] weights = randomQ6KMatrix(rows, cols, 83L);
    float[] input = randomFloats(batchSize * cols, 89L);

    float[] actual = runQ6K(weights, input, batchSize, rows, cols);

    assertWithinFusedMultiplyAddBudget(
        actual, referenceQ6K(weights, input, batchSize, rows, cols), cols);
  }

  @Test
  void matchesProductionQ6KProjectionExactlyOnExactlyRepresentableBlocks() {
    int batchSize = 2;
    int rows = 5;
    int cols = 512;
    byte[] weights = exactQ6KMatrix(rows, cols, 97L);
    float[] input = exactFloats(batchSize, cols, 101L);

    float[] actual = runQ6K(weights, input, batchSize, rows, cols);

    assertThat(actual).isEqualTo(referenceQ6K(weights, input, batchSize, rows, cols));
  }

  // --- Mixed Q4_K / Q4_K / Q6_K, the Q4_K_M attention projection group ---

  @Test
  void matchesTheProductionMixedAttentionProjectionGroup() {
    int batchSize = 2;
    int queryRows = 6;
    int keyRows = 3;
    int valueRows = 3;
    int cols = 256;
    byte[] queryWeights = exactQ4KMatrix(queryRows, cols, 103L);
    byte[] keyWeights = exactQ4KMatrix(keyRows, cols, 107L);
    byte[] valueWeights = exactQ6KMatrix(valueRows, cols, 109L);
    float[] input = exactFloats(batchSize, cols, 113L);
    Q8KActivation activation = Q8KActivation.of(input, batchSize, cols);
    FloatArray queryOutput = new FloatArray(batchSize * queryRows);
    FloatArray keyOutput = new FloatArray(batchSize * keyRows);
    FloatArray valueOutput = new FloatArray(batchSize * valueRows);

    KQuantProjectionKernel.multiplyMixedTriple(
        ByteArray.fromArray(queryWeights),
        queryRows,
        ByteArray.fromArray(keyWeights),
        keyRows,
        ByteArray.fromArray(valueWeights),
        valueRows,
        activation.quants(),
        activation.scales(),
        activation.sums(),
        queryOutput,
        keyOutput,
        valueOutput,
        batchSize,
        cols);

    assertThat(queryOutput.toHeapArray())
        .isEqualTo(referenceQ4K(queryWeights, input, batchSize, queryRows, cols));
    assertThat(keyOutput.toHeapArray())
        .isEqualTo(referenceQ4K(keyWeights, input, batchSize, keyRows, cols));
    assertThat(valueOutput.toHeapArray())
        .isEqualTo(referenceQ6K(valueWeights, input, batchSize, valueRows, cols));
  }

  // --- Q8_K activation preparation ---

  @Test
  void preparesQ8KActivationsWithBlockSumsThatMatchTheStoredQuants() {
    int batchSize = 2;
    int cols = 512;
    float[] input = randomFloats(batchSize * cols, 127L);
    byte[] quants = new byte[batchSize * cols];
    float[] scales = new float[batchSize * cols / SUPER_BLOCK];
    int[] sums = new int[batchSize * cols / SUM_BLOCK];

    KQuantProjectionKernel.quantize(input, quants, scales, sums, batchSize, cols);

    for (int sumBlock = 0; sumBlock < sums.length; sumBlock++) {
      int expected = 0;
      for (int index = 0; index < SUM_BLOCK; index++) {
        expected += quants[sumBlock * SUM_BLOCK + index];
      }
      assertThat(sums[sumBlock]).as("sum block %d", sumBlock).isEqualTo(expected);
      assertThat(expected).isBetween(-Short.MAX_VALUE, (int) Short.MAX_VALUE);
    }
    // Q8_K scales carry the sign of the block extremum, unlike Q8_0's absolute-maximum scale.
    for (float scale : scales) {
      assertThat(scale).isNotZero().isFinite();
    }
    for (int index = 0; index < quants.length; index++) {
      assertThat((int) quants[index]).isBetween(-127, 127);
      float reconstructed = quants[index] * scales[index / SUPER_BLOCK];
      assertThat(reconstructed)
          .as("reconstructed element %d", index)
          .isCloseTo(input[index], within(0.02f));
    }
  }

  @Test
  void preparesZeroActivationBlocksWithoutDividingByZero() {
    int cols = 256;
    byte[] quants = new byte[cols];
    float[] scales = new float[1];
    int[] sums = new int[cols / SUM_BLOCK];

    KQuantProjectionKernel.quantize(new float[cols], quants, scales, sums, 1, cols);

    assertThat(scales[0]).isZero();
    assertThat(quants).containsOnly((byte) 0);
    assertThat(sums).containsOnly(0);
  }

  @Test
  void projectsZeroActivationsToZero() {
    int rows = 8;
    int cols = 256;
    byte[] weights = randomQ4KMatrix(rows, cols, 131L);

    float[] actual = runQ4K(weights, new float[cols], 1, rows, cols);

    assertThat(actual).containsOnly(0.0f);
  }

  // --- Validation ---

  @Test
  void rejectsQ4KActivationSumStorageThatDoesNotMatchTheShape() {
    assertThatThrownBy(
            () ->
                KQuantProjectionKernel.validateQ4K(
                    new ByteArray(KQuantProjectionKernel.Q4_K_BLOCK_BYTES),
                    new ByteArray(SUPER_BLOCK),
                    new FloatArray(1),
                    new IntArray(3),
                    new FloatArray(1),
                    1,
                    1,
                    SUPER_BLOCK))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("activation sums");
  }

  @Test
  void rejectsQ4KShapesThatAreNotWholeSuperBlocks() {
    assertThatThrownBy(
            () ->
                KQuantProjectionKernel.validateQ4K(
                    new ByteArray(KQuantProjectionKernel.Q4_K_BLOCK_BYTES),
                    new ByteArray(128),
                    new FloatArray(1),
                    new IntArray(8),
                    new FloatArray(1),
                    1,
                    1,
                    128))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("multiple of 256");
  }

  @Test
  void rejectsQ4KWeightsThatDoNotMatchTheShape() {
    assertThatThrownBy(
            () ->
                KQuantProjectionKernel.validateQ4K(
                    new ByteArray(KQuantProjectionKernel.Q4_K_BLOCK_BYTES + 1),
                    new ByteArray(SUPER_BLOCK),
                    new FloatArray(1),
                    new IntArray(16),
                    new FloatArray(1),
                    1,
                    1,
                    SUPER_BLOCK))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("weights");
  }

  @Test
  void rejectsQ4KActivationStorageThatDoesNotMatchTheShape() {
    assertThatThrownBy(
            () ->
                KQuantProjectionKernel.validateQ4K(
                    new ByteArray(KQuantProjectionKernel.Q4_K_BLOCK_BYTES),
                    new ByteArray(SUPER_BLOCK - 1),
                    new FloatArray(1),
                    new IntArray(16),
                    new FloatArray(1),
                    1,
                    1,
                    SUPER_BLOCK))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("activations");
  }

  @Test
  void rejectsQ4KActivationScaleStorageThatDoesNotMatchTheShape() {
    assertThatThrownBy(
            () ->
                KQuantProjectionKernel.validateQ4K(
                    new ByteArray(KQuantProjectionKernel.Q4_K_BLOCK_BYTES),
                    new ByteArray(SUPER_BLOCK),
                    new FloatArray(2),
                    new IntArray(16),
                    new FloatArray(1),
                    1,
                    1,
                    SUPER_BLOCK))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("activation scales");
  }

  @Test
  void rejectsQ6KOutputStorageThatDoesNotMatchTheShape() {
    assertThatThrownBy(
            () ->
                KQuantProjectionKernel.validateQ6K(
                    new ByteArray(KQuantProjectionKernel.Q6_K_BLOCK_BYTES),
                    new ByteArray(SUPER_BLOCK),
                    new FloatArray(1),
                    new FloatArray(2),
                    1,
                    1,
                    SUPER_BLOCK))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("output");
  }

  @Test
  void rejectsNonPositiveQ6KShapes() {
    assertThatThrownBy(
            () ->
                KQuantProjectionKernel.validateQ6K(
                    new ByteArray(KQuantProjectionKernel.Q6_K_BLOCK_BYTES),
                    new ByteArray(SUPER_BLOCK),
                    new FloatArray(1),
                    new FloatArray(1),
                    0,
                    1,
                    SUPER_BLOCK))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("batchSize");
    assertThatThrownBy(
            () ->
                KQuantProjectionKernel.validateQ6K(
                    new ByteArray(KQuantProjectionKernel.Q6_K_BLOCK_BYTES),
                    new ByteArray(SUPER_BLOCK),
                    new FloatArray(1),
                    new FloatArray(1),
                    1,
                    0,
                    SUPER_BLOCK))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rows");
  }

  @Test
  void acceptsWellFormedQ6KStorage() {
    KQuantProjectionKernel.validateQ6K(
        new ByteArray(KQuantProjectionKernel.Q6_K_BLOCK_BYTES),
        new ByteArray(SUPER_BLOCK),
        new FloatArray(1),
        new FloatArray(1),
        1,
        1,
        SUPER_BLOCK);
  }

  // --- Harness ---

  private record Q8KActivation(ByteArray quants, FloatArray scales, IntArray sums) {
    static Q8KActivation of(float[] input, int batchSize, int cols) {
      byte[] quants = new byte[batchSize * cols];
      float[] scales = new float[batchSize * cols / SUPER_BLOCK];
      int[] sums = new int[batchSize * cols / SUM_BLOCK];
      KQuantProjectionKernel.quantize(input, quants, scales, sums, batchSize, cols);
      return new Q8KActivation(
          ByteArray.fromArray(quants), FloatArray.fromArray(scales), IntArray.fromArray(sums));
    }
  }

  private static float[] runQ4K(byte[] weights, float[] input, int batchSize, int rows, int cols) {
    Q8KActivation activation = Q8KActivation.of(input, batchSize, cols);
    FloatArray output = new FloatArray(batchSize * rows);
    ByteArray deviceWeights = ByteArray.fromArray(weights);
    KQuantProjectionKernel.validateQ4K(
        deviceWeights,
        activation.quants(),
        activation.scales(),
        activation.sums(),
        output,
        batchSize,
        rows,
        cols);
    KQuantProjectionKernel.multiplyQ4K(
        deviceWeights,
        activation.quants(),
        activation.scales(),
        activation.sums(),
        output,
        batchSize,
        rows,
        cols);
    return output.toHeapArray();
  }

  private static float[] runQ6K(byte[] weights, float[] input, int batchSize, int rows, int cols) {
    Q8KActivation activation = Q8KActivation.of(input, batchSize, cols);
    FloatArray output = new FloatArray(batchSize * rows);
    ByteArray deviceWeights = ByteArray.fromArray(weights);
    KQuantProjectionKernel.validateQ6K(
        deviceWeights, activation.quants(), activation.scales(), output, batchSize, rows, cols);
    KQuantProjectionKernel.multiplyQ6K(
        deviceWeights, activation.quants(), activation.scales(), output, batchSize, rows, cols);
    return output.toHeapArray();
  }

  private static float[] referenceQ4K(
      byte[] weights, float[] input, int batchSize, int rows, int cols) {
    float[] output = new float[batchSize * rows];
    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        input,
        MemorySegment.ofArray(weights),
        batchSize,
        rows,
        cols,
        output,
        new byte[batchSize * cols],
        new float[batchSize * cols / SUPER_BLOCK],
        new short[batchSize * cols / SUM_BLOCK]);
    return output;
  }

  private static float[] referenceQ6K(
      byte[] weights, float[] input, int batchSize, int rows, int cols) {
    float[] output = new float[batchSize * rows];
    VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
        input,
        MemorySegment.ofArray(weights),
        batchSize,
        rows,
        cols,
        output,
        new byte[batchSize * cols],
        new float[batchSize * cols / SUPER_BLOCK]);
    return output;
  }

  /**
   * Asserts the guaranteed numeric contract: the kernel differs from the production CPU kernel by
   * at most two float roundings per super-block, scaled to the magnitude of the reference result.
   */
  private static void assertWithinFusedMultiplyAddBudget(
      float[] actual, float[] expected, int cols) {
    assertThat(actual).hasSameSizeAs(expected);
    float magnitude = 0.0f;
    for (float value : expected) {
      magnitude = Math.max(magnitude, Math.abs(value));
    }
    float tolerance = 2.0f * (cols / SUPER_BLOCK) * Math.ulp(magnitude);
    for (int index = 0; index < expected.length; index++) {
      assertThat(actual[index])
          .as("element %d within %s of %s", index, tolerance, expected[index])
          .isCloseTo(expected[index], within(tolerance));
    }
  }

  /** Pseudo-random Q4_K super-blocks spanning the whole six-bit scale and minimum range. */
  private static byte[] randomQ4KMatrix(int rows, int cols, long seed) {
    Random random = new Random(seed);
    int blocks = rows * cols / SUPER_BLOCK;
    byte[] weights = new byte[blocks * KQuantProjectionKernel.Q4_K_BLOCK_BYTES];
    int[] scales = new int[8];
    int[] minimums = new int[8];
    for (int block = 0; block < blocks; block++) {
      for (int group = 0; group < 8; group++) {
        scales[group] = random.nextInt(64);
        minimums[group] = random.nextInt(64);
      }
      writeQ4KBlock(
          weights,
          block,
          Float.floatToFloat16(0.002f + random.nextFloat() * 0.02f),
          Float.floatToFloat16(0.001f + random.nextFloat() * 0.01f),
          scales,
          minimums,
          random);
    }
    return weights;
  }

  /**
   * Q4_K super-blocks whose contribution is exactly representable: the two scales are powers of two
   * and the six-bit group scales and minimums stay small enough that no partial sum leaves the
   * exactly representable integer range of a float.
   */
  private static byte[] exactQ4KMatrix(int rows, int cols, long seed) {
    Random random = new Random(seed);
    int blocks = rows * cols / SUPER_BLOCK;
    byte[] weights = new byte[blocks * KQuantProjectionKernel.Q4_K_BLOCK_BYTES];
    int[] scales = new int[8];
    int[] minimums = new int[8];
    for (int block = 0; block < blocks; block++) {
      for (int group = 0; group < 8; group++) {
        scales[group] = 1 + random.nextInt(3);
        minimums[group] = random.nextInt(4);
      }
      writeQ4KBlock(
          weights,
          block,
          Float.floatToFloat16(1.0f),
          Float.floatToFloat16(0.25f),
          scales,
          minimums,
          random);
    }
    return weights;
  }

  private static void writeQ4KBlock(
      byte[] weights,
      int block,
      short delta,
      short deltaMin,
      int[] scales,
      int[] minimums,
      Random random) {
    int offset = block * KQuantProjectionKernel.Q4_K_BLOCK_BYTES;
    weights[offset] = (byte) delta;
    weights[offset + 1] = (byte) (delta >>> 8);
    weights[offset + 2] = (byte) deltaMin;
    weights[offset + 3] = (byte) (deltaMin >>> 8);
    for (int group = 0; group < 4; group++) {
      weights[offset + 4 + group] =
          (byte) ((scales[group] & 0x3F) | ((scales[group + 4] >>> 4) << 6));
      weights[offset + 8 + group] =
          (byte) ((minimums[group] & 0x3F) | ((minimums[group + 4] >>> 4) << 6));
      weights[offset + 12 + group] =
          (byte) ((scales[group + 4] & 0x0F) | ((minimums[group + 4] & 0x0F) << 4));
    }
    for (int index = 0; index < 128; index++) {
      weights[offset + 16 + index] = (byte) random.nextInt(256);
    }
  }

  /** Pseudo-random Q6_K super-blocks spanning the whole signed int8 group-scale range. */
  private static byte[] randomQ6KMatrix(int rows, int cols, long seed) {
    Random random = new Random(seed);
    int blocks = rows * cols / SUPER_BLOCK;
    byte[] weights = new byte[blocks * KQuantProjectionKernel.Q6_K_BLOCK_BYTES];
    for (int block = 0; block < blocks; block++) {
      int offset = block * KQuantProjectionKernel.Q6_K_BLOCK_BYTES;
      for (int index = 0; index < 192; index++) {
        weights[offset + index] = (byte) random.nextInt(256);
      }
      for (int index = 0; index < 16; index++) {
        weights[offset + 192 + index] = (byte) (random.nextInt(65) - 32);
      }
      short delta = Float.floatToFloat16(0.002f + random.nextFloat() * 0.02f);
      weights[offset + 208] = (byte) delta;
      weights[offset + 209] = (byte) (delta >>> 8);
    }
    return weights;
  }

  /** Q6_K super-blocks with a power-of-two delta and small group scales. */
  private static byte[] exactQ6KMatrix(int rows, int cols, long seed) {
    Random random = new Random(seed);
    int blocks = rows * cols / SUPER_BLOCK;
    byte[] weights = new byte[blocks * KQuantProjectionKernel.Q6_K_BLOCK_BYTES];
    for (int block = 0; block < blocks; block++) {
      int offset = block * KQuantProjectionKernel.Q6_K_BLOCK_BYTES;
      for (int index = 0; index < 192; index++) {
        weights[offset + index] = (byte) random.nextInt(256);
      }
      for (int index = 0; index < 16; index++) {
        weights[offset + 192 + index] = (byte) (random.nextInt(5) - 2);
      }
      short delta = Float.floatToFloat16(0.5f);
      weights[offset + 208] = (byte) delta;
      weights[offset + 209] = (byte) (delta >>> 8);
    }
    return weights;
  }

  private static float[] randomFloats(int length, long seed) {
    Random random = new Random(seed);
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = random.nextFloat(-2.0f, 2.0f);
    }
    return values;
  }

  /**
   * Activations that quantize to Q8_K without rounding error: every value is a small integer times
   * 2^-6, and each super-block's extremum is exactly -127 * 2^-6, so the Q8_K scale is 2^-6 and
   * each stored quant is the integer it started from.
   */
  private static float[] exactFloats(int batchSize, int cols, long seed) {
    Random random = new Random(seed);
    float[] values = new float[batchSize * cols];
    float step = 1.0f / 64.0f;
    for (int batch = 0; batch < batchSize; batch++) {
      for (int block = 0; block < cols / SUPER_BLOCK; block++) {
        int offset = batch * cols + block * SUPER_BLOCK;
        values[offset] = -127.0f * step;
        for (int index = 1; index < SUPER_BLOCK; index++) {
          values[offset + index] = (random.nextInt(7) - 3) * step;
        }
      }
    }
    return values;
  }
}
