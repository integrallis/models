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
package com.integrallis.models.backend.purejava.gptoss;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.Mxfp4Matrix;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GptOssMxfp4MoeTest {

  private static final int HIDDEN = 32;
  private static final int INTERMEDIATE = 32;
  private static final float ALPHA = 1.702f;
  private static final float LIMIT = 7.0f;

  @Test
  void exactRoutedLayerMatchesIndependentDequantizedReference() {
    GptOssMxfp4ExpertWeights weights = weights();
    GptOssMxfp4Moe moe = new GptOssMxfp4Moe(weights, ALPHA, LIMIT);
    float[] hidden = hidden();
    int[] selected = {1, 0};
    float[] routing = {0.65f, 0.35f};
    float[] expected = reference(weights, hidden, selected, routing);
    float[] actual = filled(HIDDEN, 99.0f);

    moe.forwardExact(hidden, selected, routing, actual);

    assertThat(actual).containsExactly(expected, within(2.0e-5f));
  }

  @Test
  void w4a8RoutedLayerTracksTheExactPathWithoutExpandingWeights() {
    GptOssMxfp4ExpertWeights weights = weights();
    GptOssMxfp4Moe moe = new GptOssMxfp4Moe(weights, ALPHA, LIMIT);
    float[] hidden = hidden();
    int[] selected = {1, 0};
    float[] routing = {0.65f, 0.35f};
    float[] exact = new float[HIDDEN];
    float[] approximate = new float[HIDDEN];

    moe.forwardExact(hidden, selected, routing, exact);
    moe.forwardQ8(hidden, selected, routing, approximate);

    assertThat(cosine(exact, approximate)).isGreaterThan(0.9999f);
    assertThat(maximumAbsoluteDifference(exact, approximate)).isLessThan(0.06f);
  }

  @Test
  void rejectsAliasingDuplicateRoutesAndMismatchedBuffers() {
    GptOssMxfp4Moe moe = new GptOssMxfp4Moe(weights(), ALPHA, LIMIT);
    float[] hidden = hidden();

    assertThatThrownBy(() -> moe.forwardExact(hidden, new int[] {0}, new float[] {1.0f}, hidden))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("alias");
    assertThatThrownBy(
            () ->
                moe.forwardExact(
                    hidden, new int[] {0, 0}, new float[] {0.5f, 0.5f}, new float[HIDDEN]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("duplicate");
    assertThatThrownBy(
            () ->
                moe.forwardExact(
                    new float[HIDDEN - 1], new int[] {0}, new float[] {1.0f}, new float[HIDDEN]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("hidden");
  }

  private static GptOssMxfp4ExpertWeights weights() {
    return GptOssMxfp4ExpertWeights.of(
        new GptOssMxfp4ExpertWeights.Expert(
            matrix(2 * INTERMEDIATE, HIDDEN, 1, 2, 126),
            sequence(2 * INTERMEDIATE, -0.16f, 0.005f),
            matrix(HIDDEN, INTERMEDIATE, 3, 4, 125),
            sequence(HIDDEN, -0.08f, 0.005f)),
        new GptOssMxfp4ExpertWeights.Expert(
            matrix(2 * INTERMEDIATE, HIDDEN, 5, 6, 125),
            sequence(2 * INTERMEDIATE, 0.12f, -0.004f),
            matrix(HIDDEN, INTERMEDIATE, 6, 5, 124),
            sequence(HIDDEN, 0.06f, -0.003f)));
  }

  /**
   * A matrix whose rows genuinely differ, which the uniform {@link #matrix} cannot provide.
   *
   * <p>{@code matrix} fills every block with one byte, so every row is identical -- and then
   * splitting the fused gate/up by row parity and splitting it into halves give the same two
   * matrices. A test built on it passes whichever split the code does, which is what it did until
   * this existed.
   */
  private static Mxfp4Matrix varyingMatrix(int rows, int columns, int seed) {
    byte[] blocks = new byte[rows * columns / 2];
    byte[] scales = new byte[rows * columns / 32];
    int blockBytesPerRow = columns / 2;
    int scaleBytesPerRow = columns / 32;
    for (int row = 0; row < rows; row++) {
      for (int index = 0; index < blockBytesPerRow; index++) {
        int even = (seed + row * 5 + index * 3) % 16;
        int odd = (seed + row * 7 + index * 11 + 4) % 16;
        blocks[row * blockBytesPerRow + index] = (byte) ((odd << 4) | even);
      }
      for (int index = 0; index < scaleBytesPerRow; index++) {
        // Around 127 (an exponent near 1.0), so the values stay in a sane range.
        scales[row * scaleBytesPerRow + index] = (byte) (124 + ((seed + row) % 5));
      }
    }
    return Mxfp4Matrix.of(
        MemorySegment.ofArray(blocks), MemorySegment.ofArray(scales), rows, columns);
  }

  private static Mxfp4Matrix matrix(
      int rows, int columns, int evenCode, int oddCode, int scaleCode) {
    byte[] blocks = new byte[rows * columns / 2];
    byte[] scales = new byte[rows * columns / 32];
    Arrays.fill(blocks, (byte) ((oddCode << 4) | evenCode));
    Arrays.fill(scales, (byte) scaleCode);
    return Mxfp4Matrix.of(
        MemorySegment.ofArray(blocks), MemorySegment.ofArray(scales), rows, columns);
  }

  private static float[] reference(
      GptOssMxfp4ExpertWeights weights, float[] hidden, int[] selected, float[] routing) {
    float[] output = new float[HIDDEN];
    for (int route = 0; route < selected.length; route++) {
      GptOssMxfp4ExpertWeights.Expert expert = weights.expert(selected[route]);
      float[] gateUp = multiply(expert.gateUp(), hidden, expert.gateUpBias());
      float[] activation = new float[INTERMEDIATE];
      for (int index = 0; index < activation.length; index++) {
        float gate = Math.min(gateUp[2 * index], LIMIT);
        float up = Math.max(-LIMIT, Math.min(gateUp[2 * index + 1], LIMIT));
        activation[index] = gate * (float) (1.0 / (1.0 + Math.exp(-ALPHA * gate))) * (up + 1.0f);
      }
      float[] down = multiply(expert.down(), activation, expert.downBias());
      for (int index = 0; index < output.length; index++) {
        output[index] += routing[route] * down[index];
      }
    }
    return output;
  }

  private static float[] multiply(Mxfp4Matrix matrix, float[] input, float[] bias) {
    float[] output = new float[matrix.rows()];
    for (int row = 0; row < matrix.rows(); row++) {
      float sum = bias[row];
      for (int column = 0; column < matrix.columns(); column++) {
        sum += matrix.value(row, column) * input[column];
      }
      output[row] = sum;
    }
    return output;
  }

  private static float[] hidden() {
    float[] values = new float[HIDDEN];
    for (int index = 0; index < values.length; index++) {
      values[index] = (index - HIDDEN / 2) * 0.013f + (index % 3) * 0.002f;
    }
    return values;
  }

  private static float[] sequence(int length, float start, float increment) {
    float[] values = new float[length];
    for (int index = 0; index < values.length; index++) {
      values[index] = start + index * increment;
    }
    return values;
  }

  private static float[] filled(int length, float value) {
    float[] values = new float[length];
    Arrays.fill(values, value);
    return values;
  }

  private static float cosine(float[] left, float[] right) {
    double dot = 0.0;
    double leftNorm = 0.0;
    double rightNorm = 0.0;
    for (int index = 0; index < left.length; index++) {
      dot += left[index] * right[index];
      leftNorm += left[index] * left[index];
      rightNorm += right[index] * right[index];
    }
    return (float) (dot / Math.sqrt(leftNorm * rightNorm));
  }

  private static float maximumAbsoluteDifference(float[] left, float[] right) {
    float maximum = 0.0f;
    for (int index = 0; index < left.length; index++) {
      maximum = Math.max(maximum, Math.abs(left[index] - right[index]));
    }
    return maximum;
  }

  /**
   * The split (GGUF) expert shape must compute what the fused (safetensors) one does.
   *
   * <p>The two shapes exist because the two artifacts store the same weights differently:
   * safetensors fuses gate and up into one tensor whose rows interleave them, GGUF ships two
   * tensors. Converting either way would copy the largest tensors in the model, so both are read in
   * place -- which means two code paths, which means they need a test that they agree.
   *
   * <p>The split weights are extracted from the fused matrix by multiplying it by basis vectors, so
   * the numbers are exactly what MXFP4 dequantizes to rather than a second guess at the format.
   * Rows are split by parity: even rows are the gate, odd rows the up.
   */
  @Test
  void theSplitExpertShapeMatchesTheFusedOne() {
    GptOssMxfp4ExpertWeights fused = varyingWeights();
    GptOssMxfp4ExpertWeights split = splitEquivalent(fused);

    float[] hidden = sequence(HIDDEN, -0.4f, 0.03f);
    int[] selected = {1, 0};
    float[] routing = {0.6f, 0.4f};
    float[] fromFused = new float[HIDDEN];
    float[] fromSplit = new float[HIDDEN];

    new GptOssMxfp4Moe(fused, ALPHA, LIMIT).forwardExact(hidden, selected, routing, fromFused);
    new GptOssMxfp4Moe(split, ALPHA, LIMIT).forwardExact(hidden, selected, routing, fromSplit);

    for (int index = 0; index < HIDDEN; index++) {
      assertThat(fromSplit[index])
          .describedAs("split output %s must match the fused shape", index)
          // 1e-4, not exact: the two paths run different kernels over the same numbers -- an MXFP4
          // matmul and an F32 one -- so they sum in a different order. Measured difference is
          // 1.1e-5;
          // reading the weights with the wrong row parity moves the output by orders of magnitude
          // more.
          .isEqualTo(fromFused[index], within(1.0e-4f));
    }
    // And the output is not trivially zero, or the comparison proves nothing.
    float largest = 0.0f;
    for (float value : fromFused) {
      largest = Math.max(largest, Math.abs(value));
    }
    assertThat(largest).isGreaterThan(1.0e-3f);
  }

  /**
   * A mixed set -- one fused expert and one split -- is refused rather than read with one indexing.
   */
  @Test
  void expertsOfDifferentStorageShapesAreRefused() {
    GptOssMxfp4ExpertWeights fused = varyingWeights();
    GptOssMxfp4ExpertWeights split = splitEquivalent(fused);
    GptOssMxfp4ExpertWeights mixed = GptOssMxfp4ExpertWeights.of(fused.expert(0), split.expert(1));

    assertThatThrownBy(() -> new GptOssMxfp4Moe(mixed, ALPHA, LIMIT))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("storage shape");
  }

  /** Experts whose gate and up rows differ, so a wrong split is visible. */
  private static GptOssMxfp4ExpertWeights varyingWeights() {
    return GptOssMxfp4ExpertWeights.of(
        new GptOssMxfp4ExpertWeights.Expert(
            varyingMatrix(2 * INTERMEDIATE, HIDDEN, 3),
            sequence(2 * INTERMEDIATE, -0.16f, 0.005f),
            varyingMatrix(HIDDEN, INTERMEDIATE, 8),
            sequence(HIDDEN, -0.08f, 0.005f)),
        new GptOssMxfp4ExpertWeights.Expert(
            varyingMatrix(2 * INTERMEDIATE, HIDDEN, 5),
            sequence(2 * INTERMEDIATE, 0.12f, -0.004f),
            varyingMatrix(HIDDEN, INTERMEDIATE, 9),
            sequence(HIDDEN, 0.06f, -0.003f)));
  }

  /**
   * Proves the fused fixture's rows actually differ by parity.
   *
   * <p>Without this the equivalence test above can silently stop discriminating -- which it did,
   * when it was built on the uniform fixture.
   */
  @Test
  void theVaryingFixtureDistinguishesParityFromHalves() {
    float[][] rows = rowsOf(varyingMatrix(2 * INTERMEDIATE, HIDDEN, 3), 2 * INTERMEDIATE, HIDDEN);

    assertThat(rows[0]).isNotEqualTo(rows[1]);
    // Row 1 is the first `up` row under a parity split and the second `gate` row under a halves
    // split.
    assertThat(rows[1]).isNotEqualTo(rows[INTERMEDIATE]);
  }

  /** Rebuilds the same experts in the split shape, reading the fused weights out exactly. */
  private static GptOssMxfp4ExpertWeights splitEquivalent(GptOssMxfp4ExpertWeights fused) {
    GptOssMxfp4ExpertWeights.Expert[] experts =
        new GptOssMxfp4ExpertWeights.Expert[fused.expertCount()];
    for (int index = 0; index < experts.length; index++) {
      GptOssMxfp4ExpertWeights.Expert expert = fused.expert(index);
      float[][] gateUpRows = rowsOf(expert.gateUp(), 2 * INTERMEDIATE, HIDDEN);
      float[] gate = new float[INTERMEDIATE * HIDDEN];
      float[] up = new float[INTERMEDIATE * HIDDEN];
      float[] gateBias = new float[INTERMEDIATE];
      float[] upBias = new float[INTERMEDIATE];
      for (int row = 0; row < INTERMEDIATE; row++) {
        System.arraycopy(gateUpRows[2 * row], 0, gate, row * HIDDEN, HIDDEN);
        System.arraycopy(gateUpRows[2 * row + 1], 0, up, row * HIDDEN, HIDDEN);
        gateBias[row] = expert.gateUpBias()[2 * row];
        upBias[row] = expert.gateUpBias()[2 * row + 1];
      }
      float[][] downRows = rowsOf(expert.down(), HIDDEN, INTERMEDIATE);
      float[] down = new float[HIDDEN * INTERMEDIATE];
      for (int row = 0; row < HIDDEN; row++) {
        System.arraycopy(downRows[row], 0, down, row * INTERMEDIATE, INTERMEDIATE);
      }
      experts[index] =
          new GptOssMxfp4ExpertWeights.Expert(
              float32(gate, INTERMEDIATE, HIDDEN),
              gateBias,
              float32(up, INTERMEDIATE, HIDDEN),
              upBias,
              float32(down, HIDDEN, INTERMEDIATE),
              expert.downBias());
    }
    return GptOssMxfp4ExpertWeights.of(experts);
  }

  /**
   * Reads a matrix out row by row by multiplying it by basis vectors.
   *
   * <p>So the extracted numbers are exactly what the MXFP4 reader produces, rather than a second
   * implementation of the block format that could be wrong in the same direction as the first.
   */
  private static float[][] rowsOf(Mxfp4Matrix matrix, int rows, int columns) {
    float[][] result = new float[rows][columns];
    float[] basis = new float[columns];
    float[] product = new float[rows];
    for (int column = 0; column < columns; column++) {
      Arrays.fill(basis, 0.0f);
      basis[column] = 1.0f;
      matrix.multiply(basis, product);
      for (int row = 0; row < rows; row++) {
        result[row][column] = product[row];
      }
    }
    return result;
  }

  private static GptOssProjection float32(float[] values, int rows, int columns) {
    java.nio.ByteBuffer buffer =
        java.nio.ByteBuffer.allocate(values.length * Float.BYTES)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putFloat(value);
    }
    return GptOssProjection.ofGguf(
        MemorySegment.ofArray(buffer.array()),
        com.integrallis.models.backend.purejava.gguf.GgufTensorType.F32,
        rows,
        columns);
  }
}
