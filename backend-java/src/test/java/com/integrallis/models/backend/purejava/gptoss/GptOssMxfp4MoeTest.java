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
}
