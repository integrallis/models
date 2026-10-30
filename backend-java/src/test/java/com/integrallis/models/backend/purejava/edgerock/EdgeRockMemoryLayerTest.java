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
package com.integrallis.models.backend.purejava.edgerock;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.edgerock.EdgeRockMemoryLayer.Projection;
import com.integrallis.models.backend.purejava.edgerock.EdgeRockMemoryLayer.Weights;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class EdgeRockMemoryLayerTest {

  private static final int EMBEDDING = 8;
  private static final int HEADS = 2;
  private static final int GROUPS = 1;
  private static final int DIM = 4;
  private static final int ROPE = 2;
  private static final int CAPACITY = 4;
  private static final int CHUNK = 1;
  private static final int RANK = 2;
  private static final int WRITER = 4;
  private static final float TOLERANCE = 1.0e-5f;

  @Test
  void aTokenNeverAttendsToItself() {
    // The read runs before the write and against committed state only. With chunk size 1 the
    // write commits immediately, so if the order were reversed the very first token would read
    // itself and this would be non-zero.
    EdgeRockMemoryLayer layer = new EdgeRockMemoryLayer(config(), identityish());
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    float[] out = new float[EMBEDDING];

    layer.forward(ones(EMBEDDING), 0, cos(), sin(), bank, out);

    assertThat(out).usingComparatorWithPrecision(TOLERANCE).containsOnly(0f);
    // ...and the token did land in the bank, so the read being zero is order, not a no-op.
    assertThat(bank.selection().occupancy()).isEqualTo(1);
  }

  @Test
  void theSecondTokenReadsTheFirst() {
    EdgeRockMemoryLayer layer = new EdgeRockMemoryLayer(config(), identityish());
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    float[] out = new float[EMBEDDING];

    layer.forward(ones(EMBEDDING), 0, cos(), sin(), bank, out);
    Arrays.fill(out, 0f);
    layer.forward(ones(EMBEDDING), 1, cos(), sin(), bank, out);

    // One committed record, tied against the sink, so the read is half the stored value and the
    // output projection passes it through. The exact magnitude depends on the norms; the point
    // is that it is no longer zero.
    assertThat(maxMagnitude(out)).isGreaterThan(TOLERANCE);
    assertThat(bank.selection().occupancy()).isEqualTo(2);
  }

  @Test
  void anUntrainedLayerContributesExactlyNothing() {
    // read_scale and write_down and admission_up are all zero-initialised in the reference. A
    // freshly initialised memory layer must therefore be the identity on the residual stream --
    // not small, exactly zero -- or every layer would inject noise before training.
    Weights untrained =
        new Weights(
            ones(EMBEDDING),
            projection(EMBEDDING, HEADS * DIM, 1f),
            ones(DIM),
            projection(EMBEDDING, GROUPS * DIM, 1f),
            ones(DIM),
            projection(EMBEDDING, GROUPS * DIM, 1f),
            projection(HEADS * DIM, EMBEDDING, 1f),
            projection(EMBEDDING, WRITER, 1f),
            projection(EMBEDDING, WRITER, 1f),
            projection(WRITER, EMBEDDING, 0f),
            projection(EMBEDDING, RANK, 1f),
            projection(RANK, GROUPS, 0f),
            new float[][] {new float[DIM], new float[DIM]},
            new float[HEADS],
            new float[HEADS]);
    EdgeRockMemoryLayer layer = new EdgeRockMemoryLayer(config(), untrained);
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    float[] out = new float[EMBEDDING];

    layer.forward(ones(EMBEDDING), 0, cos(), sin(), bank, out);
    Arrays.fill(out, 0f);
    layer.forward(ones(EMBEDDING), 1, cos(), sin(), bank, out);

    assertThat(out).usingComparatorWithPrecision(TOLERANCE).containsOnly(0f);
  }

  @Test
  void admissionScoresAllTokensEquallyWhenTheScorerIsFlat() {
    EdgeRockMemoryLayer layer = new EdgeRockMemoryLayer(config(), identityish());
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    float[] out = new float[EMBEDDING];

    for (int position = 0; position < CAPACITY; position++) {
      layer.forward(ones(EMBEDDING), position, cos(), sin(), bank, out);
    }

    // Identical inputs give identical scores, so the bank fills in position order and the
    // tie-break keeps the earliest -- which is what makes a full bank stable rather than churning.
    assertThat(bank.selection().positionsCopy()).containsExactly(0, 1, 2, 3);
  }

  private static EdgeRockConfig config() {
    return new EdgeRockConfig(
        4, EMBEDDING, 16, HEADS, GROUPS, DIM, 32, 128, 1.0e-6f, 10_000f, ROPE * 2, List.of(), 4, 8,
        GROUPS, 2, 8, 4, 8, CAPACITY, CHUNK, RANK, WRITER);
  }

  /** Weights that pass signal through, so the test observes order rather than arithmetic. */
  private static Weights identityish() {
    return new Weights(
        ones(EMBEDDING),
        projection(EMBEDDING, HEADS * DIM, 0.5f),
        ones(DIM),
        projection(EMBEDDING, GROUPS * DIM, 0.5f),
        ones(DIM),
        projection(EMBEDDING, GROUPS * DIM, 0.5f),
        projection(HEADS * DIM, EMBEDDING, 0.5f),
        projection(EMBEDDING, WRITER, 0.25f),
        projection(EMBEDDING, WRITER, 0.25f),
        projection(WRITER, EMBEDDING, 0f),
        projection(EMBEDDING, RANK, 0.5f),
        projection(RANK, GROUPS, 0.5f),
        new float[][] {new float[DIM], new float[DIM]},
        new float[HEADS],
        fill(HEADS, 1f));
  }

  /** A projection whose every weight is {@code value}; out[j] = value * sum(in). */
  private static Projection projection(int inputs, int outputs, float value) {
    return (input, output) -> {
      if (input.length != inputs || output.length != outputs) {
        throw new IllegalArgumentException(
            "projection expects "
                + inputs
                + "->"
                + outputs
                + " but got "
                + input.length
                + "->"
                + output.length);
      }
      float sum = 0f;
      for (float x : input) {
        sum += x;
      }
      Arrays.fill(output, value * sum);
    };
  }

  private static float maxMagnitude(float[] values) {
    float most = 0f;
    for (float value : values) {
      most = Math.max(most, Math.abs(value));
    }
    return most;
  }

  private static float[] ones(int length) {
    return fill(length, 1f);
  }

  private static float[] fill(int length, float value) {
    float[] out = new float[length];
    Arrays.fill(out, value);
    return out;
  }

  private static float[] cos() {
    return fill(ROPE, 1f);
  }

  private static float[] sin() {
    return new float[ROPE];
  }
}
