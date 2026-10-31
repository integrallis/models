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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class QuantizedProjectionTest {

  private static final float TOLERANCE = 1.0e-5f;

  @Test
  void projectsAnF32WeightThroughTheSharedKernel() {
    // F32 so the expected value is exact and the test is about wiring, not dequantisation.
    // Weights are row-major: row r dotted with the input gives out[r].
    try (Arena arena = Arena.ofConfined()) {
      float[][] weight = {{1f, 2f}, {3f, 4f}, {5f, 6f}};
      QuantizedProjection projection =
          new QuantizedProjection(segment(arena, weight), GgufTensorType.F32, 3, 2, "probe");
      float[] out = new float[3];

      projection.apply(new float[] {1f, 1f}, out);

      assertThat(out).usingComparatorWithPrecision(TOLERANCE).containsExactly(3f, 7f, 11f);
    }
  }

  @Test
  void rejectsAnInputOfTheWrongWidth() {
    try (Arena arena = Arena.ofConfined()) {
      QuantizedProjection projection =
          new QuantizedProjection(
              segment(arena, new float[][] {{1f, 2f}}), GgufTensorType.F32, 1, 2, "probe");

      assertThatThrownBy(() -> projection.apply(new float[3], new float[1]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("probe")
          .hasMessageContaining("expects an input of 2");
    }
  }

  @Test
  void rejectsAnOutputOfTheWrongWidth() {
    try (Arena arena = Arena.ofConfined()) {
      QuantizedProjection projection =
          new QuantizedProjection(
              segment(arena, new float[][] {{1f, 2f}}), GgufTensorType.F32, 1, 2, "probe");

      assertThatThrownBy(() -> projection.apply(new float[2], new float[5]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("produces 1");
    }
  }

  @Test
  void refusesAHadamardRotatedCheckpoint() {
    // A rotated weight projected without its transform yields finite activations and fluent
    // output, so this has to fail at load rather than drift at inference.
    assertThatThrownBy(() -> QuantizedProjection.requireNoHadamard(true))
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("prism.hadamard.version");
  }

  @Test
  void acceptsACheckpointWithoutHadamard() {
    QuantizedProjection.requireNoHadamard(false);
  }

  private static MemorySegment segment(Arena arena, float[][] rows) {
    int columns = rows[0].length;
    MemorySegment segment = arena.allocate((long) rows.length * columns * Float.BYTES, Float.BYTES);
    int index = 0;
    for (float[] row : rows) {
      for (float value : row) {
        segment.setAtIndex(ValueLayout.JAVA_FLOAT, index++, value);
      }
    }
    return segment;
  }
}
