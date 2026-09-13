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
package com.integrallis.models.backend.purejava.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LoraProjectionTest {

  @Test
  void addsScaledLowRankDeltaWithoutReplacingBaseProjection() {
    // A [2 x 3], B [2 x 2], x [3]. A*x = [8, 1], B*(A*x) = [10, -7].
    LoraProjection projection =
        new LoraProjection(
            MemorySegment.ofArray(new float[] {1, 2, 3, -1, 0, 2}),
            MemorySegment.ofArray(new float[] {1, 2, -1, 1}),
            3,
            2,
            2,
            0.5f);
    float[] baseOutput = {10, 20};

    projection.addTo(baseOutput, 0, new float[] {1, 2, 1}, 0);

    assertThat(baseOutput).containsExactly(new float[] {15, 16.5f}, within(1.0e-6f));
  }

  @Test
  void supportsRowsInsideBatchMajorBuffersWithoutAllocatingCallerSlices() {
    LoraProjection projection =
        new LoraProjection(
            MemorySegment.ofArray(new float[] {1, 0, 0, 1}),
            MemorySegment.ofArray(new float[] {2, 0, 0, 3}),
            2,
            2,
            2,
            1.0f);
    float[] inputs = {99, 1, 2, 98};
    float[] outputs = {77, 10, 20, 76};

    projection.addTo(outputs, 1, inputs, 1);

    assertThat(outputs).containsExactly(77, 12, 26, 76);
  }

  @Test
  void rejectsShapeAndRangeMismatchesBeforeTouchingOutput() {
    assertThatThrownBy(
            () ->
                new LoraProjection(
                    MemorySegment.ofArray(new float[5]),
                    MemorySegment.ofArray(new float[4]),
                    3,
                    2,
                    2,
                    1.0f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("A byte size");

    LoraProjection projection =
        new LoraProjection(
            MemorySegment.ofArray(new float[4]),
            MemorySegment.ofArray(new float[2]),
            2,
            1,
            2,
            1.0f);
    assertThatThrownBy(() -> projection.addTo(new float[1], 1, new float[2], 0))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }
}
