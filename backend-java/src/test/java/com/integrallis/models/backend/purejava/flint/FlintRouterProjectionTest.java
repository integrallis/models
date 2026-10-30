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
package com.integrallis.models.backend.purejava.flint;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.flint.FlintWeights.Matrix;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FlintRouterProjectionTest {
  @Test
  void cachedScoresRetainMappedReductionBitsAcrossWidthsAndRepeatedInputs() {
    Random random = new Random(1729);
    for (int columns : new int[] {4, 17, 32, 128, 2048}) {
      int rows = 256;
      try (var arena = Arena.ofConfined()) {
        MemorySegment data = arena.allocate((long) rows * columns * Float.BYTES);
        for (int i = 0; i < rows * columns; i++)
          data.setAtIndex(
              ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN),
              i,
              (random.nextFloat() - .5f) * (i % 2 == 0 ? 100f : .01f));
        Matrix matrix = new Matrix(data, GgufTensorType.F32, rows, columns);
        var cached = FlintRouterProjection.prepare(matrix, data.byteSize());
        assertThat(cached.bytes()).isEqualTo(data.byteSize());
        float[] input = new float[columns], expected = new float[rows], actual = new float[rows];
        for (int repeat = 0; repeat < 30; repeat++) {
          for (int i = 0; i < columns; i++) input[i] = random.nextFloat() - .5f;
          TensorOps.ggufMatmul(
              expected,
              input,
              data,
              GgufTensorType.F32,
              rows,
              columns,
              null,
              null,
              null,
              null,
              GgufQ4Kernel.WIDENED);
          cached.multiply(input, actual);
          for (int row = 0; row < rows; row++)
            assertThat(Float.floatToRawIntBits(actual[row]))
                .as("width=%s repeat=%s row=%s", columns, repeat, row)
                .isEqualTo(Float.floatToRawIntBits(expected[row]));
        }
        assertThat(FlintRouterProjection.prepare(matrix, data.byteSize() - 1)).isNull();
        assertThat(
                FlintRouterProjection.prepare(
                    new Matrix(data, GgufTensorType.F16, rows, columns), Long.MAX_VALUE))
            .isNull();
      }
    }
  }
}
