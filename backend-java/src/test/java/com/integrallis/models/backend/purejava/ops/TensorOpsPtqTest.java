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
package com.integrallis.models.backend.purejava.ops;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TensorOpsPtqTest {
  @Test
  void sharedParallelAndConfinedSerialProductsAreBitIdentical() {
    int rows = 2048;
    int columns = 4096;
    int rowBytes = columns / 128 * 28;
    byte[] packed = new byte[rows * rowBytes];
    Random random = new Random(631);
    random.nextBytes(packed);
    for (int block = 0; block < packed.length / 28; block++) {
      packed[block * 28 + 26] = 0;
      packed[block * 28 + 27] = 0x30;
    }
    float[] input = new float[columns];
    for (int i = 0; i < columns; i++) input[i] = random.nextFloat() * 2 - 1;
    try (Arena confined = Arena.ofConfined();
        Arena shared = Arena.ofShared()) {
      MemorySegment serial = confined.allocate(packed.length);
      MemorySegment parallel = shared.allocate(packed.length);
      MemorySegment.copy(packed, 0, serial, ValueLayout.JAVA_BYTE, 0, packed.length);
      MemorySegment.copy(packed, 0, parallel, ValueLayout.JAVA_BYTE, 0, packed.length);
      assertThat(TensorOps.packedProjectionThreads(rows, rowBytes, serial)).isEqualTo(1);
      assertThat(TensorOps.packedProjectionThreads(rows, rowBytes, parallel))
          .isEqualTo(Math.min(rows, Runtime.getRuntime().availableProcessors()));
      float[] expected = new float[rows];
      float[] actual = new float[rows];
      TensorOps.ggufMatmul(expected, input, serial, GgufTensorType.PTQ1_0, rows, columns);
      TensorOps.ggufMatmul(actual, input, parallel, GgufTensorType.PTQ1_0, rows, columns);
      assertThat(actual).containsExactly(expected);
    }
  }
}
