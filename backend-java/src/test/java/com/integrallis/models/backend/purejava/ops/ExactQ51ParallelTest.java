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
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ExactQ51ParallelTest {
  @Test
  void oversizedDecodedScratchRetainsItsUnusedTail() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment weights = arena.allocate(24);
      weights.fill((byte) 0);
      float[] decoded = new float[1536];
      java.util.Arrays.fill(decoded, 17.0f);
      GgufTensorValues.dequantizeRow(weights, GgufTensorType.Q5_1, 0, 32, decoded);
      assertThat(java.util.Arrays.copyOf(decoded, 32)).containsOnly(0.0f);
      assertThat(java.util.Arrays.copyOfRange(decoded, 32, decoded.length)).containsOnly(17.0f);
    }
  }

  @Test
  void sharedParallelAndConfinedSerialKeepEveryFloatBitAcrossRepeatedBatches() {
    for (int cols : new int[] {256, 128, 256}) {
      int rows = 128;
      int batch = 65;
      byte[] packed = new byte[rows * cols / 32 * 24];
      var bytes = java.nio.ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
      for (int block = 0; block < rows * cols / 32; block++) {
        int base = block * 24;
        bytes.putShort(base, (short) (0x3000 | (block * 37 & 1023)));
        bytes.putShort(
            base + 2, (short) ((block % 2 == 0 ? 0 : 0x8000) | 0x3000 | (block * 53 & 1023)));
        bytes.putInt(base + 4, block * 0x9e3779b9);
        for (int n = 0; n < 16; n++) packed[base + 8 + n] = (byte) (block * 13 + n * 29);
      }
      float[] input = new float[batch * cols];
      var random = new java.util.SplittableRandom(918329);
      for (int n = 0; n < input.length; n++) input[n] = (float) random.nextDouble(-2, 2);
      try (Arena shared = Arena.ofShared();
          Arena confined = Arena.ofConfined()) {
        MemorySegment common = shared.allocateFrom(ValueLayout.JAVA_BYTE, packed);
        MemorySegment local = confined.allocateFrom(ValueLayout.JAVA_BYTE, packed);
        float[] expected = new float[batch * rows];
        float[] actual = new float[expected.length];
        float[] decoded = new float[cols];
        for (int repeat = 0; repeat < 3; repeat++) {
          input[repeat] += 0.0625f;
          TensorOps.ggufExactBatchedMatmul(
              expected, input, local, GgufTensorType.Q5_1, batch, rows, cols, decoded);
          TensorOps.ggufExactBatchedMatmul(
              actual, input, common, GgufTensorType.Q5_1, batch, rows, cols, decoded);
          assertThat(actual).containsExactly(expected);
        }
        assertThat(actual[0]).isNotEqualTo(actual[rows]);
      }
    }
  }
}
