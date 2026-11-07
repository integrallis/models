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
package com.integrallis.models.backend.purejava.quant;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class Q8_0QuantizerTest {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Test
  void roundTripQuantization() {
    float[] original = new float[64]; // 2 blocks of 32
    for (int i = 0; i < 32; i++) {
      original[i] = (i - 16) * 0.1f; // Range -1.6 to +1.5
    }
    for (int i = 32; i < 64; i++) {
      original[i] = (i - 48) * 0.05f; // Range -0.8 to +0.75
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment q8_0 = Q8_0Quantizer.quantize(original, 2, 32, arena);

      // Dequantize and compare
      float[] recovered = new float[64];
      GgufTensorValues.dequantizeRow(q8_0, GgufTensorType.Q8_0, 0, 32, recovered);
      // Second row starts at offset 34 bytes
      // Need to dequantize from offset 34
      // Actually dequantizeRow assumes row-major with contiguous rows.
      // Let me use Q8_0Dequantizer directly for the second block.
      com.integrallis.models.backend.purejava.quant.Q8_0Dequantizer deq = new Q8_0Dequantizer();
      deq.dequantize(q8_0, 34, recovered, 32, 32);

      // Q8_0 has limited precision; check relative error
      for (int i = 0; i < 64; i++) {
        float relError = Math.abs(original[i] - recovered[i]) / (Math.abs(original[i]) + 1e-6f);
        assertThat(relError).isLessThan(0.02f); // < 2% relative error
      }
    }
  }

  @Test
  void zeroBlockProducesZeroQuantizedValues() {
    float[] zeros = new float[32];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment q8_0 = Q8_0Quantizer.quantize(zeros, 1, 32, arena);
      // Scale should be 1.0 (fallback for zero max)
      short scaleBits = q8_0.get(LE_SHORT, 0);
      float scale = Float16.toFloat(scaleBits);
      assertThat(scale).isEqualTo(1.0f);

      // All quantized values should be 0
      for (int i = 0; i < 32; i++) {
        byte q = q8_0.get(ValueLayout.JAVA_BYTE, 2 + i);
        assertThat(q).isEqualTo((byte) 0);
      }
    }
  }

  @Test
  void blockStructureIsCorrect() {
    float[] weights = new float[96]; // 3 blocks
    for (int i = 0; i < 96; i++) weights[i] = (float) Math.sin(i * 0.1);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment q8_0 = Q8_0Quantizer.quantize(weights, 3, 32, arena);
      assertThat(q8_0.byteSize()).isEqualTo(3L * 34); // 3 blocks * 34 bytes

      // Verify each block has a valid scale
      for (int block = 0; block < 3; block++) {
        long offset = block * 34L;
        short scaleBits = q8_0.get(LE_SHORT, offset);
        float scale = Float16.toFloat(scaleBits);
        assertThat(scale).isGreaterThan(0.0f);
      }
    }
  }
}
