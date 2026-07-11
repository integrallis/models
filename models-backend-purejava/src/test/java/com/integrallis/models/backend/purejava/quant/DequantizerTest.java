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
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class DequantizerTest {

  @Nested
  static class Q4_0 {

    @Test
    void allZeroBlockProducesNegativeEights() {
      // Block: scale=1.0 (f16 0x3C00), all nibble bytes = 0x00
      // Each nibble is 0, so value = (0 - 8) * 1.0 = -8.0
      byte[] block = new byte[18];
      ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(0, (short) 0x3C00);
      // nibbles are already 0

      var segment = Arena.ofConfined().allocate(18);
      MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 18);

      float[] dst = new float[32];
      new Q4_0Dequantizer().dequantize(segment, 0, dst, 0, 32);

      for (float v : dst) {
        assertThat(v).isEqualTo(-8.0f);
      }
    }

    @Test
    void knownNibblePatternWithScale() {
      // GGML stores the low nibbles at positions 0..15 and high nibbles at 16..31.
      // Block: scale=2.0 (f16 0x4000), first nibble byte = 0x98 (lo=8,hi=9).
      byte[] block = new byte[18];
      ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(0, (short) 0x4000);
      block[2] = (byte) 0x98; // lo=8, hi=9

      var segment = Arena.ofConfined().allocate(18);
      MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 18);

      float[] dst = new float[32];
      new Q4_0Dequantizer().dequantize(segment, 0, dst, 0, 32);

      assertThat(dst[0]).isCloseTo(0.0f, within(0.001f));
      assertThat(dst[1]).isCloseTo(-16.0f, within(0.001f));
      assertThat(dst[16]).isCloseTo(2.0f, within(0.001f));
    }

    @Test
    void multiBlockDequantization() {
      // 2 blocks = 64 values, 36 bytes
      byte[] data = new byte[36];
      ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
      buf.putShort(0, (short) 0x3C00); // block 0 scale=1.0
      buf.putShort(18, (short) 0x4000); // block 1 scale=2.0

      var segment = Arena.ofConfined().allocate(36);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 36);

      float[] dst = new float[64];
      new Q4_0Dequantizer().dequantize(segment, 0, dst, 0, 64);

      // Block 0: all nibbles 0, so all values = (0-8)*1.0 = -8.0
      assertThat(dst[0]).isEqualTo(-8.0f);
      // Block 1: all nibbles 0, so all values = (0-8)*2.0 = -16.0
      assertThat(dst[32]).isEqualTo(-16.0f);
    }
  }

  @Nested
  static class Q8_0 {

    @Test
    void knownScaleAndQuants() {
      // Block: scale=0.5 (f16 0x3800), quant[0]=4, quant[1]=-2
      byte[] block = new byte[34];
      ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(0, (short) 0x3800);
      block[2] = 4;
      block[3] = (byte) -2;

      var segment = Arena.ofConfined().allocate(34);
      MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 34);

      float[] dst = new float[32];
      new Q8_0Dequantizer().dequantize(segment, 0, dst, 0, 32);

      assertThat(dst[0]).isCloseTo(2.0f, within(0.001f)); // 4 * 0.5
      assertThat(dst[1]).isCloseTo(-1.0f, within(0.001f)); // -2 * 0.5
    }

    @Test
    void zeroScaleProducesZeros() {
      byte[] block = new byte[34];
      // scale = 0 (f16 0x0000)
      block[2] = 127; // even with max quant, scale=0 means zero output

      var segment = Arena.ofConfined().allocate(34);
      MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 34);

      float[] dst = new float[32];
      new Q8_0Dequantizer().dequantize(segment, 0, dst, 0, 32);

      for (float v : dst) {
        assertThat(v).isEqualTo(0.0f);
      }
    }
  }

  @Nested
  static class F16 {

    @Test
    void dequantizesKnownValues() {
      // Write 1.0, 2.0, 0.5 as f16
      byte[] data = new byte[6];
      ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
      buf.putShort(0, (short) 0x3C00); // 1.0
      buf.putShort(2, (short) 0x4000); // 2.0
      buf.putShort(4, (short) 0x3800); // 0.5

      var segment = Arena.ofConfined().allocate(6);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 6);

      float[] dst = new float[3];
      new F16Dequantizer().dequantize(segment, 0, dst, 0, 3);

      assertThat(dst[0]).isEqualTo(1.0f);
      assertThat(dst[1]).isEqualTo(2.0f);
      assertThat(dst[2]).isEqualTo(0.5f);
    }
  }
}
