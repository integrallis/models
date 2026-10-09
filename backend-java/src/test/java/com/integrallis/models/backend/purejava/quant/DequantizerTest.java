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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
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
  class Q4_0 {

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

    @Test
    void rejectsPartialBlocksInsteadOfSilentlyDroppingValues() {
      MemorySegment segment = Arena.ofConfined().allocate(18);
      float[] dst = new float[31];

      assertThatThrownBy(() -> new Q4_0Dequantizer().dequantize(segment, 0, dst, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
    }
  }

  @Nested
  class Q8_0 {

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

    @Test
    void rejectsPartialBlocksInsteadOfSilentlyDroppingValues() {
      MemorySegment segment = Arena.ofConfined().allocate(34);
      float[] dst = new float[31];

      assertThatThrownBy(() -> new Q8_0Dequantizer().dequantize(segment, 0, dst, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
    }
  }

  @Nested
  class F16 {

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

  @Nested
  class Q4_1 {

    /**
     * Pins the one thing {@code Q4_1} does differently from {@code Q4_0}, which is also the one
     * thing that fails silently: the value is {@code q * d + m} with an unsigned quant, where
     * {@code Q4_0} centres its quants by subtracting eight. Applying that bias here would shift
     * every weight in the block by {@code -8 * d} and still produce plausible text.
     */
    @Test
    void appliesMinimumRatherThanCentringTheQuant() {
      // d = 1.0 (0x3C00), m = -3.0 (0xC200), every nibble zero.
      byte[] block = new byte[20];
      ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
      buffer.putShort(0, (short) 0x3C00);
      buffer.putShort(2, (short) 0xC200);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment segment = arena.allocate(20);
        MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 20);
        float[] dst = new float[32];
        new Q4_1Dequantizer().dequantize(segment, 0, dst, 0, 32);
        for (float value : dst) {
          // 0 * 1.0 + (-3.0). A Q4_0-style centring bias would have produced -8.0.
          assertThat(value).isEqualTo(-3.0f);
        }
      }
    }

    @Test
    void splitsLowAndHighNibblesAcrossTheBlockHalves() {
      // d = 1.0, m = 0.0, nibble byte 0 = 0x9C -> low 0xC = 12, high 0x9 = 9.
      byte[] block = new byte[20];
      ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
      buffer.putShort(0, (short) 0x3C00);
      block[4] = (byte) 0x9C;

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment segment = arena.allocate(20);
        MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 20);
        float[] dst = new float[32];
        new Q4_1Dequantizer().dequantize(segment, 0, dst, 0, 32);
        assertThat(dst[0]).as("low nibble is element j").isEqualTo(12.0f);
        assertThat(dst[16]).as("high nibble is element j + 16, not j + 1").isEqualTo(9.0f);
        assertThat(dst[1]).isEqualTo(0.0f);
      }
    }

    @Test
    void reachesTheFullFourBitRangeAndIsUnsigned() {
      // d = 1.0, m = 0.0, nibble byte = 0xFF -> both halves 15, the maximum. Unsigned: a signed
      // read would give -1.
      byte[] block = new byte[20];
      ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(0, (short) 0x3C00);
      block[4] = (byte) 0xFF;

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment segment = arena.allocate(20);
        MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 20);
        float[] dst = new float[32];
        new Q4_1Dequantizer().dequantize(segment, 0, dst, 0, 32);
        assertThat(dst[0]).isEqualTo(15.0f);
        assertThat(dst[16]).isEqualTo(15.0f);
      }
    }
  }

  @Nested
  class Q5_1 {

    /**
     * Pins the three things {@code Q5_1} does differently from {@code Q4_0} and {@code Q5_0}, each
     * transcribed from {@code dequantize_row_q5_1} in llama.cpp rather than derived: the value is
     * {@code q * d + m} with an unsigned quant and no centring bias, the fifth bits live in a
     * separate four-byte {@code qh} word, and element {@code j + 16} takes bit {@code j + 16} of it
     * rather than a bit adjacent to its own nibble.
     */
    @Test
    void appliesMinimumRatherThanCentringTheQuant() {
      // d = 1.0 (0x3C00), m = -3.0 (0xC200), qh = 0 so every fifth bit is clear, nibbles all 0.
      byte[] block = new byte[24];
      ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
      buffer.putShort(0, (short) 0x3C00);
      buffer.putShort(2, (short) 0xC200);

      var segment = Arena.ofConfined().allocate(24);
      MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 24);

      float[] dst = new float[32];
      new Q5_1Dequantizer().dequantize(segment, 0, dst, 0, 32);

      // 0 * 1.0 + (-3.0). A Q4_0-style centring bias would have produced -8.0 here.
      for (float value : dst) {
        assertThat(value).isEqualTo(-3.0f);
      }
    }

    @Test
    void readsTheFifthBitFromTheHighWordForBothHalves() {
      // d = 1.0, m = 0.0, nibble byte 0 = 0x21 (lo=1, hi=2).
      // qh bit 0 set    -> element 0 gains 16  -> 1 + 16 = 17
      // qh bit 16 set   -> element 16 gains 16 -> 2 + 16 = 18
      byte[] block = new byte[24];
      ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
      buffer.putShort(0, (short) 0x3C00);
      buffer.putShort(2, (short) 0x0000);
      buffer.putInt(4, (1 << 0) | (1 << 16));
      block[8] = (byte) 0x21;

      var segment = Arena.ofConfined().allocate(24);
      MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 24);

      float[] dst = new float[32];
      new Q5_1Dequantizer().dequantize(segment, 0, dst, 0, 32);

      assertThat(dst[0]).as("low nibble plus its fifth bit").isEqualTo(17.0f);
      assertThat(dst[16]).as("high nibble plus bit j+16, not an adjacent bit").isEqualTo(18.0f);
      assertThat(dst[1]).as("an unset fifth bit leaves the nibble alone").isEqualTo(0.0f);
    }

    @Test
    void reachesTheFullFiveBitRange() {
      // d = 1.0, m = 0.0, nibble byte = 0xFF, both fifth bits set -> 15 + 16 = 31, the maximum.
      byte[] block = new byte[24];
      ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
      buffer.putShort(0, (short) 0x3C00);
      buffer.putInt(4, (1 << 0) | (1 << 16));
      block[8] = (byte) 0xFF;

      var segment = Arena.ofConfined().allocate(24);
      MemorySegment.copy(block, 0, segment, ValueLayout.JAVA_BYTE, 0, 24);

      float[] dst = new float[32];
      new Q5_1Dequantizer().dequantize(segment, 0, dst, 0, 32);

      assertThat(dst[0]).isEqualTo(31.0f);
      assertThat(dst[16]).isEqualTo(31.0f);
    }
  }
}
