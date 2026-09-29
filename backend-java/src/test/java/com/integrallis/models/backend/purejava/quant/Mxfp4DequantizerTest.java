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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * MXFP4, pinned against the three details that are not guessable from the block struct.
 *
 * <p>Each is a silent-wrongness trap: get any of them wrong and the weights still dequantize to
 * plausible numbers, and the model produces slightly worse output rather than an error.
 */
@Tag("unit")
class Mxfp4DequantizerTest {

  private static MemorySegment block(Arena arena, int scaleCode, int[] packedBytes) {
    MemorySegment segment = arena.allocate(Mxfp4Dequantizer.BLOCK_BYTES);
    segment.set(ValueLayout.JAVA_BYTE, 0, (byte) scaleCode);
    for (int i = 0; i < packedBytes.length; i++) {
      segment.set(ValueLayout.JAVA_BYTE, 1 + i, (byte) packedBytes[i]);
    }
    return segment;
  }

  @Test
  void blockGeometryMatchesTheFormat() {
    assertThat(Mxfp4Dequantizer.BLOCK_SIZE).isEqualTo(32);
    assertThat(Mxfp4Dequantizer.BLOCK_BYTES)
        .describedAs("one E8M0 scale byte plus 16 packed bytes")
        .isEqualTo(17);
  }

  @Test
  void theTwoNibblesOfAByteAreSixteenElementsApartNotAdjacent() {
    try (Arena arena = Arena.ofConfined()) {
      // Byte 0 only: low nibble code 2 (value 2), high nibble code 5 (value 6). Scale code 128
      // gives
      // 2^(128-128) = 1.0, so the doubled table values appear directly.
      int[] packed = new int[16];
      packed[0] = (5 << 4) | 2;
      float[] out = new float[32];

      Mxfp4Dequantizer.dequantize(block(arena, 128, packed), 0, out, 0, 32);

      assertThat(out[0]).describedAs("low nibble is element 0").isEqualTo(2.0f);
      assertThat(out[16]).describedAs("high nibble is element 16, not element 1").isEqualTo(6.0f);
      assertThat(out[1]).isZero();
      for (int i = 1; i < 16; i++) {
        assertThat(out[i]).isZero();
        assertThat(out[i + 16]).isZero();
      }
    }
  }

  @Test
  void everyCodeMapsToTheDoubledE2m1MagnitudeWithSignInTheHighBit() {
    try (Arena arena = Arena.ofConfined()) {
      // Codes 0..15 laid into the low nibbles of bytes 0..15, so they land in elements 0..15.
      int[] packed = new int[16];
      for (int code = 0; code < 16; code++) {
        packed[code] = code;
      }
      float[] out = new float[32];

      Mxfp4Dequantizer.dequantize(block(arena, 128, packed), 0, out, 0, 32);

      // Doubled magnitudes, not the true E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6}: the halved scale
      // compensates, and mixing the two conventions is silently off by a factor of two.
      float[] expected = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
      for (int code = 0; code < 16; code++) {
        assertThat(out[code]).describedAs("code %d", code).isEqualTo(expected[code]);
      }
    }
  }

  @Test
  void theScaleIsHalvedAndKeepsItsDenormalsAtTheBottomOfTheRange() {
    try (Arena arena = Arena.ofConfined()) {
      int[] one = new int[16];
      one[0] = 2; // code 2 -> doubled value 2
      float[] out = new float[32];

      // code 128 -> 2^0 = 1.0
      Mxfp4Dequantizer.dequantize(block(arena, 128, one), 0, out, 0, 32);
      assertThat(out[0]).isEqualTo(2.0f);

      // code 129 -> 2^1 = 2.0
      Mxfp4Dequantizer.dequantize(block(arena, 129, one), 0, out, 0, 32);
      assertThat(out[0]).isEqualTo(4.0f);

      // code 127 -> 2^-1 = 0.5, i.e. HALF of 2^(127-127); an unhalved decode would give 2.0 here.
      Mxfp4Dequantizer.dequantize(block(arena, 127, one), 0, out, 0, 32);
      assertThat(out[0]).isEqualTo(1.0f);

      // The denormal special case: codes 0 and 1 are built by shifting, not by exponent arithmetic.
      Mxfp4Dequantizer.dequantize(block(arena, 0, one), 0, out, 0, 32);
      assertThat(out[0]).isEqualTo(2.0f * Float.intBitsToFloat(0x00200000));
      assertThat(out[0]).describedAs("denormal, not zero and not infinity").isNotZero();
      Mxfp4Dequantizer.dequantize(block(arena, 1, one), 0, out, 0, 32);
      assertThat(out[0]).isEqualTo(2.0f * Float.intBitsToFloat(0x00400000));
    }
  }

  @Test
  void refusesACountThatIsNotAWholeNumberOfBlocks() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = block(arena, 128, new int[16]);

      assertThatThrownBy(() -> Mxfp4Dequantizer.dequantize(segment, 0, new float[40], 0, 40))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("32");
    }
  }

  @Test
  void theDotProductAgreesWithDequantisingThenMultiplying() {
    // The two read the same bytes in the same split-halves order, so they must not drift apart.
    // This
    // is the whole safety argument for consuming the packed format directly in the matmul.
    try (Arena arena = Arena.ofConfined()) {
      java.util.Random rng = new java.util.Random(20260929L);
      int blocks = 5;
      int count = blocks * Mxfp4Dequantizer.BLOCK_SIZE;
      MemorySegment segment = arena.allocate((long) blocks * Mxfp4Dequantizer.BLOCK_BYTES);
      for (int block = 0; block < blocks; block++) {
        long base = (long) block * Mxfp4Dequantizer.BLOCK_BYTES;
        // Scale codes around the denormal boundary as well as ordinary ones.
        segment.set(ValueLayout.JAVA_BYTE, base, (byte) (120 + block));
        for (int j = 0; j < Mxfp4Dequantizer.BLOCK_SIZE / 2; j++) {
          segment.set(ValueLayout.JAVA_BYTE, base + 1 + j, (byte) rng.nextInt(256));
        }
      }
      float[] vector = new float[count];
      for (int i = 0; i < count; i++) {
        vector[i] = (rng.nextFloat() - 0.5f) * 3.0f;
      }

      float[] weights = new float[count];
      Mxfp4Dequantizer.dequantize(segment, 0, weights, 0, count);
      float expected = 0.0f;
      for (int i = 0; i < count; i++) {
        expected += weights[i] * vector[i];
      }

      assertThat(Mxfp4Dequantizer.dotProduct(segment, 0, vector, 0, count))
          .isEqualTo(expected, within(1.0e-3f));
    }
  }

  @Test
  void theDotProductIsExactForASingleKnownBlock() {
    // Scale code 129 is 2^1 against the doubled table, so the halving in e8m0ToFloatHalf makes the
    // effective factor 1.0 and the expected values are the table entries themselves.
    try (Arena arena = Arena.ofConfined()) {
      int[] packed = new int[Mxfp4Dequantizer.BLOCK_SIZE / 2];
      packed[0] = 0x01; // low nibble 1 -> element 0; high nibble 0 -> element 16
      MemorySegment segment = block(arena, 129, packed);
      float[] vector = new float[Mxfp4Dequantizer.BLOCK_SIZE];
      vector[0] = 2.0f;

      float[] weights = new float[Mxfp4Dequantizer.BLOCK_SIZE];
      Mxfp4Dequantizer.dequantize(segment, 0, weights, 0, Mxfp4Dequantizer.BLOCK_SIZE);

      assertThat(Mxfp4Dequantizer.dotProduct(segment, 0, vector, 0, Mxfp4Dequantizer.BLOCK_SIZE))
          .isEqualTo(weights[0] * 2.0f);
    }
  }

  @Test
  void aRunThatIsNotAWholeNumberOfBlocksIsRefused() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = block(arena, 128, new int[Mxfp4Dequantizer.BLOCK_SIZE / 2]);
      float[] vector = new float[Mxfp4Dequantizer.BLOCK_SIZE];

      assertThatThrownBy(() -> Mxfp4Dequantizer.dotProduct(segment, 0, vector, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of the 32-weight MXFP4 block");
    }
  }
}
