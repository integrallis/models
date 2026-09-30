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
import static org.assertj.core.data.Offset.offset;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Ternary and 2-bit dequantization against the reference algorithms.
 *
 * <p>The expected values are not hand-derived from the struct layouts. They come from an
 * independent transcription of ggml's own dequantize_row_* functions -- upstream for TQ1_0 and
 * TQ2_0, the PrismML fork for PTQ1_0 and PQ2_0 -- run over pseudo-random block bytes with the scale
 * pinned to 0.125, which is exact in fp16 so the oracle carries no rounding of its own. Two
 * independent transcriptions agreeing is the evidence; one implementation checked against my own
 * reading of the packing would only prove I read it the same way twice.
 *
 * <p>The blocks are random rather than crafted, so every base-3 position and both high-byte paths
 * are exercised and a byte-order or interleave mistake cannot hide behind a symmetric fixture.
 */
@Tag("unit")
class TernaryDequantizerTest {

  @Test
  void dequantizePtq1_0MatchesTheReference() {
    byte[] block = {
      (byte) 0x8f,
      (byte) 0x10,
      (byte) 0x6d,
      (byte) 0x30,
      (byte) 0xe3,
      (byte) 0xb4,
      (byte) 0x64,
      (byte) 0xab,
      (byte) 0x10,
      (byte) 0xec,
      (byte) 0x0e,
      (byte) 0x85,
      (byte) 0xcd,
      (byte) 0x32,
      (byte) 0x1d,
      (byte) 0x60,
      (byte) 0x8a,
      (byte) 0x0c,
      (byte) 0x79,
      (byte) 0xf6,
      (byte) 0x93,
      (byte) 0xab,
      (byte) 0x58,
      (byte) 0x6a,
      (byte) 0x5b,
      (byte) 0x70,
      (byte) 0x00,
      (byte) 0x30
    };
    float[] first16 = {
      0.000000f,
      -0.125000f,
      0.000000f,
      -0.125000f,
      0.125000f,
      0.125000f,
      0.000000f,
      0.125000f,
      -0.125000f,
      0.125000f,
      -0.125000f,
      0.000000f,
      0.125000f,
      -0.125000f,
      -0.125000f,
      0.000000f
    };
    float[] last8 = {
      0.000000f, 0.000000f, -0.125000f, -0.125000f, -0.125000f, 0.125000f, 0.000000f, 0.125000f
    };

    float[] out = new float[128];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(block.length);
      MemorySegment.copy(block, 0, seg, ValueLayout.JAVA_BYTE, 0, block.length);
      TernaryDequantizer.dequantizePtq1_0(seg, 0, out, 0, 128);
    }

    assertThat(Arrays.copyOfRange(out, 0, 16)).containsExactly(first16, offset(0.0f));
    assertThat(Arrays.copyOfRange(out, 128 - 8, 128)).containsExactly(last8, offset(0.0f));
    float sum = 0.0f;
    for (float value : out) {
      sum += value;
    }
    assertThat(sum).isEqualTo(-2.125000f, offset(1.0e-4f));
  }

  @Test
  void dequantizePq2_0MatchesTheReference() {
    byte[] block = {
      (byte) 0x00,
      (byte) 0x30,
      (byte) 0x80,
      (byte) 0xd9,
      (byte) 0x00,
      (byte) 0x04,
      (byte) 0xb2,
      (byte) 0x18,
      (byte) 0x77,
      (byte) 0x7c,
      (byte) 0x00,
      (byte) 0xf4,
      (byte) 0xd8,
      (byte) 0xdb,
      (byte) 0x08,
      (byte) 0x2e,
      (byte) 0xb9,
      (byte) 0xa5,
      (byte) 0x56,
      (byte) 0x41,
      (byte) 0x2e,
      (byte) 0x20,
      (byte) 0xb6,
      (byte) 0xbd,
      (byte) 0x36,
      (byte) 0x2e,
      (byte) 0x6a,
      (byte) 0x08,
      (byte) 0x8e,
      (byte) 0x44,
      (byte) 0x32,
      (byte) 0xe9,
      (byte) 0x17,
      (byte) 0x51
    };
    float[] first16 = {
      -0.125000f,
      -0.125000f,
      -0.125000f,
      0.125000f,
      0.000000f,
      0.125000f,
      0.000000f,
      0.250000f,
      -0.125000f,
      -0.125000f,
      -0.125000f,
      -0.125000f,
      -0.125000f,
      0.000000f,
      -0.125000f,
      -0.125000f
    };
    float[] last8 = {
      0.250000f, 0.000000f, 0.000000f, -0.125000f, 0.000000f, -0.125000f, 0.000000f, 0.000000f
    };

    float[] out = new float[128];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(block.length);
      MemorySegment.copy(block, 0, seg, ValueLayout.JAVA_BYTE, 0, block.length);
      TernaryDequantizer.dequantizePq2_0(seg, 0, out, 0, 128);
    }

    assertThat(Arrays.copyOfRange(out, 0, 16)).containsExactly(first16, offset(0.0f));
    assertThat(Arrays.copyOfRange(out, 128 - 8, 128)).containsExactly(last8, offset(0.0f));
    float sum = 0.0f;
    for (float value : out) {
      sum += value;
    }
    assertThat(sum).isEqualTo(4.625000f, offset(1.0e-4f));
  }

  @Test
  void dequantizeTq1_0MatchesTheReference() {
    byte[] block = {
      (byte) 0xae,
      (byte) 0x33,
      (byte) 0x50,
      (byte) 0x31,
      (byte) 0x51,
      (byte) 0xab,
      (byte) 0x2e,
      (byte) 0x5e,
      (byte) 0xbe,
      (byte) 0xb3,
      (byte) 0x87,
      (byte) 0x82,
      (byte) 0x6c,
      (byte) 0x73,
      (byte) 0x00,
      (byte) 0x67,
      (byte) 0xde,
      (byte) 0x84,
      (byte) 0x0a,
      (byte) 0xcb,
      (byte) 0x09,
      (byte) 0x72,
      (byte) 0x68,
      (byte) 0xcb,
      (byte) 0x3d,
      (byte) 0xfd,
      (byte) 0xdd,
      (byte) 0xbe,
      (byte) 0x20,
      (byte) 0x9e,
      (byte) 0x97,
      (byte) 0x05,
      (byte) 0x48,
      (byte) 0xf6,
      (byte) 0xd6,
      (byte) 0xcc,
      (byte) 0xc7,
      (byte) 0xdc,
      (byte) 0x3b,
      (byte) 0xc3,
      (byte) 0x04,
      (byte) 0xc3,
      (byte) 0x5a,
      (byte) 0xed,
      (byte) 0x9c,
      (byte) 0x20,
      (byte) 0x8c,
      (byte) 0xce,
      (byte) 0x98,
      (byte) 0x7b,
      (byte) 0x1e,
      (byte) 0x3c,
      (byte) 0x00,
      (byte) 0x30
    };
    float[] first16 = {
      0.125000f,
      -0.125000f,
      -0.125000f,
      -0.125000f,
      -0.125000f,
      0.125000f,
      -0.125000f,
      0.000000f,
      0.125000f,
      0.125000f,
      0.000000f,
      0.000000f,
      0.000000f,
      0.000000f,
      -0.125000f,
      0.000000f
    };
    float[] last8 = {
      0.000000f, -0.125000f, -0.125000f, -0.125000f, -0.125000f, 0.125000f, -0.125000f, -0.125000f
    };

    float[] out = new float[256];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(block.length);
      MemorySegment.copy(block, 0, seg, ValueLayout.JAVA_BYTE, 0, block.length);
      TernaryDequantizer.dequantizeTq1_0(seg, 0, out, 0, 256);
    }

    assertThat(Arrays.copyOfRange(out, 0, 16)).containsExactly(first16, offset(0.0f));
    assertThat(Arrays.copyOfRange(out, 256 - 8, 256)).containsExactly(last8, offset(0.0f));
    float sum = 0.0f;
    for (float value : out) {
      sum += value;
    }
    assertThat(sum).isEqualTo(-1.875000f, offset(1.0e-4f));
  }

  @Test
  void dequantizeTq2_0MatchesTheReference() {
    byte[] block = {
      (byte) 0x09,
      (byte) 0x03,
      (byte) 0xef,
      (byte) 0x4d,
      (byte) 0xe6,
      (byte) 0xb8,
      (byte) 0xba,
      (byte) 0x6c,
      (byte) 0x5d,
      (byte) 0x82,
      (byte) 0x09,
      (byte) 0xf5,
      (byte) 0x29,
      (byte) 0xe7,
      (byte) 0xf5,
      (byte) 0x71,
      (byte) 0xd5,
      (byte) 0xaf,
      (byte) 0xa8,
      (byte) 0x36,
      (byte) 0xe0,
      (byte) 0x4b,
      (byte) 0x2d,
      (byte) 0x4e,
      (byte) 0xc3,
      (byte) 0x64,
      (byte) 0xca,
      (byte) 0x3f,
      (byte) 0xf3,
      (byte) 0xa3,
      (byte) 0x46,
      (byte) 0x55,
      (byte) 0x23,
      (byte) 0x92,
      (byte) 0xc3,
      (byte) 0xd8,
      (byte) 0xf7,
      (byte) 0x8a,
      (byte) 0x4c,
      (byte) 0x73,
      (byte) 0xde,
      (byte) 0xf4,
      (byte) 0x33,
      (byte) 0xc0,
      (byte) 0xdf,
      (byte) 0xa0,
      (byte) 0x40,
      (byte) 0x4e,
      (byte) 0x7b,
      (byte) 0xa3,
      (byte) 0x5f,
      (byte) 0x28,
      (byte) 0xce,
      (byte) 0x39,
      (byte) 0x40,
      (byte) 0x5a,
      (byte) 0x89,
      (byte) 0x99,
      (byte) 0x37,
      (byte) 0xda,
      (byte) 0x19,
      (byte) 0x2e,
      (byte) 0x1b,
      (byte) 0x92,
      (byte) 0x00,
      (byte) 0x30
    };
    float[] first16 = {
      0.000000f,
      0.250000f,
      0.250000f,
      0.000000f,
      0.125000f,
      -0.125000f,
      0.125000f,
      -0.125000f,
      0.000000f,
      0.125000f,
      0.000000f,
      0.000000f,
      0.000000f,
      0.250000f,
      0.000000f,
      0.000000f
    };
    float[] last8 = {
      0.125000f, 0.125000f, -0.125000f, 0.250000f, -0.125000f, -0.125000f, -0.125000f, 0.125000f
    };

    float[] out = new float[256];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(block.length);
      MemorySegment.copy(block, 0, seg, ValueLayout.JAVA_BYTE, 0, block.length);
      TernaryDequantizer.dequantizeTq2_0(seg, 0, out, 0, 256);
    }

    assertThat(Arrays.copyOfRange(out, 0, 16)).containsExactly(first16, offset(0.0f));
    assertThat(Arrays.copyOfRange(out, 256 - 8, 256)).containsExactly(last8, offset(0.0f));
    float sum = 0.0f;
    for (float value : out) {
      sum += value;
    }
    assertThat(sum).isEqualTo(17.125000f, offset(1.0e-4f));
  }

  @Test
  void refusesACountThatIsNotAWholeNumberOfBlocks() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(256);
      assertThatThrownBy(() -> TernaryDequantizer.dequantizePtq1_0(seg, 0, new float[64], 0, 64))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("128-weight block size");
      assertThatThrownBy(() -> TernaryDequantizer.dequantizeTq1_0(seg, 0, new float[128], 0, 128))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("256-weight block size");
    }
  }

  @Test
  void theFourthTwoBitCodeIsPlusTwoNotClampedToOne() {
    // PQ2_0 is not ternary: 0b11 means +2. A ternary reading clamps it to +1 and is then quietly
    // wrong on every weight that uses the fourth state.
    byte[] block = new byte[34];
    block[0] = (byte) 0x00;
    block[1] = (byte) 0x3c; // fp16 1.0
    block[2] = (byte) 0xE4; // codes 0,1,2,3 across the byte
    float[] out = new float[128];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(block.length);
      MemorySegment.copy(block, 0, seg, ValueLayout.JAVA_BYTE, 0, block.length);
      TernaryDequantizer.dequantizePq2_0(seg, 0, out, 0, 128);
    }
    assertThat(Arrays.copyOfRange(out, 0, 4)).containsExactly(-1.0f, 0.0f, 1.0f, 2.0f);
  }
}
