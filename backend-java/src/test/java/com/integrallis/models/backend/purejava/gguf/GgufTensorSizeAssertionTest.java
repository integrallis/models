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
package com.integrallis.models.backend.purejava.gguf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Per-tensor size assertions: every tensor's byte length is derived from its type's block layout
 * and its shape, and a data region that is short, overlaps the next tensor, or runs past the file
 * fails loudly with the tensor name, the expected length, and the bytes actually available.
 */
@Tag("unit")
class GgufTensorSizeAssertionTest {

  @Test
  void truncatedLastTensorNamesTheTensorExpectedAndAvailableBytes() {
    byte[] valid =
        new SyntheticGgufBuilder()
            .addTensor("blk.0.ffn_up.weight", GgufTensorType.F32, new long[] {2}, new byte[8])
            .build();
    byte[] truncated = Arrays.copyOf(valid, valid.length - 3);

    assertThatThrownBy(() -> GgufParser.parseSegment(MemorySegment.ofArray(truncated)))
        .isInstanceOf(MalformedGgufException.class)
        .hasMessageContaining("'blk.0.ffn_up.weight'")
        .hasMessageContaining("expected 8 bytes")
        .hasMessageContaining("available 5")
        .hasMessageContaining("end of file");
  }

  @Test
  void quantizedTensorShorterThanItsBlockLayoutIsRejectedWithBlockArithmetic() {
    // 64 Q4_0 elements are two 18-byte blocks; only one block is present.
    byte[] data =
        new SyntheticGgufBuilder()
            .addTensor("blk.3.ffn_down.weight", GgufTensorType.Q4_0, new long[] {64}, new byte[18])
            .build();

    assertThatThrownBy(() -> GgufParser.parseSegment(MemorySegment.ofArray(data)))
        .isInstanceOf(MalformedGgufException.class)
        .hasMessageContaining("'blk.3.ffn_down.weight'")
        .hasMessageContaining("expected 36 bytes")
        .hasMessageContaining("available 18")
        .hasMessageContaining("Q4_0");
  }

  @Test
  void tensorOverlappingTheNextTensorIsRejectedEvenWhenBothEndInsideTheFile() {
    // Cold expert "a" claims 8 bytes but "b" starts 4 bytes after it. Both ranges are inside the
    // file, so a file-bounds check alone accepts this and "a" silently reads b's bytes.
    byte[] data =
        new SyntheticGgufBuilder()
            .addUint32("general.alignment", 4)
            .addTensor(
                "blk.0.ffn_gate_exps.weight", GgufTensorType.F32, new long[] {2}, new byte[8])
            .addTensor("blk.0.ffn_up_exps.weight", GgufTensorType.F32, new long[] {2}, new byte[8])
            .tensorOffset("blk.0.ffn_up_exps.weight", 4)
            .build();

    assertThatThrownBy(() -> GgufParser.parseSegment(MemorySegment.ofArray(data)))
        .isInstanceOf(MalformedGgufException.class)
        .hasMessageContaining("'blk.0.ffn_gate_exps.weight'")
        .hasMessageContaining("overlaps")
        .hasMessageContaining("'blk.0.ffn_up_exps.weight'")
        .hasMessageContaining("expected 8 bytes")
        .hasMessageContaining("available 4");
  }

  @Test
  void overlapIsDetectedRegardlessOfTensorTableOrder() {
    byte[] data =
        new SyntheticGgufBuilder()
            .addUint32("general.alignment", 4)
            .addTensor("late", GgufTensorType.F32, new long[] {2}, new byte[8])
            .addTensor("early", GgufTensorType.F32, new long[] {3}, new byte[12])
            .tensorOffset("late", 8)
            .tensorOffset("early", 0)
            .build();

    assertThatThrownBy(() -> GgufParser.parseSegment(MemorySegment.ofArray(data)))
        .isInstanceOf(MalformedGgufException.class)
        .hasMessageContaining("'early'")
        .hasMessageContaining("overlaps")
        .hasMessageContaining("'late'")
        .hasMessageContaining("expected 12 bytes")
        .hasMessageContaining("available 8");
  }

  @Test
  void twoTensorsSharingOneOffsetAreRejected() {
    byte[] data =
        new SyntheticGgufBuilder()
            .addTensor("first", GgufTensorType.F32, new long[] {1}, new byte[4])
            .addTensor("second", GgufTensorType.F32, new long[] {1}, new byte[4])
            .tensorOffset("second", 0)
            .build();

    assertThatThrownBy(() -> GgufParser.parseSegment(MemorySegment.ofArray(data)))
        .isInstanceOf(MalformedGgufException.class)
        .hasMessageContaining("overlaps")
        .hasMessageContaining("available 0");
  }

  @Test
  void tensorStartingPastTheEndOfTheFileIsRejected() {
    byte[] data =
        new SyntheticGgufBuilder()
            .addTensor("orphan", GgufTensorType.F32, new long[] {1}, new byte[4])
            .tensorOffset("orphan", 1_000_000)
            .build();

    assertThatThrownBy(() -> GgufParser.parseSegment(MemorySegment.ofArray(data)))
        .isInstanceOf(MalformedGgufException.class)
        .hasMessageContaining("'orphan'")
        .hasMessageContaining("expected 4 bytes")
        .hasMessageContaining("available 0")
        .hasMessageContaining("past the end of the file");
  }

  @Test
  void validAdjacentTensorsStillParseAndSliceToTheirExactLength() {
    byte[] data =
        new SyntheticGgufBuilder()
            .addTensor("a", GgufTensorType.F32, new long[] {2}, new byte[8])
            .addTensor("b", GgufTensorType.Q8_0, new long[] {32}, new byte[34])
            .addTensor("c", GgufTensorType.F16, new long[] {3}, new byte[6])
            .build();

    GgufFile file = GgufParser.parseSegment(MemorySegment.ofArray(data));

    assertThat(file.getTensor("a").dataSegment().byteSize()).isEqualTo(8);
    assertThat(file.getTensor("b").dataSegment().byteSize()).isEqualTo(34);
    assertThat(file.getTensor("c").dataSegment().byteSize()).isEqualTo(6);
  }
}
