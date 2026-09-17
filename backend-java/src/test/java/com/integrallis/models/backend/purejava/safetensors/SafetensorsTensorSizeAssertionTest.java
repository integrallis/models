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
package com.integrallis.models.backend.purejava.safetensors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Per-tensor size assertions for Safetensors: dtype width times shape must equal the declared data
 * range, and a range that is short, overlaps another tensor, or runs past the data buffer fails
 * with the tensor name, the expected byte count, and the bytes actually available.
 */
@Tag("unit")
class SafetensorsTensorSizeAssertionTest {

  @Test
  void shortDataRangeNamesTheTensorDtypeShapeExpectedAndAvailableBytes() {
    byte[] artifact =
        SyntheticSafetensorsBuilder.file(
            "{\"experts.7.down\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
            new byte[4]);

    assertThatThrownBy(() -> parse(artifact))
        .isInstanceOf(MalformedSafetensorsException.class)
        .hasMessageContaining("experts.7.down")
        .hasMessageContaining("F32")
        .hasMessageContaining("[2]")
        .hasMessageContaining("expected 8 bytes")
        .hasMessageContaining("available 4");
  }

  @Test
  void rangeRunningPastTheDataBufferNamesTheTensor() {
    byte[] artifact =
        SyntheticSafetensorsBuilder.file(
            "{\"experts.9.up\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}",
            new byte[4]);

    assertThatThrownBy(() -> parse(artifact))
        .isInstanceOf(MalformedSafetensorsException.class)
        .hasMessageContaining("experts.9.up")
        .hasMessageContaining("past the end")
        .hasMessageContaining("expected 8 bytes")
        .hasMessageContaining("available 4");
  }

  @Test
  void overlappingRangesNameBothTensors() {
    byte[] artifact =
        SyntheticSafetensorsBuilder.file(
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
                + "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}",
            new byte[12]);

    assertThatThrownBy(() -> parse(artifact))
        .isInstanceOf(MalformedSafetensorsException.class)
        .hasMessageContaining("tensor b")
        .hasMessageContaining("overlaps tensor a")
        .hasMessageContaining("offset");
  }

  @Test
  void validAdjacentTensorsStillParse() {
    byte[] artifact =
        new SyntheticSafetensorsBuilder()
            .addF32("a", new long[] {2}, 1.0f, 2.0f)
            .add("b", "U8", new long[] {3}, 1, 2, 3)
            .build();

    SafetensorsFile file = parse(artifact);

    assertThat(file.tensorNames()).containsExactly("a", "b");
    assertThat(file.tensor("a").data().byteSize()).isEqualTo(8);
  }

  private static SafetensorsFile parse(byte[] artifact) {
    return SafetensorsParser.parseSegment(MemorySegment.ofArray(artifact));
  }
}
