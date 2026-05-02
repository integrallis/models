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

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

@Tag("unit")
class GgufValueTypeTest {

  @Nested
  static class FromId {

    @ParameterizedTest
    @CsvSource({
      "0,UINT8",
      "1,INT8",
      "2,UINT16",
      "3,INT16",
      "4,UINT32",
      "5,INT32",
      "6,FLOAT32",
      "7,BOOL",
      "8,STRING",
      "9,ARRAY",
      "10,UINT64",
      "11,INT64",
      "12,FLOAT64"
    })
    void mapsIdToCorrectType(int id, String expectedName) {
      assertThat(GgufValueType.fromId(id).name()).isEqualTo(expectedName);
    }

    @Test
    void throwsForNegativeId() {
      assertThatThrownBy(() -> GgufValueType.fromId(-1))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void throwsForIdBeyondRange() {
      assertThatThrownBy(() -> GgufValueType.fromId(13))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Nested
  static class RoundTrip {

    @Test
    void allEnumValuesRoundTrip() {
      for (GgufValueType type : GgufValueType.values()) {
        assertThat(GgufValueType.fromId(type.id())).isEqualTo(type);
      }
    }
  }
}
