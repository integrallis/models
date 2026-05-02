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

@Tag("unit")
class GgufTensorTypeTest {

  @Nested
  static class Properties {

    @Test
    void f32HasCorrectProperties() {
      assertThat(GgufTensorType.F32.id()).isEqualTo(0);
      assertThat(GgufTensorType.F32.blockSize()).isEqualTo(1);
      assertThat(GgufTensorType.F32.typeSize()).isEqualTo(4);
    }

    @Test
    void f16HasCorrectProperties() {
      assertThat(GgufTensorType.F16.id()).isEqualTo(1);
      assertThat(GgufTensorType.F16.blockSize()).isEqualTo(1);
      assertThat(GgufTensorType.F16.typeSize()).isEqualTo(2);
    }

    @Test
    void q4_0HasCorrectProperties() {
      assertThat(GgufTensorType.Q4_0.id()).isEqualTo(2);
      assertThat(GgufTensorType.Q4_0.blockSize()).isEqualTo(32);
      assertThat(GgufTensorType.Q4_0.typeSize()).isEqualTo(18);
    }

    @Test
    void q8_0HasCorrectProperties() {
      assertThat(GgufTensorType.Q8_0.id()).isEqualTo(8);
      assertThat(GgufTensorType.Q8_0.blockSize()).isEqualTo(32);
      assertThat(GgufTensorType.Q8_0.typeSize()).isEqualTo(34);
    }

    @Test
    void q4_kMHasCorrectProperties() {
      assertThat(GgufTensorType.Q4_K_M.id()).isEqualTo(15);
      assertThat(GgufTensorType.Q4_K_M.blockSize()).isEqualTo(256);
      assertThat(GgufTensorType.Q4_K_M.typeSize()).isEqualTo(144);
    }
  }

  @Nested
  static class FromId {

    @Test
    void roundTripsAllTypes() {
      for (GgufTensorType type : GgufTensorType.values()) {
        assertThat(GgufTensorType.fromId(type.id())).isEqualTo(type);
      }
    }

    @Test
    void throwsForUnknownId() {
      assertThatThrownBy(() -> GgufTensorType.fromId(99))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }
}
