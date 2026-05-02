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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class DTypeTest {

  @Nested
  static class ByteSize {

    @Test
    void f32Is4Bytes() {
      assertThat(DType.F32.byteSize()).isEqualTo(4);
    }

    @Test
    void f16Is2Bytes() {
      assertThat(DType.F16.byteSize()).isEqualTo(2);
    }

    @Test
    void bf16Is2Bytes() {
      assertThat(DType.BF16.byteSize()).isEqualTo(2);
    }

    @Test
    void q8_0Is1Byte() {
      assertThat(DType.Q8_0.byteSize()).isEqualTo(1);
    }

    @Test
    void q4_0Is1Byte() {
      assertThat(DType.Q4_0.byteSize()).isEqualTo(1);
    }
  }
}
