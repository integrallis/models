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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufConstantsTest {

  @Nested
  static class Magic {

    @Test
    void magicSpellsGgufInLittleEndian() {
      byte[] bytes = new byte[4];
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(GgufConstants.MAGIC);
      String str = new String(bytes, java.nio.charset.StandardCharsets.US_ASCII);
      assertThat(str).isEqualTo("GGUF");
    }

    @Test
    void magicHasExpectedHexValue() {
      assertThat(GgufConstants.MAGIC).isEqualTo(0x46554747);
    }
  }

  @Nested
  static class Versions {

    @Test
    void supportsVersionTwo() {
      assertThat(GgufConstants.isVersionSupported(2)).isTrue();
    }

    @Test
    void supportsVersionThree() {
      assertThat(GgufConstants.isVersionSupported(3)).isTrue();
    }

    @Test
    void doesNotSupportVersionOne() {
      assertThat(GgufConstants.isVersionSupported(1)).isFalse();
    }

    @Test
    void doesNotSupportVersionZero() {
      assertThat(GgufConstants.isVersionSupported(0)).isFalse();
    }
  }

  @Nested
  static class Alignment {

    @Test
    void defaultAlignmentIsThirtyTwo() {
      assertThat(GgufConstants.DEFAULT_ALIGNMENT).isEqualTo(32);
    }
  }
}
