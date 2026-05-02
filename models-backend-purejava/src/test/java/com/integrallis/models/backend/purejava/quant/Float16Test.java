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

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Float16Test {

  @Nested
  static class ToFloat {

    @Test
    void zeroIsZero() {
      assertThat(Float16.toFloat((short) 0x0000)).isEqualTo(0.0f);
    }

    @Test
    void onePointZero() {
      // f16: 0x3C00 = 1.0
      assertThat(Float16.toFloat((short) 0x3C00)).isEqualTo(1.0f);
    }

    @Test
    void negativeOnePointZero() {
      // f16: 0xBC00 = -1.0
      assertThat(Float16.toFloat((short) 0xBC00)).isEqualTo(-1.0f);
    }

    @Test
    void twoPointZero() {
      // f16: 0x4000 = 2.0
      assertThat(Float16.toFloat((short) 0x4000)).isEqualTo(2.0f);
    }

    @Test
    void oneHalf() {
      // f16: 0x3800 = 0.5
      assertThat(Float16.toFloat((short) 0x3800)).isEqualTo(0.5f);
    }

    @Test
    void positiveInfinity() {
      assertThat(Float16.toFloat((short) 0x7C00)).isEqualTo(Float.POSITIVE_INFINITY);
    }

    @Test
    void negativeInfinity() {
      assertThat(Float16.toFloat((short) 0xFC00)).isEqualTo(Float.NEGATIVE_INFINITY);
    }

    @Test
    void nanBits() {
      assertThat(Float16.toFloat((short) 0x7E00)).isNaN();
    }
  }

  @Nested
  static class RoundTrip {

    @Test
    void roundTripsCommonValues() {
      float[] values = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -0.25f, 100.0f};
      for (float val : values) {
        short bits = Float16.fromFloat(val);
        float recovered = Float16.toFloat(bits);
        assertThat(recovered).isCloseTo(val, within(Math.abs(val) * 0.001f + 0.0001f));
      }
    }

    @Test
    void roundTripsInfinity() {
      assertThat(Float16.toFloat(Float16.fromFloat(Float.POSITIVE_INFINITY)))
          .isEqualTo(Float.POSITIVE_INFINITY);
      assertThat(Float16.toFloat(Float16.fromFloat(Float.NEGATIVE_INFINITY)))
          .isEqualTo(Float.NEGATIVE_INFINITY);
    }
  }
}
