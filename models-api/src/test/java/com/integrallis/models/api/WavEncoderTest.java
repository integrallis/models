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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class WavEncoderTest {

  @Test
  void encodesPortablePcm16WaveDataAndClipsSamples() {
    byte[] wave =
        WavEncoder.pcm16(new PcmAudio(16_000, 1, new float[] {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}));
    ByteBuffer data = ByteBuffer.wrap(wave).order(ByteOrder.LITTLE_ENDIAN);

    assertThat(new String(wave, 0, 4, StandardCharsets.US_ASCII)).isEqualTo("RIFF");
    assertThat(new String(wave, 8, 4, StandardCharsets.US_ASCII)).isEqualTo("WAVE");
    assertThat(data.getShort(20)).isEqualTo((short) 1);
    assertThat(data.getShort(22)).isEqualTo((short) 1);
    assertThat(data.getInt(24)).isEqualTo(16_000);
    assertThat(data.getShort(34)).isEqualTo((short) 16);
    assertThat(data.getInt(40)).isEqualTo(10);
    assertThat(data.getShort(44)).isEqualTo(Short.MIN_VALUE);
    assertThat(data.getShort(46)).isEqualTo(Short.MIN_VALUE);
    assertThat(data.getShort(48)).isZero();
    assertThat(data.getShort(50)).isEqualTo(Short.MAX_VALUE);
    assertThat(data.getShort(52)).isEqualTo(Short.MAX_VALUE);
  }
}
