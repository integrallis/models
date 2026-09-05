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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PcmAudioTest {

  @Test
  void describesInterleavedPcmWithoutExposingMutableStorage() {
    float[] samples = {0.25f, -0.25f, 0.5f, -0.5f};

    PcmAudio audio = new PcmAudio(2, 2, samples);
    samples[0] = 1.0f;
    float[] returned = audio.samples();
    returned[1] = 1.0f;

    assertThat(audio.sampleRate()).isEqualTo(2);
    assertThat(audio.channels()).isEqualTo(2);
    assertThat(audio.frameCount()).isEqualTo(2);
    assertThat(audio.duration()).isEqualTo(Duration.ofSeconds(1));
    assertThat(audio.samples()).containsExactly(0.25f, -0.25f, 0.5f, -0.5f);
  }

  @Test
  void rejectsMalformedPcm() {
    assertThatThrownBy(() -> new PcmAudio(0, 1, new float[] {0.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sampleRate");
    assertThatThrownBy(() -> new PcmAudio(16_000, 0, new float[] {0.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("channels");
    assertThatThrownBy(() -> new PcmAudio(16_000, 2, new float[] {0.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("interleaved");
    assertThatThrownBy(() -> new PcmAudio(16_000, 1, new float[] {Float.NaN}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("finite");
  }
}
