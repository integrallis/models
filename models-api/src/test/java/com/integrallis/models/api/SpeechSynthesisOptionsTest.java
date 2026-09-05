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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SpeechSynthesisOptionsTest {

  @Test
  void defaultsLeaveModelSpecificChoicesToTheModel() {
    SpeechSynthesisOptions options = SpeechSynthesisOptions.builder().build();

    assertThat(options.sampling()).isNull();
    assertThat(options.language()).isNull();
    assertThat(options.voice()).isNull();
    assertThat(options.referenceAudio()).isNull();
    assertThat(options.referenceText()).isNull();
    assertThat(options.speed()).isEqualTo(1.0f);
  }

  @Test
  void carriesPortableSpeechControls() {
    PcmAudio reference = new PcmAudio(24_000, 1, new float[] {0.0f, 0.1f});
    SamplingOptions sampling = SamplingOptions.builder().temperature(0.3f).seed(42).build();

    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .sampling(sampling)
            .language("en")
            .voice("alba")
            .speed(1.1f)
            .referenceAudio(reference)
            .referenceText("Reference words")
            .build();

    assertThat(options.sampling()).isSameAs(sampling);
    assertThat(options.language()).isEqualTo("en");
    assertThat(options.voice()).isEqualTo("alba");
    assertThat(options.speed()).isEqualTo(1.1f);
    assertThat(options.referenceAudio()).isSameAs(reference);
    assertThat(options.referenceText()).isEqualTo("Reference words");
  }

  @Test
  void rejectsAmbiguousOrInvalidControls() {
    assertThatThrownBy(() -> SpeechSynthesisOptions.builder().language(" ").build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("language");
    assertThatThrownBy(() -> SpeechSynthesisOptions.builder().speed(0.0f).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("speed");
    assertThatThrownBy(
            () -> SpeechSynthesisOptions.builder().referenceText("orphan transcript").build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("referenceAudio");
  }
}
