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

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TextToSpeechModelTest {

  @Test
  void convenienceCallUsesPortableDefaults() {
    var model = new StubTextToSpeechModel();

    PcmAudio audio = model.synthesize("hello");

    assertThat(audio.samples()).containsExactly(0.25f);
    assertThat(model.options).isEqualTo(SpeechSynthesisOptions.builder().build());
  }

  @Test
  void defaultStreamingAdapterDeliversAudioAndOneTerminalSignal() {
    var model = new StubTextToSpeechModel();
    List<String> events = new ArrayList<>();

    model.synthesize(
        "hello",
        SpeechSynthesisOptions.builder().build(),
        new AudioStream() {
          @Override
          public void onAudio(PcmAudio audio) {
            events.add("audio:" + audio.frameCount());
          }

          @Override
          public void onComplete() {
            events.add("complete");
          }

          @Override
          public void onError(Throwable failure) {
            events.add("error");
          }
        });

    assertThat(events).containsExactly("audio:1", "complete");
  }

  private static final class StubTextToSpeechModel implements TextToSpeechModel {
    private SpeechSynthesisOptions options;

    @Override
    public String modelName() {
      return "stub";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("stub");
    }

    @Override
    public PcmAudio synthesize(String text, SpeechSynthesisOptions options) {
      this.options = options;
      return new PcmAudio(1, 1, new float[] {0.25f});
    }
  }
}
