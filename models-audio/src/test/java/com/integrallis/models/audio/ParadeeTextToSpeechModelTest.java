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
package com.integrallis.models.audio;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.AudioStream;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.PcmAudio;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeechSynthesisOptions;
import com.integrallis.models.audio.phonemizer.MisakiEnglishPhonemizer;
import com.integrallis.models.audio.phonemizer.MisakiLexicon;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ParadeeTextToSpeechModelTest {

  private static final String CONFIG =
      "{\"model_type\":\"paradee\",\"version\":\"1.2\",\"sample_rate\":24000,\"n_token\":178,"
          + "\"voices\":[\"af_heart\",\"bf_isabella\",\"am_puck\",\"bm_fable\"],"
          + "\"vocab\":{\" \":16,\".\":4,\"k\":52,\"æ\":58,\"t\":61,\"ˈ\":156,\"ɐ\":100}}";

  private static MisakiEnglishPhonemizer phonemizer(boolean british) {
    Map<String, Object> gold = new HashMap<>();
    gold.put("cat", british ? "kˈat" : "kˈæt");
    gold.put("a", "A");
    return new MisakiEnglishPhonemizer(MisakiLexicon.of(gold, Map.of(), british), "");
  }

  /** A synthesizer that emits one sample per id so calls can be checked arithmetically. */
  private static final class RecordingSynthesizer implements ParadeeTextToSpeechModel.Synthesizer {
    final List<long[]> ids = new ArrayList<>();
    final List<Float> speeds = new ArrayList<>();
    final List<Integer> voices = new ArrayList<>();

    @Override
    public float[] synthesize(
        long[] ids,
        float speed,
        int voiceIndex,
        com.integrallis.models.backend.onnx.NoiseSource noise) {
      this.ids.add(ids);
      speeds.add(speed);
      voices.add(voiceIndex);
      float[] out = new float[ids.length];
      for (int i = 0; i < out.length; i++) {
        out[i] = ids[i] / 200f;
      }
      return out;
    }
  }

  private static ParadeeTextToSpeechModel model(
      RecordingSynthesizer synthesizer, boolean withBritish) {
    return new ParadeeTextToSpeechModel(
        ParadeeTextToSpeechModel.Config.parse(CONFIG),
        synthesizer,
        phonemizer(false),
        withBritish ? phonemizer(true) : null,
        BackendDiagnostics.unavailable("pure-java"));
  }

  @Test
  void parsesTheConfig() {
    ParadeeTextToSpeechModel.Config config = ParadeeTextToSpeechModel.Config.parse(CONFIG);
    assertThat(config.version()).isEqualTo("1.2");
    assertThat(config.sampleRate()).isEqualTo(24000);
    assertThat(config.voices()).containsExactly("af_heart", "bf_isabella", "am_puck", "bm_fable");
    assertThat(config.vocab()).containsEntry("k", 52);
    ParadeeTextToSpeechModel.Config minimal =
        ParadeeTextToSpeechModel.Config.parse("{\"vocab\":{\"a\":1}}");
    assertThat(minimal.voices()).containsExactly("af_heart");
    assertThat(minimal.sampleRate()).isEqualTo(24000);
    assertThat(minimal.version()).isEqualTo("?");
    assertThatThrownBy(() -> ParadeeTextToSpeechModel.Config.parse("{\"voices\":[]}"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void buildsIdsWithPadsAndDropsCharactersOutsideTheVocabulary() {
    ParadeeTextToSpeechModel model = model(new RecordingSynthesizer(), true);
    assertThat(model.ids("kˈæt.")).containsExactly(0, 52, 156, 58, 61, 4, 0);
    assertThat(model.ids("kZæ")).containsExactly(0, 52, 58, 0);
    long[] capped = model.ids("k".repeat(600));
    assertThat(capped).hasSize(ParadeeTextToSpeechModel.MAX_PHONEMES + 2);
    assertThat(model.modelName()).isEqualTo("Paradee-8M v1.2");
    assertThat(model.config().voices()).hasSize(4);
    assertThat(model.diagnostics().backend()).isEqualTo("pure-java");
  }

  @Test
  void splitsSentencesPhonemizesPerVoiceAndCutsLongOnesAtASpace() {
    ParadeeTextToSpeechModel model = model(new RecordingSynthesizer(), true);
    assertThat(model.phonemeChunks("A cat. A cat!\nA cat", "af_heart"))
        .containsExactly("ɐ kˈæt.", "ɐ kˈæt!", "ɐ kˈæt");
    assertThat(model.phonemeChunks("a cat", "bm_fable")).containsExactly("ɐ kˈat");
    StringBuilder longText = new StringBuilder();
    for (int i = 0; i < 120; i++) {
      longText.append("cat ");
    }
    List<String> chunks = model.phonemeChunks(longText.toString(), "af_heart");
    assertThat(chunks).hasSizeGreaterThan(1);
    assertThat(chunks.get(0).length()).isLessThanOrEqualTo(ParadeeTextToSpeechModel.MAX_PHONEMES);
    assertThat(chunks.get(0)).endsWith("kˈæt");
    assertThat(model.phonemeChunks("   ", "af_heart")).isEmpty();
    assertThatThrownBy(() -> model(new RecordingSynthesizer(), false).phonemizerFor("bf_isabella"))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void synthesizesPerSentenceWithVoiceSpeedAndSeed() {
    RecordingSynthesizer synthesizer = new RecordingSynthesizer();
    ParadeeTextToSpeechModel model = model(synthesizer, true);
    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .voice("am_puck")
            .speed(1.25f)
            .sampling(SamplingOptions.builder().seed(7).build())
            .build();
    PcmAudio audio = model.synthesize("A cat. A cat.", options);
    assertThat(audio.sampleRate()).isEqualTo(24000);
    assertThat(audio.channels()).isEqualTo(1);
    assertThat(synthesizer.ids).hasSize(2);
    assertThat(synthesizer.ids.get(0)).containsExactly(0, 100, 16, 52, 156, 58, 61, 4, 0);
    assertThat(synthesizer.voices).containsExactly(2, 2);
    assertThat(synthesizer.speeds).containsExactly(1.25f, 1.25f);
    assertThat(audio.frameCount()).isEqualTo(18);

    PcmAudio defaults = model.synthesize("A cat.");
    assertThat(synthesizer.voices.get(2)).isZero();
    assertThat(defaults.frameCount()).isEqualTo(9);

    PcmAudio direct = model.synthesizePhonemes("kˈæt", "bf_isabella", 1f, 3L);
    assertThat(direct.frameCount()).isEqualTo(6);
    assertThat(synthesizer.voices.get(3)).isEqualTo(1);
    assertThatThrownBy(() -> model.synthesizePhonemes("k", "zz", 1f, null))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("no voice");
    assertThatThrownBy(() -> model.synthesizePhonemes("k", "af_heart", 0f, null))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                model.synthesize("A cat.", SpeechSynthesisOptions.builder().voice("nope").build()))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void streamsOneChunkPerSentenceAndReportsFailures() {
    RecordingSynthesizer synthesizer = new RecordingSynthesizer();
    ParadeeTextToSpeechModel model = model(synthesizer, true);
    List<PcmAudio> chunks = new ArrayList<>();
    List<Throwable> failures = new ArrayList<>();
    int[] completed = {0};
    AudioStream stream =
        new AudioStream() {
          @Override
          public void onAudio(PcmAudio audio) {
            chunks.add(audio);
          }

          @Override
          public void onComplete() {
            completed[0]++;
          }

          @Override
          public void onError(Throwable failure) {
            failures.add(failure);
          }
        };
    model.synthesize("A cat. A cat.", SpeechSynthesisOptions.builder().build(), stream);
    assertThat(chunks).hasSize(2);
    assertThat(completed[0]).isEqualTo(1);
    assertThat(failures).isEmpty();

    model.synthesize("A cat.", SpeechSynthesisOptions.builder().voice("nope").build(), stream);
    assertThat(failures).hasSize(1);
    assertThat(completed[0]).isEqualTo(1);

    model.close();
    assertThatThrownBy(() -> model.synthesize("A cat.")).isInstanceOf(IllegalStateException.class);
    model.synthesize("A cat.", SpeechSynthesisOptions.builder().build(), stream);
    assertThat(failures).hasSize(2);
  }
}
