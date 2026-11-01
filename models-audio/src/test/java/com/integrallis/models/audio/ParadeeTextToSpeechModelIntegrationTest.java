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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.PcmAudio;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeechSynthesisOptions;
import com.integrallis.models.api.WavEncoder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * End-to-end Paradee synthesis from text with the pinned v1.2 int8 graph and misaki's dictionaries.
 * Needs {@code MODELS_PARADEE_ONNX_DIR} (graphs and config.json) and {@code
 * MODELS_MISAKI_DATA_DIR}; writes one WAV per voice into {@code MODELS_PARADEE_OUTPUT_DIR} when
 * set.
 */
@Tag("integration")
class ParadeeTextToSpeechModelIntegrationTest {

  private static final String SENTENCE =
      "The library opens at nine, but the coffee shop next door is usually busy by eight.";

  @Test
  void speaksEveryVoiceFromText() throws Exception {
    String onnxDir = System.getenv("MODELS_PARADEE_ONNX_DIR");
    String lexiconDir = System.getenv("MODELS_MISAKI_DATA_DIR");
    assumeTrue(
        onnxDir != null && lexiconDir != null,
        "MODELS_PARADEE_ONNX_DIR / MODELS_MISAKI_DATA_DIR not set");
    Path graph = Path.of(onnxDir, "onnx_paradee_int8.onnx");
    Path config = Path.of(onnxDir, "config.json");
    assumeTrue(Files.isRegularFile(graph) && Files.isRegularFile(config));
    String outputDir = System.getenv("MODELS_PARADEE_OUTPUT_DIR");

    long loadStart = System.nanoTime();
    try (ParadeeTextToSpeechModel model =
        ParadeeTextToSpeechModel.load(graph, config, Path.of(lexiconDir))) {
      long loadMillis = (System.nanoTime() - loadStart) / 1_000_000L;
      System.out.printf(
          "PARADEE_TTS loaded %s in %d ms; %s%n",
          model.modelName(), loadMillis, model.diagnostics().environment());
      for (String voice : model.config().voices()) {
        SpeechSynthesisOptions options =
            SpeechSynthesisOptions.builder()
                .voice(voice)
                .sampling(SamplingOptions.builder().seed(42).build())
                .build();
        System.out.printf(
            "PARADEE_TTS %s phonemes: %s%n", voice, model.phonemeChunks(SENTENCE, voice));
        model.synthesize(SENTENCE, options);
        long start = System.nanoTime();
        PcmAudio audio = model.synthesize(SENTENCE, options);
        long millis = (System.nanoTime() - start) / 1_000_000L;
        double seconds = audio.duration().toNanos() / 1e9;
        assertThat(audio.sampleRate()).isEqualTo(24_000);
        assertThat(seconds).isBetween(2.0, 10.0);
        float peak = 0f;
        for (float s : audio.samples()) {
          peak = Math.max(peak, Math.abs(s));
        }
        assertThat(peak).isGreaterThan(0.05f);
        System.out.printf(
            "PARADEE_TTS %s %.2fs audio in %d ms (%.2fx realtime) peak %.3f%n",
            voice, seconds, millis, seconds * 1000.0 / Math.max(1, millis), peak);
        if (outputDir != null) {
          Path out = Path.of(outputDir, "paradee-" + voice + ".wav");
          Files.createDirectories(out.getParent());
          Files.write(out, WavEncoder.pcm16(audio));
          System.out.printf("PARADEE_TTS wrote %s%n", out);
        }
      }
    }
  }
}
