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
import com.integrallis.models.api.SpeechSynthesisOptions;
import com.integrallis.models.api.WavEncoder;
import com.integrallis.models.audio.phonemizer.JsonAccess;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Judge sanity for the Paradee qualification: a corpus spoken by the shipped, oracle-verified
 * Soprano model, in the same {@code NNN.wav} + {@code NNN.txt} + {@code pass.json} layout the two
 * Paradee arms use, so that {@code judge.py} is seen scoring a model whose waveform parity is
 * already proven before its numbers are read for the candidate.
 *
 * <p>The product route: {@link SopranoTextToSpeechModel#load(Path)} with the model's own default
 * sampling (temperature 0.3, top-p 0.95, repetition penalty 1.2, unseeded). Timing is recorded but
 * is not a measurement of anything gated here.
 *
 * <p>Environment: {@code MODELS_SOPRANO_GGUF}, {@code MODELS_PARADEE_CORPUS}, {@code
 * MODELS_SOPRANO_ARM_OUT}; optional {@code MODELS_SOPRANO_LIMIT} to speak only the first N
 * sentences.
 */
@Tag("integration")
class SopranoJudgeSanityIntegrationTest {

  @Test
  void speaksACorpusWithTheShippedSopranoModel() throws Exception {
    String gguf = System.getenv("MODELS_SOPRANO_GGUF");
    String corpusPath = System.getenv("MODELS_PARADEE_CORPUS");
    String outDir = System.getenv("MODELS_SOPRANO_ARM_OUT");
    assumeTrue(
        gguf != null && corpusPath != null && outDir != null && Files.isRegularFile(Path.of(gguf)),
        "judge sanity environment not set");
    int limit = Integer.parseInt(System.getenv().getOrDefault("MODELS_SOPRANO_LIMIT", "0"));

    Map<String, Object> corpus =
        JsonAccess.object(Files.readString(Path.of(corpusPath), StandardCharsets.UTF_8));
    @SuppressWarnings("unchecked")
    List<String> all = (List<String>) corpus.get("sentences");
    List<String> sentences = limit > 0 && limit < all.size() ? all.subList(0, limit) : all;
    Path out = Path.of(outDir);
    Files.createDirectories(out);
    SpeechSynthesisOptions options = SpeechSynthesisOptions.builder().language("en").build();

    try (SopranoTextToSpeechModel model = SopranoTextToSpeechModel.load(Path.of(gguf))) {
      model.synthesize(sentences.get(0), options); // warm-up, untimed, as the arms do
      List<Map<String, Object>> rows = new ArrayList<>();
      double totalAudio = 0;
      double totalRun = 0;
      boolean allFinite = true;
      for (int i = 0; i < sentences.size(); i++) {
        String text = sentences.get(i);
        long start = System.nanoTime();
        PcmAudio audio = model.synthesize(text, options);
        double runSeconds = (System.nanoTime() - start) / 1e9;
        float[] samples = audio.samples();
        boolean finite = true;
        for (float s : samples) {
          finite &= Float.isFinite(s);
        }
        allFinite &= finite;
        Files.write(out.resolve(String.format("%03d.wav", i)), WavEncoder.pcm16(audio));
        Files.writeString(
            out.resolve(String.format("%03d.txt", i)), text + "\n", StandardCharsets.UTF_8);
        double audioSeconds = samples.length / (double) audio.sampleRate();
        totalAudio += audioSeconds;
        totalRun += runSeconds;
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("index", i);
        row.put("text", text);
        row.put("audioSeconds", audioSeconds);
        row.put("runSeconds", runSeconds);
        row.put("realTimeFactor", audioSeconds > 0 ? runSeconds / audioSeconds : null);
        row.put("finite", finite);
        row.put("samples", samples.length);
        rows.add(row);
        System.out.printf(
            "SOPRANO_SANITY %03d %5.2fs audio %7.1f ms run%n", i, audioSeconds, runSeconds * 1000);
      }
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("arm", "judge-sanity-soprano");
      report.put(
          "runtime",
          "models-audio SopranoTextToSpeechModel.load (pure Java route) "
              + model.diagnostics().environment()
              + " via "
              + model.modelName());
      report.put("model", Path.of(gguf).getFileName().toString());
      report.put("modelSha256", sha256(Path.of(gguf)));
      report.put(
          "sampling",
          "model defaults: temperature 0.3, topP 0.95, repetitionPenalty 1.2, unseeded");
      report.put("sampleRate", model == null ? null : rows.isEmpty() ? null : sampleRateOf(out));
      report.put("corpus", corpus.get("name"));
      report.put("corpusSource", corpus.get("source"));
      report.put("limit", limit);
      report.put("java", System.getProperty("java.version"));
      report.put(
          "machine",
          System.getProperty("os.name")
              + " "
              + System.getProperty("os.version")
              + " "
              + System.getProperty("os.arch"));
      report.put("sentences", rows.size());
      report.put("totalAudioSeconds", totalAudio);
      report.put("totalRunSeconds", totalRun);
      report.put("realTimeMultiple", totalRun > 0 ? totalAudio / totalRun : null);
      report.put("allFinite", allFinite);
      report.put("rows", rows);
      Files.writeString(
          out.resolve("pass.json"),
          ParadeeCandidateArmIntegrationTest.toJson(report),
          StandardCharsets.UTF_8);
      System.out.printf(
          "SOPRANO_SANITY %s: %d sentences, %.1fs audio in %.2fs -> %.2fx realtime%n",
          corpus.get("name"), rows.size(), totalAudio, totalRun, totalAudio / totalRun);
      assertThat(allFinite).isTrue();
    }
  }

  private static Integer sampleRateOf(Path out) throws Exception {
    byte[] header = Files.readAllBytes(out.resolve("000.wav"));
    // RIFF little-endian sample rate at bytes 24..27 of a canonical PCM header.
    return (header[24] & 0xff)
        | (header[25] & 0xff) << 8
        | (header[26] & 0xff) << 16
        | (header[27] & 0xff) << 24;
  }

  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var in = Files.newInputStream(path)) {
      byte[] buffer = new byte[1 << 20];
      int read;
      while ((read = in.read(buffer)) > 0) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
