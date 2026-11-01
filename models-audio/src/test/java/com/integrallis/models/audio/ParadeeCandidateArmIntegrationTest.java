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
import com.integrallis.models.api.WavEncoder;
import com.integrallis.models.audio.phonemizer.JsonAccess;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * One pass of the candidate arm, in the same evidence layout as {@code comparator_arm.py}: {@code
 * NNN.wav}, {@code NNN.txt} and {@code pass.json} per sentence under an output directory.
 *
 * <p>Environment: {@code MODELS_PARADEE_ONNX_DIR}, {@code MODELS_MISAKI_DATA_DIR}, {@code
 * MODELS_PARADEE_CORPUS} (a corpus JSON), {@code MODELS_PARADEE_VOICE}, {@code
 * MODELS_PARADEE_PASS}, {@code MODELS_PARADEE_ARM_OUT}; optional {@code
 * MODELS_PARADEE_PHONEMES_FROM}, a comparator {@code pass.json} whose per-sentence phonemes are fed
 * instead of our own, which isolates the executor from the phonemizer. Timing follows the
 * publisher's convention: one untimed warm-up on the first sentence, then the graph time per
 * sentence; the phonemizer time is recorded separately.
 */
@Tag("integration")
class ParadeeCandidateArmIntegrationTest {

  @Test
  void runsOnePassOfTheCandidateArm() throws Exception {
    String onnxDir = System.getenv("MODELS_PARADEE_ONNX_DIR");
    String lexiconDir = System.getenv("MODELS_MISAKI_DATA_DIR");
    String corpusPath = System.getenv("MODELS_PARADEE_CORPUS");
    String voice = System.getenv("MODELS_PARADEE_VOICE");
    String outDir = System.getenv("MODELS_PARADEE_ARM_OUT");
    assumeTrue(
        onnxDir != null
            && lexiconDir != null
            && corpusPath != null
            && voice != null
            && outDir != null,
        "candidate arm environment not set");
    int pass = Integer.parseInt(System.getenv().getOrDefault("MODELS_PARADEE_PASS", "1"));
    String phonemesFrom = System.getenv("MODELS_PARADEE_PHONEMES_FROM");
    Path graph = Path.of(onnxDir, "onnx_paradee_int8.onnx");
    Path config = Path.of(onnxDir, "config.json");
    assumeTrue(Files.isRegularFile(graph) && Files.isRegularFile(config));

    Map<String, Object> corpus =
        JsonAccess.object(Files.readString(Path.of(corpusPath), StandardCharsets.UTF_8));
    @SuppressWarnings("unchecked")
    List<String> sentences = (List<String>) corpus.get("sentences");
    List<List<String>> publisherPhonemes = null;
    if (phonemesFrom != null) {
      Map<String, Object> theirs =
          JsonAccess.object(Files.readString(Path.of(phonemesFrom), StandardCharsets.UTF_8));
      publisherPhonemes = new ArrayList<>();
      @SuppressWarnings("unchecked")
      List<Map<String, Object>> rows = (List<Map<String, Object>>) theirs.get("rows");
      for (Map<String, Object> row : rows) {
        @SuppressWarnings("unchecked")
        List<String> chunks = (List<String>) row.get("phonemes");
        publisherPhonemes.add(chunks);
      }
      assertThat(publisherPhonemes).hasSize(sentences.size());
    }

    Path out = Path.of(outDir);
    Files.createDirectories(out);
    long seed = 20261101L + pass;
    try (ParadeeTextToSpeechModel model =
        ParadeeTextToSpeechModel.load(graph, config, Path.of(lexiconDir))) {
      // Warm-up: the publisher's protocol runs the first sentence once untimed.
      for (String chunk : model.phonemeChunks(sentences.get(0), voice)) {
        model.synthesizePhonemes(chunk, voice, 1f, seed);
      }
      List<Map<String, Object>> rows = new ArrayList<>();
      double totalAudio = 0;
      double totalRun = 0;
      boolean allFinite = true;
      for (int i = 0; i < sentences.size(); i++) {
        String text = sentences.get(i);
        long g2pStart = System.nanoTime();
        List<String> chunks =
            publisherPhonemes == null ? model.phonemeChunks(text, voice) : publisherPhonemes.get(i);
        double g2pSeconds = publisherPhonemes == null ? (System.nanoTime() - g2pStart) / 1e9 : 0.0;
        List<float[]> parts = new ArrayList<>();
        double runSeconds = 0;
        Double firstChunk = null;
        List<Integer> idsLengths = new ArrayList<>();
        for (String chunk : chunks) {
          idsLengths.add(model.ids(chunk).length);
          long start = System.nanoTime();
          PcmAudio audio = model.synthesizePhonemes(chunk, voice, 1f, seed);
          double dt = (System.nanoTime() - start) / 1e9;
          runSeconds += dt;
          if (firstChunk == null) {
            firstChunk = dt;
          }
          parts.add(audio.samples());
        }
        int total = 0;
        for (float[] part : parts) {
          total += part.length;
        }
        float[] merged = new float[total];
        int offset = 0;
        for (float[] part : parts) {
          System.arraycopy(part, 0, merged, offset, part.length);
          offset += part.length;
        }
        boolean finite = true;
        for (float s : merged) {
          finite &= Float.isFinite(s);
        }
        allFinite &= finite;
        PcmAudio audio = new PcmAudio(ParadeeTextToSpeechModel.SAMPLE_RATE, 1, merged);
        Files.write(out.resolve(String.format("%03d.wav", i)), WavEncoder.pcm16(audio));
        Files.writeString(
            out.resolve(String.format("%03d.txt", i)), text + "\n", StandardCharsets.UTF_8);
        double audioSeconds = merged.length / (double) ParadeeTextToSpeechModel.SAMPLE_RATE;
        totalAudio += audioSeconds;
        totalRun += runSeconds;
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("index", i);
        row.put("text", text);
        row.put("phonemes", chunks);
        row.put("idsLengths", idsLengths);
        row.put("audioSeconds", audioSeconds);
        row.put("runSeconds", runSeconds);
        row.put("g2pSeconds", g2pSeconds);
        row.put("timeToFirstAudioSeconds", g2pSeconds + (firstChunk == null ? 0.0 : firstChunk));
        row.put("realTimeFactor", audioSeconds > 0 ? runSeconds / audioSeconds : null);
        row.put("finite", finite);
        row.put("samples", merged.length);
        rows.add(row);
        System.out.printf(
            "PARADEE_ARM %03d %5.2fs audio %7.1f ms run%n", i, audioSeconds, runSeconds * 1000);
      }
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("arm", "candidate");
      report.put(
          "runtime", "models " + model.diagnostics().environment() + " via " + model.modelName());
      report.put("onnxruntimeVersion", null);
      report.put("intraOpThreads", 1);
      report.put("interOpThreads", 1);
      report.put("model", "onnx/paradee_int8.onnx");
      report.put(
          "phonemeSource",
          publisherPhonemes == null ? "java misaki port" : "publisher pass.json: " + phonemesFrom);
      report.put("voice", voice);
      report.put("voiceIndex", model.voiceIndex(voice));
      report.put("corpus", corpus.get("name"));
      report.put("corpusSource", corpus.get("source"));
      report.put("pass", pass);
      report.put("noiseSeed", seed);
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
      Files.writeString(out.resolve("pass.json"), toJson(report), StandardCharsets.UTF_8);
      System.out.printf(
          "PARADEE_ARM %s %s pass%d: %.1fs audio in %.2fs -> %.2fx realtime%n",
          corpus.get("name"), voice, pass, totalAudio, totalRun, totalAudio / totalRun);
      assertThat(allFinite).isTrue();
    }
  }

  /** A small JSON writer for maps, lists, strings, numbers, booleans and null. */
  static String toJson(Object value) {
    StringBuilder out = new StringBuilder();
    write(out, value, 0);
    out.append('\n');
    return out.toString();
  }

  private static void write(StringBuilder out, Object value, int depth) {
    if (value == null) {
      out.append("null");
    } else if (value instanceof String s) {
      out.append('"');
      for (char c : s.toCharArray()) {
        switch (c) {
          case '"' -> out.append("\\\"");
          case '\\' -> out.append("\\\\");
          case '\n' -> out.append("\\n");
          case '\r' -> out.append("\\r");
          case '\t' -> out.append("\\t");
          default -> {
            if (c < 0x20) {
              out.append(String.format("\\u%04x", (int) c));
            } else {
              out.append(c);
            }
          }
        }
      }
      out.append('"');
    } else if (value instanceof Map<?, ?> map) {
      out.append("{\n");
      int i = 0;
      for (Map.Entry<?, ?> e : map.entrySet()) {
        out.append(" ".repeat(depth + 1));
        write(out, String.valueOf(e.getKey()), depth + 1);
        out.append(": ");
        write(out, e.getValue(), depth + 1);
        out.append(++i < map.size() ? ",\n" : "\n");
      }
      out.append(" ".repeat(depth)).append('}');
    } else if (value instanceof List<?> list) {
      out.append('[');
      for (int i = 0; i < list.size(); i++) {
        if (i > 0) {
          out.append(", ");
        }
        write(out, list.get(i), depth + 1);
      }
      out.append(']');
    } else if (value instanceof Double d) {
      out.append(d.isNaN() || d.isInfinite() ? "null" : d.toString());
    } else {
      out.append(value);
    }
  }
}
