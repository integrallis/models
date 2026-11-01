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
package com.integrallis.models.backend.onnx;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Executes the pinned Paradee-8M v1.2 graphs end to end on a hand-written phoneme string. This is a
 * smoke test of operator coverage and shape plumbing, not of correctness: correctness is the
 * seeded-parity test against the publisher's runtime. Needs {@code MODELS_PARADEE_ONNX_DIR}.
 */
@Tag("integration")
class ParadeeGraphSmokeIntegrationTest {

  // misaki-style IPA for "Paradee is a small voice that runs anywhere", hand-written; only the
  // characters that are in the vocabulary are fed, exactly as the publisher's package does.
  private static final String PHONEMES = "pˌɛɹədˈi ɪz ɐ smˈɔl vˈɔɪs ðæt ɹˈʌnz ˈɛniwˌɛɹ.";

  @Test
  void runsTheFp32GraphToAFiniteWaveform() throws Exception {
    runGraph("onnx_paradee.onnx");
  }

  @Test
  void runsTheInt8GraphToAFiniteWaveform() throws Exception {
    runGraph("onnx_paradee_int8.onnx");
  }

  private static void runGraph(String file) throws Exception {
    String configured = System.getenv("MODELS_PARADEE_ONNX_DIR");
    assumeTrue(configured != null, "MODELS_PARADEE_ONNX_DIR not set");
    Path dir = Path.of(configured);
    assumeTrue(
        Files.isRegularFile(dir.resolve(file)) && Files.isRegularFile(dir.resolve("config.json")));
    Map<String, Integer> vocab =
        vocabulary(Files.readString(dir.resolve("config.json"), StandardCharsets.UTF_8));
    List<Long> ids = new ArrayList<>();
    ids.add(0L);
    PHONEMES
        .codePoints()
        .forEach(
            cp -> {
              Integer id = vocab.get(new String(Character.toChars(cp)));
              if (id != null) {
                ids.add((long) id);
              }
            });
    ids.add(0L);
    long[] idArray = ids.stream().mapToLong(Long::longValue).toArray();

    long loadStart = System.nanoTime();
    OnnxModel model = OnnxModel.load(dir.resolve(file));
    OnnxGraphExecutor executor = new OnnxGraphExecutor(model, NoiseSource.seeded(42));
    long loadMillis = (System.nanoTime() - loadStart) / 1_000_000L;
    Map<String, Tensor> feeds =
        Map.of(
            "input_ids", Tensor.ofLongs(new long[] {1, idArray.length}, idArray),
            "speed", Tensor.ofFloats(new long[] {1}, new float[] {1f}),
            "voice", Tensor.ofLongs(new long[] {1}, new long[] {0}));

    Map<String, String> producers = new HashMap<>();
    for (OnnxNode node : model.graph().nodes()) {
      for (String output : node.outputs()) {
        producers.put(output, node.opType());
      }
    }
    Map<String, Long> nanosByOp = new TreeMap<>();
    long[] last = {System.nanoTime()};
    Map<String, Tensor> out =
        executor.run(
            feeds,
            List.of("waveform"),
            (name, tensor) -> {
              long now = System.nanoTime();
              nanosByOp.merge(producers.getOrDefault(name, "?"), now - last[0], Long::sum);
              last[0] = now;
            });
    long start = System.nanoTime();
    Tensor waveform = executor.run(feeds).get("waveform");
    long runMillis = (System.nanoTime() - start) / 1_000_000L;

    assertThat(waveform.rank()).isEqualTo(2);
    assertThat(waveform.dim(0)).isEqualTo(1);
    float[] samples = waveform.floats();
    assertThat(samples.length).isGreaterThan(24_000);
    double peak = 0;
    for (float s : samples) {
      assertThat(Float.isFinite(s)).isTrue();
      peak = Math.max(peak, Math.abs(s));
    }
    assertThat(peak).isGreaterThan(1e-3);
    double seconds = samples.length / 24000.0;
    System.out.printf(
        "PARADEE_SMOKE %s ids=%d dynamicNodes=%d load=%dms run=%dms audio=%.2fs realtime=%.2fx peak=%.3f%n",
        file,
        idArray.length,
        executor.dynamicNodeCount(),
        loadMillis,
        runMillis,
        seconds,
        seconds * 1000.0 / Math.max(1, runMillis),
        peak);
    nanosByOp.entrySet().stream()
        .sorted((a, b) -> Long.compare(b.getValue(), a.getValue()))
        .limit(12)
        .forEach(
            e ->
                System.out.printf(
                    "PARADEE_SMOKE %s   %-22s %6d ms%n",
                    file, e.getKey(), e.getValue() / 1_000_000L));
    assertThat(out.get("waveform").size()).isEqualTo(samples.length);
  }

  private static Map<String, Integer> vocabulary(String configJson) {
    // The config is flat JSON; the vocab object maps single characters to integer ids.
    Map<String, Integer> vocab = new HashMap<>();
    int start = configJson.indexOf("\"vocab\"");
    int open = configJson.indexOf('{', start);
    int close = configJson.indexOf('}', open);
    Matcher matcher =
        Pattern.compile("\"((?:\\\\.|[^\"\\\\])+)\"\\s*:\\s*(\\d+)")
            .matcher(configJson.substring(open, close));
    while (matcher.find()) {
      vocab.put(unescape(matcher.group(1)), Integer.parseInt(matcher.group(2)));
    }
    return vocab;
  }

  private static String unescape(String text) {
    StringBuilder out = new StringBuilder();
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      if (c == '\\' && i + 1 < text.length()) {
        char next = text.charAt(++i);
        if (next == 'u' && i + 4 < text.length()) {
          out.append((char) Integer.parseInt(text.substring(i + 1, i + 5), 16));
          i += 4;
        } else {
          out.append(next);
        }
      } else {
        out.append(c);
      }
    }
    return out.toString();
  }
}
