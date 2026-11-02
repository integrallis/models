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
package com.integrallis.models.audio.phonemizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The CTC collapse, and the bundled models against the trainer's own held-out predictions. */
class LetterToPhonemeFallbackTest {

  @Test
  void collapsesRepeatsAndBlanksLikeCtc() {
    String[] symbols = {"a", "b", "ˈ"};
    assertThat(LetterToPhonemeFallback.collapse(new int[] {0, 3, 3, 1, 0, 1, 2, 2, 0}, symbols))
        .isEqualTo("ˈaab");
    assertThat(LetterToPhonemeFallback.collapse(new int[] {0, 0, 0}, symbols)).isEmpty();
    assertThat(LetterToPhonemeFallback.collapse(new int[] {1, 1, 1}, symbols)).isEqualTo("a");
  }

  @Test
  void bundledModelsReproduceTheTrainersHeldOutPredictions() throws Exception {
    // g2p-<accent>-dev-sample.json holds 200 held-out words with the PyTorch model's greedy
    // output; the Java executor must produce the same string for each, which proves the export
    // and the executor, not the model's accuracy (that is on the card).
    for (boolean british : new boolean[] {false, true}) {
      String accent = british ? "gb" : "us";
      List<Map<String, Object>> sample = sample(accent);
      LetterToPhonemeFallback model = LetterToPhonemeFallback.bundled(british);
      assertThat(model.british()).isEqualTo(british);
      List<String> mismatches = new ArrayList<>();
      for (Map<String, Object> row : sample) {
        String ours = model.phonemize((String) row.get("word"));
        if (!String.valueOf(row.get("model")).equals(String.valueOf(ours))) {
          mismatches.add(row.get("word") + ": java=" + ours + " torch=" + row.get("model"));
        }
      }
      // Exact agreement is expected; a near-tie in the argmax can flip on float rounding, so at
      // most two of 200 words may differ, and the message names them.
      assertThat(mismatches)
          .as(accent + " parity with the exporting model: " + mismatches)
          .hasSizeLessThanOrEqualTo(2);
      assertThat(model.phonemize("")).isNull();
      assertThat(model.phonemize("123")).isNull();
      assertThat(model.card()).containsKeys("devWordAccuracy", "devPhonemeErrorRate", "source");
    }
  }

  @Test
  void americanDictionaryTierAnswersBeforeTheModelAndInflectsPossessivesAndPlurals()
      throws Exception {
    LetterToPhonemeFallback us = LetterToPhonemeFallback.bundled(false);
    assertThat(us.dictionaryWords()).isGreaterThan(70_000);
    // Entries of cmudict-us.json, words misaki's dictionaries lack, in misaki's spelling.
    assertThat(us.phonemize("Alcock")).isEqualTo("əlkˈɑk");
    assertThat(us.phonemize("rosebush")).isEqualTo("ɹˈOzbˌʊʃ");
    assertThat(us.lookup("Hurlbut's")).isEqualTo("hˈɜɹlbəts");
    assertThat(us.lookup("Alcocks")).isEqualTo("əlkˈɑks");
    assertThat(us.lookup("rosebushes")).isEqualTo("ɹˈOzbˌʊʃᵻz");
    assertThat(us.lookup("Alcocks'")).isEqualTo("əlkˈɑks");
    assertThat(us.lookup("xenoceratops")).isNull();
    assertThat(us.phonemize("xenoceratops")).isNotBlank();
    // The British accent has no dictionary tier; its words all go to the model.
    assertThat(LetterToPhonemeFallback.bundled(true).dictionaryWords()).isZero();
  }

  @SuppressWarnings("unchecked")
  private static List<Map<String, Object>> sample(String accent) throws Exception {
    String path = "g2p/g2p-" + accent + "-dev-sample.json";
    try (InputStream in =
        LetterToPhonemeFallbackTest.class.getClassLoader().getResourceAsStream(path)) {
      assertThat(in).as(path).isNotNull();
      String json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
      return (List<Map<String, Object>>) JsonAccess.object("{\"rows\":" + json + "}").get("rows");
    }
  }
}
