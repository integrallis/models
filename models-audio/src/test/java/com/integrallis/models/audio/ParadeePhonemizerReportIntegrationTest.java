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

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.audio.phonemizer.JsonAccess;
import com.integrallis.models.audio.phonemizer.MisakiEnglishPhonemizer;
import com.integrallis.models.audio.phonemizer.MisakiLexicon;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The phonemizer's own report over a corpus, both accents: the phoneme string per sentence and the
 * words the lexicon path could not resolve. The qualification protocol reports the
 * out-of-vocabulary word rate beside the exact-match rate against the publisher's phonemes; this
 * writes the former.
 *
 * <p>Environment: {@code MODELS_MISAKI_DATA_DIR}, {@code MODELS_PARADEE_CORPUS} (a corpus JSON),
 * {@code MODELS_PARADEE_PHONEMIZER_OUT} (a directory; one JSON file per accent is written).
 */
@Tag("integration")
class ParadeePhonemizerReportIntegrationTest {

  @Test
  void reportsPhonemesAndUnknownWordsForBothAccents() throws Exception {
    String lexiconDir = System.getenv("MODELS_MISAKI_DATA_DIR");
    String corpusPath = System.getenv("MODELS_PARADEE_CORPUS");
    String outDir = System.getenv("MODELS_PARADEE_PHONEMIZER_OUT");
    assumeTrue(
        lexiconDir != null && corpusPath != null && outDir != null,
        "phonemizer report environment not set");

    Map<String, Object> corpus =
        JsonAccess.object(Files.readString(Path.of(corpusPath), StandardCharsets.UTF_8));
    @SuppressWarnings("unchecked")
    List<String> sentences = (List<String>) corpus.get("sentences");
    Path out = Path.of(outDir);
    Files.createDirectories(out);

    for (boolean british : new boolean[] {false, true}) {
      String accent = british ? "gb" : "us";
      MisakiLexicon lexicon =
          MisakiLexicon.load(
              Path.of(lexiconDir, accent + "_gold.json"),
              Path.of(lexiconDir, accent + "_silver.json"),
              british);
      MisakiEnglishPhonemizer phonemizer = new MisakiEnglishPhonemizer(lexicon, "");
      List<Map<String, Object>> rows = new ArrayList<>();
      int words = 0;
      int unknown = 0;
      int sentencesWithUnknown = 0;
      Map<String, Integer> unknownCounts = new TreeMap<>();
      for (int i = 0; i < sentences.size(); i++) {
        String text = sentences.get(i);
        MisakiEnglishPhonemizer.Result result = phonemizer.convert(text);
        words += result.wordCount();
        unknown += result.unknownWords().size();
        if (!result.unknownWords().isEmpty()) {
          sentencesWithUnknown++;
        }
        for (String word : result.unknownWords()) {
          unknownCounts.merge(word, 1, Integer::sum);
        }
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("index", i);
        row.put("text", text);
        row.put("phonemes", result.phonemes());
        row.put("unknownWords", result.unknownWords());
        row.put("wordCount", result.wordCount());
        rows.add(row);
      }
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("phonemizer", "models-audio misaki port, lexicon path only, no eSpeak fallback");
      report.put("corpus", corpus.get("name"));
      report.put("corpusSource", corpus.get("source"));
      report.put("accent", accent);
      report.put("sentences", rows.size());
      report.put("words", words);
      report.put("unknownWords", unknown);
      report.put("unknownWordRate", words > 0 ? unknown / (double) words : null);
      report.put("sentencesWithUnknownWords", sentencesWithUnknown);
      report.put("distinctUnknownWords", unknownCounts);
      report.put("rows", rows);
      Files.writeString(
          out.resolve(corpus.get("name") + "-" + accent + ".json"),
          ParadeeCandidateArmIntegrationTest.toJson(report),
          StandardCharsets.UTF_8);
      System.out.printf(
          "PARADEE_PHONEMIZER %s %s: %d words, %d unknown (%.3f%%), %d of %d sentences touched%n",
          corpus.get("name"),
          accent,
          words,
          unknown,
          words > 0 ? 100.0 * unknown / words : 0.0,
          sentencesWithUnknown,
          rows.size());
    }
  }
}
