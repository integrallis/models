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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Runs the phonemizer over misaki's real dictionaries when {@code MODELS_MISAKI_DATA_DIR} holds
 * {@code us_gold.json}, {@code us_silver.json}, {@code gb_gold.json} and {@code gb_silver.json}.
 * Prints the phonemes for the Paradee card sentences so they can be compared with the publisher's
 * pipeline; asserts only what does not depend on spaCy.
 */
@Tag("integration")
class MisakiLexiconIntegrationTest {

  private static final List<String> CARD =
      List.of(
          "The library opens at nine, but the coffee shop next door is usually busy by eight.",
          "Honestly, I didn't expect the experiment to work on the very first try!",
          "Could you read me the last paragraph again, a little more slowly this time?",
          "In 1969, a little over six hundred million people watched two astronauts walk on the moon. Most of them saw it on small black and white televisions, and many stayed up past midnight to see it happen.",
          "Turn left at the second light, then keep going until you see the bakery.",
          "Paradee is a small voice that runs anywhere.",
          "Dr. Smith arrived at 3 p.m. on Tuesday, carrying an umbrella, a laptop, and a very large sandwich.",
          "In 1998, the company reported revenue of 4.2 billion dollars, up 17 percent from the year before.");

  @Test
  void phonemizesTheCardSentencesInBothAccents() throws Exception {
    String configured = System.getenv("MODELS_MISAKI_DATA_DIR");
    assumeTrue(configured != null, "MODELS_MISAKI_DATA_DIR not set");
    Path dir = Path.of(configured);
    assumeTrue(Files.isRegularFile(dir.resolve("us_gold.json")));
    for (boolean british : new boolean[] {false, true}) {
      String prefix = british ? "gb" : "us";
      long start = System.nanoTime();
      MisakiLexicon lexicon =
          MisakiLexicon.load(
              dir.resolve(prefix + "_gold.json"), dir.resolve(prefix + "_silver.json"), british);
      long loadMillis = (System.nanoTime() - start) / 1_000_000L;
      MisakiEnglishPhonemizer g2p = new MisakiEnglishPhonemizer(lexicon, "");
      System.out.printf("MISAKI %s lexicon loaded in %d ms%n", prefix, loadMillis);
      for (String sentence : CARD) {
        MisakiEnglishPhonemizer.Result result = g2p.convert(sentence);
        System.out.printf("MISAKI %s | %s%n", prefix, sentence);
        System.out.printf("MISAKI %s > %s%n", prefix, result.phonemes());
        if (!result.unknownWords().isEmpty()) {
          System.out.printf("MISAKI %s ! unknown: %s%n", prefix, result.unknownWords());
        }
        assertThat(result.phonemes()).isNotBlank();
        assertThat(result.wordCount()).isGreaterThan(3);
      }
      assertThat(g2p.phonemize("the apple")).startsWith("ði ");
      assertThat(g2p.phonemize("the pear")).startsWith("ðə ");
      assertThat(g2p.convert("Paradee is a small voice that runs anywhere.").unknownWords())
          .containsExactly("Paradee");
    }
  }
}
