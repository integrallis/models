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
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The dictionaries shipped in this jar are misaki 0.9.4's, and both accents load from them. */
class MisakiBundledLexiconTest {

  /** SHA-256 of each file in the misaki 0.9.4 wheel's {@code misaki/data/}, as META-INF/NOTICE. */
  private static final Map<String, String> WHEEL_DIGESTS =
      Map.of(
          "us_gold.json", "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f",
          "us_silver.json", "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475",
          "gb_gold.json", "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71",
          "gb_silver.json", "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce");

  @Test
  void bundledDictionariesAreTheWheelsBytes() throws Exception {
    for (Map.Entry<String, String> expected : WHEEL_DIGESTS.entrySet()) {
      try (InputStream in =
          MisakiLexicon.class
              .getClassLoader()
              .getResourceAsStream(MisakiLexicon.BUNDLED_RESOURCE_ROOT + expected.getKey())) {
        assertThat(in).as(expected.getKey()).isNotNull();
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(in.readAllBytes());
        assertThat(HexFormat.of().formatHex(digest))
            .as(expected.getKey())
            .isEqualTo(expected.getValue());
      }
    }
  }

  @Test
  void agreesWithMisakiOnTheCasesTheFirstExactMatchPassDisagreedOn() throws Exception {
    // Expected strings are misaki 0.9.4's own output on the comparator pass for these sentences
    // (evidence/devbox/comparator/wikitext103-heldout-200/af_heart/t1/pass1, rows 29 and 49).
    MisakiEnglishPhonemizer us = new MisakiEnglishPhonemizer(MisakiLexicon.bundled(false), "");
    assertThat(us.phonemize("John Wilson writes that the tomb lies.")).contains("ðæt ðə tˈum");
    assertThat(us.phonemize("On 3 July 2012, US Secretary of State")).contains("jˌuˈɛs sˈɛkɹətˌ");
  }

  @Test
  void bothAccentsLoadAndResolveCommonWords() throws Exception {
    MisakiLexicon us = MisakiLexicon.bundled(false);
    MisakiLexicon gb = MisakiLexicon.bundled(true);
    assertThat(us.british()).isFalse();
    assertThat(gb.british()).isTrue();
    MisakiEnglishPhonemizer.Result american =
        new MisakiEnglishPhonemizer(us, "").convert("The cat sat.");
    MisakiEnglishPhonemizer.Result british =
        new MisakiEnglishPhonemizer(gb, "").convert("The cat sat.");
    assertThat(american.unknownWords()).isEmpty();
    assertThat(british.unknownWords()).isEmpty();
    assertThat(american.phonemes()).contains("kˈæt");
    assertThat(british.phonemes()).isNotBlank().isNotEqualTo(american.phonemes());
  }
}
