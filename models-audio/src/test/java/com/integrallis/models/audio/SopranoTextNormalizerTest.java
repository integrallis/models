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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SopranoTextNormalizerTest {

  @Test
  void matchesTheOfficialNormalizerForTechnicalTextNumbersAndPunctuation() {
    assertThat(SopranoTextNormalizer.normalize("Dr. Smith uses 2 GPUs at 8:05."))
        .isEqualTo("doctor smith uses two g p u's at eight oh five.");
    assertThat(SopranoTextNormalizer.normalize("The API costs $2.47 & runs at 3.5 kHz."))
        .isEqualTo(
            "the a p i costs two dollars, forty-seven cents and runs at three point five kilohertz.");
    assertThat(SopranoTextNormalizer.normalize("Call (575) 395-1234 on 9/4/2026."))
        .isEqualTo(
            "call five seven five, three nine five, one two three four on nine dash four dash twenty twenty-six.");
    assertThat(SopranoTextNormalizer.normalize("Café — déjà vu!!!")).isEqualTo("cafe, deja vu!");
  }

  @Test
  void matchesTheOfficialNormalizerForCodeLikeAndMultilineText() {
    assertThat(SopranoTextNormalizer.normalize("LMDeployDecoderModel"))
        .isEqualTo("l m deploy decoder model.");
    assertThat(SopranoTextNormalizer.normalize("Line one\nLine two"))
        .isEqualTo("line one. line two.");
    assertThat(SopranoTextNormalizer.normalize("I am soooo happy..."))
        .isEqualTo("i am soo happy...");
    assertThat(SopranoTextNormalizer.normalize("Visit https://modeljars.org/a_b"))
        .isEqualTo("visit h t t p s colon slash slash modeljars dot org slash a b.");
  }
}
