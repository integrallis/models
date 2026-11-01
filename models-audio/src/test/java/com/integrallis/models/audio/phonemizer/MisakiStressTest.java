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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MisakiStressTest {

  @Test
  void appliesEveryStressLevelTheOriginalDistinguishes() {
    assertThat(MisakiStress.applyStress("hˈɛlO", null)).isEqualTo("hˈɛlO");
    assertThat(MisakiStress.applyStress(null, 1.0)).isNull();
    assertThat(MisakiStress.applyStress("hˈɛlˌO", -2.0)).isEqualTo("hɛlO");
    assertThat(MisakiStress.applyStress("hˈɛlˌO", -1.0)).isEqualTo("hˌɛlO");
    assertThat(MisakiStress.applyStress("hˈɛlO", 0.0)).isEqualTo("hˌɛlO");
    assertThat(MisakiStress.applyStress("hˈɛlO", -0.5)).isEqualTo("hˌɛlO");
    assertThat(MisakiStress.applyStress("hɛlO", 0.0)).isEqualTo("hˌɛlO");
    assertThat(MisakiStress.applyStress("hɛlO", 0.5)).isEqualTo("hˌɛlO");
    assertThat(MisakiStress.applyStress("hɛlO", 1.0)).isEqualTo("hˌɛlO");
    assertThat(MisakiStress.applyStress("hˌɛlO", 1.0)).isEqualTo("hˈɛlO");
    assertThat(MisakiStress.applyStress("hɛlO", 2.0)).isEqualTo("hˈɛlO");
    assertThat(MisakiStress.applyStress("sss", 2.0)).isEqualTo("sss");
    assertThat(MisakiStress.applyStress("sss", 0.0)).isEqualTo("sss");
    assertThat(MisakiStress.applyStress("hˈɛlO", 2.0)).isEqualTo("hˈɛlO");
    assertThat(MisakiStress.applyStress("hˌɛlˌO", 0.5)).isEqualTo("hˌɛlˌO");
  }

  @Test
  void movesStressMarksToTheNextVowelAndWeighsDiphthongs() {
    assertThat(MisakiStress.restress("ˌstɹɔŋ")).isEqualTo("stɹˌɔŋ");
    assertThat(MisakiStress.restress("ˈsss")).isEqualTo("ˈsss");
    assertThat(MisakiStress.stressWeight("bˈI")).isEqualTo(4);
    assertThat(MisakiStress.stressWeight("bi")).isEqualTo(2);
    assertThat(MisakiStress.stressWeight("")).isZero();
    assertThat(MisakiStress.stressWeight(null)).isZero();
    assertThat(MisakiStress.hasVowel("ps")).isFalse();
    assertThat(MisakiStress.isStress('ˈ')).isTrue();
    assertThat(MisakiStress.chars("ab")).containsExactlyInAnyOrder('a', 'b');
  }
}
