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

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** misaki's phoneme alphabet facts and its stress-placement rules. */
final class MisakiStress {

  static final char PRIMARY = 'ˈ';
  static final char SECONDARY = 'ˌ';
  static final Set<Character> VOWELS = chars("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ");
  static final Set<Character> CONSONANTS = chars("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ");
  static final Set<Character> DIPHTHONGS = chars("AIOQWYʤʧ");
  static final Set<Character> US_TAUS = chars("AIOWYiuæɑəɛɪɹʊʌ");
  static final Set<Character> PUNCTS = chars(";:,.!?—…\"“”");
  static final Set<Character> NON_QUOTE_PUNCTS = chars(";:,.!?—…");
  static final Set<Character> SUBTOKEN_JUNKS = chars("',-._‘’/");

  private MisakiStress() {}

  static Set<Character> chars(String text) {
    Set<Character> set = new java.util.HashSet<>();
    for (char c : text.toCharArray()) {
      set.add(c);
    }
    return Set.copyOf(set);
  }

  static boolean isStress(char c) {
    return c == PRIMARY || c == SECONDARY;
  }

  static boolean hasPrimary(String ps) {
    return ps.indexOf(PRIMARY) >= 0;
  }

  static boolean hasSecondary(String ps) {
    return ps.indexOf(SECONDARY) >= 0;
  }

  static boolean hasAnyStress(String ps) {
    return hasPrimary(ps) || hasSecondary(ps);
  }

  static boolean hasVowel(String ps) {
    for (char c : ps.toCharArray()) {
      if (VOWELS.contains(c)) {
        return true;
      }
    }
    return false;
  }

  /** Two for diphthongs and affricates, one for everything else: misaki's {@code stress_weight}. */
  static int stressWeight(String ps) {
    if (ps == null || ps.isEmpty()) {
      return 0;
    }
    int weight = 0;
    for (char c : ps.toCharArray()) {
      weight += DIPHTHONGS.contains(c) ? 2 : 1;
    }
    return weight;
  }

  /** misaki's {@code apply_stress}: a stress level of {@code null} leaves the string alone. */
  static String applyStress(String ps, Double stress) {
    if (stress == null || ps == null) {
      return ps;
    }
    double s = stress;
    if (s < -1) {
      return ps.replace(String.valueOf(PRIMARY), "").replace(String.valueOf(SECONDARY), "");
    }
    if (s == -1 || ((s == 0 || s == -0.5) && hasPrimary(ps))) {
      return ps.replace(String.valueOf(SECONDARY), "").replace(PRIMARY, SECONDARY);
    }
    if ((s == 0 || s == 0.5 || s == 1) && !hasAnyStress(ps)) {
      return hasVowel(ps) ? restress(SECONDARY + ps) : ps;
    }
    if (s >= 1 && !hasPrimary(ps) && hasSecondary(ps)) {
      return ps.replace(SECONDARY, PRIMARY);
    }
    if (s > 1 && !hasAnyStress(ps)) {
      return hasVowel(ps) ? restress(PRIMARY + ps) : ps;
    }
    return ps;
  }

  /** Moves each stress mark to just before the next vowel, as misaki's {@code restress} does. */
  static String restress(String ps) {
    // Each character gets a position; a stress mark takes the position of the next vowel minus
    // one half, and the characters are re-emitted in position order.
    List<double[]> positions = new ArrayList<>();
    char[] chars = ps.toCharArray();
    for (int i = 0; i < chars.length; i++) {
      double position = i;
      if (isStress(chars[i])) {
        int j = i;
        while (j < chars.length && !VOWELS.contains(chars[j])) {
          j++;
        }
        if (j < chars.length) {
          position = j - 0.5;
        }
      }
      positions.add(new double[] {position, i});
    }
    positions.sort((a, b) -> Double.compare(a[0], b[0]));
    StringBuilder out = new StringBuilder();
    for (double[] entry : positions) {
      out.append(chars[(int) entry[1]]);
    }
    return out.toString();
  }
}
