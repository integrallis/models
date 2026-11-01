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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MisakiLexiconTest {

  /** A tiny gold dictionary with the entries the special cases and number reading reach for. */
  static Map<String, Object> gold() {
    Map<String, Object> g = new HashMap<>();
    g.put("the", "ði");
    g.put("a", "A");
    g.put("A", "ˈA");
    g.put("am", "æm");
    g.put("an", "æn");
    g.put("to", "tu");
    g.put("in", "ɪn");
    g.put("I", "ˈI");
    g.put("by", Map.of("DEFAULT", "bI", "None", "bˈI"));
    g.put("used", Map.of("DEFAULT", "jˈuzd", "VBD", "jˈust"));
    g.put("read", Map.of("ADJ", "ɹˈɛd", "DEFAULT", "ɹˈid", "VBD", "ɹˈɛd"));
    Map<String, String> aa = new HashMap<>();
    aa.put("DEFAULT", "ˈɑˌɑ");
    aa.put("NOUN", null);
    g.put("AA", aa);
    g.put("cat", "kˈæt");
    g.put("dog", "dˈɔɡ");
    g.put("walk", "wˈɔk");
    g.put("play", "plˈA");
    g.put("hop", "hˈɑp");
    g.put("bake", "bˈAk");
    g.put("wait", "wˈAt");
    g.put("need", "nˈid");
    g.put("fish", "fˈɪʃ");
    g.put("fly", "flˈI");
    g.put("minus", "mˈInəs");
    g.put("point", "pˈYnt");
    g.put("O", "ˈO");
    g.put("oh", "ˈO");
    g.put("zero", "zˈɪɹO");
    g.put("one", "wˈʌn");
    g.put("two", "tˈu");
    g.put("three", "θɹˈi");
    g.put("four", "fˈɔɹ");
    g.put("five", "fˈIv");
    g.put("six", "sˈɪks");
    g.put("nine", "nˈIn");
    g.put("ten", "tˈɛn");
    g.put("twenty", "twˈɛnti");
    g.put("hundred", "hˈʌndɹəd");
    g.put("thousand", "θˈWzᵊnd");
    g.put("nineteen", "nˌIntˈin");
    g.put("ninety", "nˈInti");
    g.put("eight", "ˈAt");
    g.put("first", "fˈɜɹst");
    g.put("second", "sˈɛkənd");
    g.put("and", "ænd");
    g.put("dollar", "dˈɑləɹ");
    g.put("cent", "sˈɛnt");
    g.put("percent", "pəɹsˈɛnt");
    g.put("versus", "vˈɜɹsəs");
    g.put("dot", "dˈɑt");
    g.put("plus", "plˈʌs");
    g.put("at", "æt");
    g.put("slash", "slˈæʃ");
    g.put("B", "bˈi");
    g.put("C", "sˈi");
    g.put("U", "jˈu");
    g.put("S", "ˈɛs");
    g.put("N", "ˈɛn");
    g.put("Paris", "pˈɛɹɪs");
    g.put("don't", "dˈOnt");
    g.put("cat's", "kˈæts");
    g.put("M", "ˈɛm");
    g.put("fourteen", "fˈɔɹtˈin");
    g.put("forty", "fˈɔɹti");
    g.put("it's", "ɪts");
    return g;
  }

  static Map<String, Object> silver() {
    Map<String, Object> s = new HashMap<>();
    s.put("runs", "ɹˈʌnz");
    s.put("tree", "tɹˈi");
    return s;
  }

  static MisakiLexicon us() {
    return MisakiLexicon.of(gold(), silver(), false);
  }

  static MisakiLexicon gb() {
    return MisakiLexicon.of(gold(), silver(), true);
  }

  private static final MisakiLexicon.Context NO_CONTEXT = MisakiLexicon.Context.INITIAL;

  @Test
  void growsTheDictionaryAcrossCapitalisation() {
    MisakiLexicon lexicon = us();
    assertThat(lexicon.lookup("Cat", null, null, NO_CONTEXT).phonemes()).isEqualTo("kˈæt");
    assertThat(lexicon.lookup("paris", null, null, NO_CONTEXT).phonemes()).isEqualTo("pˈɛɹɪs");
    assertThat(MisakiLexicon.capitalize("")).isEmpty();
    assertThat(MisakiLexicon.capitalize("hELLO")).isEqualTo("Hello");
    assertThatThrownBy(() -> MisakiLexicon.of(Map.of("x", Map.of("NOUN", "y")), Map.of(), false))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> MisakiLexicon.of(Map.of("x", 3L), Map.of(), false))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void resolvesTaggedEntriesAndSpellsInitialisms() {
    MisakiLexicon lexicon = us();
    assertThat(lexicon.lookup("read", "VBD", null, NO_CONTEXT).phonemes()).isEqualTo("ɹˈɛd");
    assertThat(lexicon.lookup("read", "JJ", null, NO_CONTEXT).phonemes()).isEqualTo("ɹˈɛd");
    assertThat(lexicon.lookup("read", "NN", null, NO_CONTEXT).phonemes()).isEqualTo("ɹˈid");
    assertThat(lexicon.lookup("by", "IN", null, NO_CONTEXT).phonemes()).isEqualTo("bˈI");
    assertThat(lexicon.lookup("by", "IN", null, new MisakiLexicon.Context(true, false)).phonemes())
        .isEqualTo("bI");
    // A tag whose stored value is null falls through to letter-by-letter spelling.
    assertThat(lexicon.lookup("AA", "NN", null, NO_CONTEXT).phonemes()).isEqualTo("ˌAˈA");
    assertThat(lexicon.lookup("AA", "VB", null, NO_CONTEXT).phonemes()).isEqualTo("ˈɑˌɑ");
    assertThat(lexicon.getNnp("U.S.").phonemes()).isEqualTo("jˌuˈɛs");
    assertThat(lexicon.getNnp("UX").found()).isFalse();
    assertThat(lexicon.lookup("BCS", "NNP", null, NO_CONTEXT).phonemes()).isEqualTo("bˌisˌiˈɛs");
    assertThat(lexicon.lookup("ZZZ", "NNP", null, NO_CONTEXT).found()).isFalse();
    assertThat(MisakiLexicon.getParentTag(null)).isNull();
    assertThat(MisakiLexicon.getParentTag("RBR")).isEqualTo("ADV");
    assertThat(MisakiLexicon.getParentTag("DT")).isEqualTo("DT");
  }

  @Test
  void handlesTheSpecialCaseWords() {
    MisakiLexicon lexicon = us();
    MisakiLexicon.Context vowel = new MisakiLexicon.Context(true, false);
    MisakiLexicon.Context consonant = new MisakiLexicon.Context(false, false);
    assertThat(lexicon.getSpecialCase("a", "DT", null, NO_CONTEXT).phonemes()).isEqualTo("ɐ");
    assertThat(lexicon.getSpecialCase("A", "NN", null, NO_CONTEXT).phonemes()).isEqualTo("ˈA");
    assertThat(lexicon.getSpecialCase("am", "VBP", null, NO_CONTEXT).phonemes()).isEqualTo("æm");
    assertThat(lexicon.getSpecialCase("am", "VBP", null, vowel).phonemes()).isEqualTo("ɐm");
    assertThat(lexicon.getSpecialCase("AM", "NNP", null, vowel).phonemes()).isEqualTo("ˌAˈɛm");
    assertThat(lexicon.getSpecialCase("an", "DT", null, NO_CONTEXT).phonemes()).isEqualTo("ɐn");
    assertThat(lexicon.getSpecialCase("AN", "NNP", null, NO_CONTEXT).phonemes()).isEqualTo("ˌAˈɛn");
    assertThat(lexicon.getSpecialCase("I", "PRP", null, NO_CONTEXT).phonemes()).isEqualTo("ˌI");
    assertThat(lexicon.getSpecialCase("by", "RB", null, NO_CONTEXT).phonemes()).isEqualTo("bˈI");
    assertThat(lexicon.getSpecialCase("to", "TO", null, NO_CONTEXT).phonemes()).isEqualTo("tu");
    assertThat(lexicon.getSpecialCase("to", "TO", null, vowel).phonemes()).isEqualTo("tʊ");
    assertThat(lexicon.getSpecialCase("to", "TO", null, consonant).phonemes()).isEqualTo("tə");
    assertThat(lexicon.getSpecialCase("in", "IN", null, NO_CONTEXT).phonemes()).isEqualTo("ˈɪn");
    assertThat(lexicon.getSpecialCase("in", "IN", null, vowel).phonemes()).isEqualTo("ɪn");
    assertThat(lexicon.getSpecialCase("the", "DT", null, vowel).phonemes()).isEqualTo("ði");
    assertThat(lexicon.getSpecialCase("the", "DT", null, consonant).phonemes()).isEqualTo("ðə");
    assertThat(lexicon.getSpecialCase("vs.", "IN", null, NO_CONTEXT).phonemes())
        .isEqualTo("vˈɜɹsəs");
    assertThat(
            lexicon
                .getSpecialCase("used", "VBD", null, new MisakiLexicon.Context(null, true))
                .phonemes())
        .isEqualTo("jˈust");
    assertThat(lexicon.getSpecialCase("used", "VBN", null, NO_CONTEXT).phonemes())
        .isEqualTo("jˈuzd");
    assertThat(lexicon.getSpecialCase("%", "NN", null, NO_CONTEXT).phonemes())
        .isEqualTo("pəɹsˈɛnt");
    assertThat(lexicon.getSpecialCase(".", "ADD", null, NO_CONTEXT).phonemes()).isEqualTo("dˌɑt");
    assertThat(lexicon.getSpecialCase("U.S.", "NNP", null, NO_CONTEXT).phonemes())
        .isEqualTo("jˌuˈɛs");
    assertThat(lexicon.getSpecialCase("cat", "NN", null, NO_CONTEXT).found()).isFalse();
  }

  @Test
  void stemsPluralsPastTensesAndProgressivesInBothAccents() {
    MisakiLexicon us = us();
    MisakiLexicon gb = gb();
    assertThat(us.getWord("cats", "NNS", null, NO_CONTEXT).phonemes()).isEqualTo("kˈæts");
    assertThat(us.getWord("dogs", "NNS", null, NO_CONTEXT).phonemes()).isEqualTo("dˈɔɡz");
    assertThat(us.getWord("fishes", "NNS", null, NO_CONTEXT).phonemes()).isEqualTo("fˈɪʃᵻz");
    assertThat(gb.getWord("fishes", "NNS", null, NO_CONTEXT).phonemes()).isEqualTo("fˈɪʃɪz");
    assertThat(us.getWord("flies", "NNS", null, NO_CONTEXT).phonemes()).isEqualTo("flˈIz");
    assertThat(us.getWord("cat's", "NN", null, NO_CONTEXT).phonemes()).isEqualTo("kˈæts");
    assertThat(us.getWord("walked", "VBD", null, NO_CONTEXT).phonemes()).isEqualTo("wˈɔkt");
    assertThat(us.getWord("played", "VBD", null, NO_CONTEXT).phonemes()).isEqualTo("plˈAd");
    assertThat(us.getWord("needed", "VBD", null, NO_CONTEXT).phonemes()).isEqualTo("nˈidᵻd");
    assertThat(gb.getWord("needed", "VBD", null, NO_CONTEXT).phonemes()).isEqualTo("nˈidɪd");
    assertThat(us.getWord("waited", "VBD", null, NO_CONTEXT).phonemes()).isEqualTo("wˈAɾᵻd");
    assertThat(gb.getWord("waited", "VBD", null, NO_CONTEXT).phonemes()).isEqualTo("wˈAtɪd");
    assertThat(us.getWord("baked", "VBD", null, NO_CONTEXT).phonemes()).isEqualTo("bˈAkt");
    assertThat(us.getWord("walking", "VBG", null, NO_CONTEXT).phonemes()).isEqualTo("wˈɔkɪŋ");
    assertThat(us.getWord("baking", "VBG", null, NO_CONTEXT).phonemes()).isEqualTo("bˈAkɪŋ");
    assertThat(us.getWord("hopping", "VBG", null, NO_CONTEXT).phonemes()).isEqualTo("hˈɑpɪŋ");
    assertThat(us.getWord("waiting", "VBG", null, NO_CONTEXT).phonemes()).isEqualTo("wˈAɾɪŋ");
    assertThat(gb.getWord("waiting", "VBG", null, NO_CONTEXT).phonemes()).isEqualTo("wˈAtɪŋ");
    assertThat(us.getWord("cats'", "NNS", null, NO_CONTEXT).phonemes()).isEqualTo("kˈæts");
    assertThat(us.getWord("dogs'", "NNS", null, NO_CONTEXT).found()).isFalse();
    assertThat(us.getWord("cat'", "NN", null, NO_CONTEXT).phonemes()).isEqualTo("kˈæt");
    assertThat(us.getWord("Cats", "NNS", 0.5, NO_CONTEXT).phonemes()).isEqualTo("kˈæts");
    assertThat(us.getWord("zzz", "NN", null, NO_CONTEXT).found()).isFalse();
    assertThat(us.getWord("zzzed", "VBD", null, NO_CONTEXT).found()).isFalse();
    assertThat(us.getWord("zzzing", "VBG", null, NO_CONTEXT).found()).isFalse();
    assertThat(us.pluralS(null)).isNull();
    assertThat(us.pastEd("")).isNull();
    assertThat(us.progressiveIng(null)).isNull();
    assertThat(gb.progressiveIng("bˈɛtə")).isNull();
    assertThat(us.pastEd("t")).isEqualTo("tɪd");
  }

  @Test
  void knowsWhatItKnows() {
    MisakiLexicon lexicon = us();
    assertThat(lexicon.isKnown("cat", null)).isTrue();
    assertThat(lexicon.isKnown("runs", null)).isTrue();
    assertThat(lexicon.isKnown("%", null)).isTrue();
    assertThat(lexicon.isKnown("x", null)).isTrue();
    assertThat(lexicon.isKnown("CAT", null)).isTrue();
    assertThat(lexicon.isKnown("NASA", null)).isTrue();
    assertThat(lexicon.isKnown("nasa", null)).isFalse();
    assertThat(lexicon.isKnown("café", null)).isFalse();
    assertThat(lexicon.isKnown("a1", null)).isFalse();
    assertThat(MisakiLexicon.allLexiconOrds("it's-ok")).isTrue();
    assertThat(MisakiLexicon.allLexiconOrds("naïve")).isFalse();
  }

  @Test
  void readsNumbersCurrencyAndSuffixes() {
    MisakiLexicon lexicon = us();
    assertThat(lexicon.getNumber("1998", null, true, "").phonemes())
        .isEqualTo("nˌIntˈin nˈInti ˈAt");
    assertThat(lexicon.getNumber("22nd", null, true, "").phonemes()).isEqualTo("twˈɛnti sˈɛkənd");
    assertThat(lexicon.getNumber("3", null, true, "").phonemes()).isEqualTo("θɹˈi");
    assertThat(lexicon.getNumber("1,000", null, true, "").phonemes()).isEqualTo("wˈʌn θˈWzᵊnd");
    assertThat(lexicon.getNumber("105", null, true, "").phonemes()).isEqualTo("wˈʌn hˈʌndɹəd fˈIv");
    assertThat(lexicon.getNumber("105", null, true, "&").phonemes())
        .isEqualTo("wˈʌn hˈʌndɹəd ænd fˈIv");
    assertThat(lexicon.getNumber("105", null, true, "n").phonemes())
        .isEqualTo("wˈʌn hˈʌndɹədən fˈIv");
    assertThat(lexicon.getNumber("110", null, true, "a").phonemes()).isEqualTo("ə hˈʌndɹəd tˈɛn");
    assertThat(lexicon.getNumber("4.2", null, true, "").phonemes()).isEqualTo("fˈɔɹ pYnt tˈu");
    assertThat(lexicon.getNumber(".5", null, true, "").phonemes()).isEqualTo("pYnt fˈIv");
    assertThat(lexicon.getNumber("-3", null, true, "").phonemes()).isEqualTo("mˈInəs θɹˈi");
    assertThat(lexicon.getNumber("05", null, false, "").phonemes()).isEqualTo("zˈɪɹO fˈIv");
    assertThat(lexicon.getNumber("306", null, false, "").phonemes()).isEqualTo("θɹˈi O sˈɪks");
    assertThat(lexicon.getNumber("325", null, false, "").phonemes()).isEqualTo("θɹˈi twˈɛnti fˈIv");
    assertThat(lexicon.getNumber("42", null, false, "").phonemes()).isEqualTo("fˈɔɹti tˈu");
    assertThat(lexicon.getNumber("1.2.3", null, true, "").phonemes()).isEqualTo("wˈʌn tˈu θɹˈi");
    assertThat(lexicon.getNumber("3.14", null, false, "").phonemes()).isEqualTo("θɹˈi fˈɔɹtˈin");
    assertThat(lexicon.getNumber("3.104", null, false, "").phonemes())
        .isEqualTo("θɹˈi wˈʌn zˈɪɹO fˈɔɹ");
    assertThat(lexicon.getNumber("4.20", "$", true, "").phonemes())
        .isEqualTo("fˈɔɹ dˈɑləɹz ænd twˈɛnti sˈɛnts");
    assertThat(lexicon.getNumber("1", "$", true, "").phonemes()).isEqualTo("wˈʌn dˈɑləɹ");
    assertThat(lexicon.getNumber("0.05", "$", true, "").phonemes()).isEqualTo("fˈIv sˈɛnts");
    assertThat(lexicon.getNumber("3s", null, true, "").phonemes()).isEqualTo("θɹˈiz");
    assertThat(lexicon.getNumber("2's", null, true, "").phonemes()).isEqualTo("tˈuz");
    assertThat(lexicon.getNumber("1ed", null, true, "").phonemes()).isEqualTo("wˈʌnd");
    assertThat(lexicon.getNumber("1ing", null, true, "").phonemes()).isEqualTo("wˈʌnɪŋ");
    assertThat(lexicon.getNumber("7", null, true, "").found()).isFalse();
    assertThat(MisakiLexicon.isNumber("22nd", true)).isTrue();
    assertThat(MisakiLexicon.isNumber("-2", true)).isTrue();
    assertThat(MisakiLexicon.isNumber("-2", false)).isFalse();
    assertThat(MisakiLexicon.isNumber("abc", true)).isFalse();
    assertThat(MisakiLexicon.isNumber("2x", true)).isFalse();
    assertThat(MisakiLexicon.isCurrency("4.2")).isTrue();
    assertThat(MisakiLexicon.isCurrency("4.123")).isFalse();
    assertThat(MisakiLexicon.isCurrency("4.000")).isTrue();
    assertThat(MisakiLexicon.isCurrency("1.2.3")).isFalse();
    assertThat(MisakiLexicon.numericIfNeeded('٣')).isEqualTo("3");
    assertThat(MisakiLexicon.numericIfNeeded('x')).isEqualTo("x");
    assertThat(lexicon.appendCurrency("fˈIv", "$")).isEqualTo("fˈIv dˈɑləɹz");
    assertThat(lexicon.appendCurrency("fˈIv", null)).isEqualTo("fˈIv");
    assertThat(lexicon.appendCurrency("fˈIv", "¥")).isEqualTo("fˈIv");
  }

  @Test
  void phonemizesTokensWithCapitalisationStressAndNumbers() {
    MisakiLexicon lexicon = us();
    MToken cat = new MToken("Cat", "NN", " ");
    assertThat(lexicon.phonemize(cat, NO_CONTEXT).phonemes()).isEqualTo("kˈæt");
    MToken shout = new MToken("CAT", "NN", " ");
    assertThat(lexicon.phonemize(shout, NO_CONTEXT).phonemes()).isEqualTo("kˈæt");
    MToken year = new MToken("1998", "CD", " ");
    assertThat(lexicon.phonemize(year, NO_CONTEXT).phonemes()).isEqualTo("nˌIntˈin nˈInti ˈAt");
    MToken priced = new MToken("5", "CD", " ");
    priced.currency = "$";
    assertThat(lexicon.phonemize(priced, NO_CONTEXT).phonemes()).isEqualTo("fˈIv dˈɑləɹz");
    MToken curly = new MToken("don’t", "VB", " ");
    assertThat(lexicon.phonemize(curly, NO_CONTEXT).phonemes()).isEqualTo("dˈOnt");
    MToken alias = new MToken("2", "CD", " ");
    alias.alias = "to";
    assertThat(lexicon.phonemize(alias, NO_CONTEXT).phonemes()).isEqualTo("tu");
    MToken stressed = new MToken("cat", "NN", " ");
    stressed.stress = 2.0;
    assertThat(lexicon.phonemize(stressed, NO_CONTEXT).phonemes()).isEqualTo("kˈæt");
    assertThat(lexicon.phonemize(new MToken("naïve", "NN", " "), NO_CONTEXT).found()).isFalse();
    assertThat(lexicon.phonemize(new MToken("zzz", "NN", " "), NO_CONTEXT).found()).isFalse();
  }
}
