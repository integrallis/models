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
package com.integrallis.models.backend.purejava.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BpePreTokenizerTest {

  @Test
  void appliesGpt2WordBoundaries() {
    assertThat(BpePreTokenizer.forName("gpt-2").split("2024 code::foo"))
        .containsExactly("2024", " code", "::", "foo");
  }

  @Test
  void appliesStarCoderSingleDigitBoundaries() {
    assertThat(BpePreTokenizer.forName("starcoder").split("2024 code::foo"))
        .containsExactly("2", "0", "2", "4", " code", "::", "foo");
  }

  @Test
  void appliesSmolLmSingleDigitBoundaries() {
    assertThat(BpePreTokenizer.forName("smollm").split("2024 code::foo"))
        .containsExactly("2", "0", "2", "4", " code", "::", "foo");
  }

  @Test
  void appliesDeepSeekCoderBoundaries() {
    assertThat(BpePreTokenizer.forName("deepseek-coder").split("2024 code::foo\n"))
        .containsExactly("2", "0", "2", "4", " code", "::", "foo", "\n");
  }

  @Test
  void hunyuanSplitsExactlyAsItsOwnTokenizerJsonSpecifies() {
    // Transcribed from tencent/Hunyuan-MT-7B tokenizer.json, whose Split regex is QWEN2_PATTERN
    // alternative for alternative. The distinguishing branch is the number one: single-digit \p{N},
    // so "2024" splits into four tokens, where LLAMA3_PATTERN's \p{N}{1,3} would give "202" + "4".
    assertThat(BpePreTokenizer.forName("hunyuan").split("2024 code::foo"))
        .isEqualTo(BpePreTokenizer.forName("qwen2").split("2024 code::foo"));
    assertThat(BpePreTokenizer.forName("hunyuan").split("2024"))
        .describedAs("single-digit number splitting, not groups of three")
        .containsExactly("2", "0", "2", "4");
    assertThat(BpePreTokenizer.forName("hunyuan").split("2024"))
        .isNotEqualTo(BpePreTokenizer.forName("llama-bpe").split("2024"));
  }

  @Test
  void anUnrecognisedPreTokenizerIsRecordedSoItIsNotInvisible() {
    // The fallback stays non-fatal -- models ship today whose names are absent and whose
    // pinned-token
    // oracles pass -- but it must be answerable rather than silent.
    BpePreTokenizer.forName("a-pretokenizer-we-have-never-heard-of");

    assertThat(BpePreTokenizer.unrecognisedNames())
        .contains("a-pretokenizer-we-have-never-heard-of");
    assertThat(BpePreTokenizer.unrecognisedNames())
        .describedAs("names we do implement must never be reported as unrecognised")
        .doesNotContain("hunyuan", "qwen2", "dbrx", "smollm", "deepseek-r1-qwen");
  }

  @Test
  void appliesQwen35CombiningMarkBoundaries() {
    assertThat(BpePreTokenizer.forName("qwen35").split(" cafe\u0301 2024"))
        .containsExactly(" cafe\u0301", " ", "2", "0", "2", "4");
  }

  @Test
  void appliesDbrxLlama3BoundariesWithoutIgnoringMerges() {
    BpePreTokenizer dbrx = BpePreTokenizer.forName("dbrx");

    assertThat(dbrx.split("1850s \"Islamist\" won 75% in 2024"))
        .as("Granite 4.1 splits digits in groups of three and keeps a space with punctuation")
        .containsExactly(
            "185",
            "0",
            "s",
            " \"",
            "Islamist",
            "\"",
            " won",
            " ",
            "75",
            "%",
            " in",
            " ",
            "202",
            "4");
    assertThat(dbrx.ignoresMerges()).isFalse();
    assertThat(dbrx.split("2024 code::foo"))
        .isEqualTo(BpePreTokenizer.forName("smaug-bpe").split("2024 code::foo"));
  }

  @Test
  void treatsNoBreakSpaceAsWhitespaceLikeTheRustRegex() {
    assertThat(BpePreTokenizer.forName("dbrx").split("risk \u00a0at \u00a0\u00a0home"))
        .as("U+00A0 is Unicode White_Space, so it never joins a punctuation run")
        .containsExactly("risk", " ", "\u00a0at", " \u00a0", "\u00a0home");
    assertThat(BpePreTokenizer.forName("llama-bpe").split("a\u00a0b \u00a0"))
        .as("a lone U+00A0 may prefix a letter run, and a trailing one is a whitespace run")
        .containsExactly("a", "\u00a0b", " \u00a0");
  }

  @Test
  void tekkenSplitsExactlyAsMistralsOwnTekkenJsonSpecifies() {
    // Every expectation here was produced by running the pattern string read out of
    // Mistral-Small-24B-Instruct-2501's own tekken.json (config.pattern, version v7) through an
    // independent regex engine, then transcribed. None was derived by reading the expression: a
    // hand
    // derivation of the apostrophe case was attempted first and was wrong.
    assertThat(BpePreTokenizer.forName("tekken").split("2024 code::foo"))
        .containsExactly("2", "0", "2", "4", " code", "::", "foo");
    assertThat(BpePreTokenizer.forName("tekken").split("don't stop"))
        .describedAs("no contraction group, yet 't stays whole via the optional punctuation prefix")
        .containsExactly("don", "'t", " stop");
    assertThat(BpePreTokenizer.forName("tekken").split("she'll go"))
        .containsExactly("she", "'ll", " go");
    assertThat(BpePreTokenizer.forName("tekken").split("3.14159"))
        .containsExactly("3", ".", "1", "4", "1", "5", "9");
    assertThat(BpePreTokenizer.forName("tekken").split("a//b path/to/x"))
        .describedAs("the trailing [\\r\\n/]* absorbs slashes into the punctuation run")
        .containsExactly("a", "//", "b", " path", "/to", "/x");
    assertThat(BpePreTokenizer.forName("tekken").split("\u00c9COLE \u00e9t\u00e9"))
        .describedAs("real Unicode letter classes, not llama.cpp's [^a-z] std::regex workaround")
        .containsExactly("\u00c9COLE", " \u00e9t\u00e9");
  }

  @Test
  void tekkenIsNotTheGptOssPatternDespiteTheSharedShape() {
    // Both are o200k-shaped, so the tempting move is to reuse GPT_OSS_PATTERN. These are the two
    // places that would have silently changed the token stream.
    assertThat(BpePreTokenizer.forName("tekken").split("don't"))
        .isNotEqualTo(BpePreTokenizer.forName("gpt-oss").split("don't"));
    assertThat(BpePreTokenizer.forName("gpt-oss").split("don't"))
        .describedAs("gpt-oss keeps the contraction whole")
        .containsExactly("don't");
    assertThat(BpePreTokenizer.forName("tekken").split("2024"))
        .isNotEqualTo(BpePreTokenizer.forName("gpt-oss").split("2024"));
    assertThat(BpePreTokenizer.forName("gpt-oss").split("2024"))
        .describedAs("gpt-oss takes digits in groups of up to three")
        .containsExactly("202", "4");
  }

  @Test
  void tekkenIgnoresMergesAndIsNoLongerReportedUnrecognised() {
    // ignore_merges is set for tekken in llama.cpp's own tokenizer_pre dispatch. Getting this wrong
    // costs nothing visible on most input and changes the ids for any piece that is itself a
    // vocabulary entry.
    assertThat(BpePreTokenizer.forName("tekken").ignoresMerges()).isTrue();
    BpePreTokenizer.forName("tekken");
    assertThat(BpePreTokenizer.unrecognisedNames())
        .describedAs("a declared pre-tokenizer we now implement must not be reported as missing")
        .doesNotContain("tekken");
  }

  @Test
  void aPreTokenizerReportsTheNameThatSelectedItAndWhetherItIsImplemented() {
    // The point of these two facts is that a qualification record can carry them. Before they
    // existed the fallback was tracked only in unrecognisedNames(), which is public on a
    // package-private class -- reachable from tests in this package and from nothing else -- so no
    // real run ever reported it.
    assertThat(BpePreTokenizer.forName("tekken").declaredName()).isEqualTo("tekken");
    assertThat(BpePreTokenizer.forName("tekken").isImplemented()).isTrue();
    assertThat(BpePreTokenizer.forName("llama3").isImplemented()).isTrue();
    assertThat(BpePreTokenizer.forName("lfm2").isImplemented()).isTrue();

    // A declared name we do not implement: implemented is false, and the name is kept so the record
    // can say which one it was.
    BpePreTokenizer missing = BpePreTokenizer.forName("some-pretokenizer-we-do-not-have");
    assertThat(missing.declaredName()).isEqualTo("some-pretokenizer-we-do-not-have");
    assertThat(missing.isImplemented()).isFalse();

    // Declaring none is not a gap: every SentencePiece model does this and must not be flagged.
    assertThat(BpePreTokenizer.forName("").declaredName()).isEmpty();
    assertThat(BpePreTokenizer.forName("").isImplemented()).isTrue();
  }

  @Test
  void namingAPreTokenizerDoesNotChangeHowItSplits() {
    // The name is carried on a copy, so tagging must not perturb the pattern or the merge policy.
    assertThat(BpePreTokenizer.forName("tekken").split("2024 code::foo"))
        .containsExactly("2", "0", "2", "4", " code", "::", "foo");
    assertThat(BpePreTokenizer.forName("llama3").ignoresMerges()).isTrue();
    assertThat(BpePreTokenizer.forName("dbrx").ignoresMerges()).isFalse();
    assertThat(BpePreTokenizer.forName("some-pretokenizer-we-do-not-have").split("2024 code"))
        .describedAs("an unimplemented name still falls back to no splitting")
        .containsExactly("2024 code");
  }

  @Test
  void gpt4oSelectsTheSameO200kSplitAsGptOss() {
    // llama.cpp's LLAMA_VOCAB_PRE_TYPE_GPT4O case quotes the original tokenizer.json regex, and it
    // is
    // GPT_OSS_PATTERN alternative for alternative. Phi-4-mini declares this name on a byte-level
    // BPE
    // vocabulary, and phi3 is runnable, so leaving it unmapped meant no pre-splitting at all.
    for (String name : new String[] {"gpt-4o", "llama4", "kanana2", "talkie"}) {
      assertThat(BpePreTokenizer.forName(name).split("don't 2024 code::foo"))
          .describedAs("%s must split as o200k", name)
          .isEqualTo(BpePreTokenizer.forName("gpt-oss").split("don't 2024 code::foo"));
      assertThat(BpePreTokenizer.forName(name).isImplemented()).isTrue();
      assertThat(BpePreTokenizer.forName(name).ignoresMerges())
          .describedAs("the GPT4O branch sets clean_spaces but not ignore_merges")
          .isFalse();
    }
    // And it is genuinely o200k, not tekken: digits in groups of three, contractions kept whole.
    assertThat(BpePreTokenizer.forName("gpt-4o").split("2024")).containsExactly("202", "4");
    assertThat(BpePreTokenizer.forName("gpt-4o").split("don't")).containsExactly("don't");
    assertThat(BpePreTokenizer.forName("gpt-4o").split("2024"))
        .isNotEqualTo(BpePreTokenizer.forName("tekken").split("2024"));
  }

  @Test
  void singleDigitSplitsAreTwoExpressionsAppliedInSequence() {
    // The pre-existing tests for this group used "2024 code::foo", which is one of the few strings
    // where joining the two expressions with | happens to agree. It does not agree in general.
    // Expectations produced by running llama.cpp's expression list through an independent engine.
    for (String name :
        new String[] {
          "smollm",
          "starcoder",
          "refact",
          "command-r",
          "codeshell",
          "exaone",
          "minerva-7b",
          "mellum2"
        }) {
      assertThat(BpePreTokenizer.forName(name).split("costs 100 dollars"))
          .describedAs(
              "%s: the bare \\p{N} must isolate digits before ?\\p{N}+ can swallow them", name)
          .containsExactly("costs", " ", "1", "0", "0", " dollars");
      assertThat(BpePreTokenizer.forName(name).split("abc 42"))
          .containsExactly("abc", " ", "4", "2");
    }
  }

  @Test
  void deepSeekCoderAppliesItsFiveExpressionsInOrder() {
    // The letter run comes before the CJK/Hangul range, so a space-plus-syllables piece is taken
    // whole and then subdivided. One alternation cannot do the second step.
    assertThat(BpePreTokenizer.forName("deepseek-coder").split("\ud55c\uae00 \ud14c\uc2a4\uc6b0"))
        .containsExactly("\ud55c\uae00", " ", "\ud14c\uc2a4\uc6b0");
    assertThat(
            BpePreTokenizer.forName("deepseek-coder").split("mixed \u4e2d\u6587 and English 123"))
        .containsExactly("mixed", " ", "\u4e2d\u6587", " and", " English", " ", "1", "2", "3");
    assertThat(BpePreTokenizer.forName("deepseek-coder").split("   leading"))
        .describedAs("the letter run keeps one leading space; the rest is a gap")
        .containsExactly("  ", " leading");
  }

  @Test
  void everyPieceConcatenatesBackToTheInput() {
    // Sequential subdivision keeps unmatched text as its own piece, which replaces the old
    // requirement that one expression tile the whole input. The invariant that survives is this
    // one.
    String[] texts = {
      "costs 100 dollars",
      "  ",
      "\n",
      "\ud55c\uae00 \ud14c\uc2a4\uc6b0",
      "a//b path/to/x",
      "don't stop",
      "   leading"
    };
    for (String name :
        new String[] {
          "smollm",
          "qwen2",
          "qwen35",
          "llama3",
          "dbrx",
          "tekken",
          "gpt-4o",
          "gpt-2",
          "deepseek-coder",
          "hunyuan"
        }) {
      for (String text : texts) {
        assertThat(String.join("", BpePreTokenizer.forName(name).split(text)))
            .describedAs("%s must not lose or duplicate input for %s", name, text)
            .isEqualTo(text);
      }
    }
  }

  @Test
  void deepSeekLlmIsImplementedAndSplitsByItsSixExpressions() {
    // The last pre-tokenizer any catalogue model declared and we did not implement. It needed the
    // sequential multi-expression support, so it could not have been added before that existed.
    BpePreTokenizer pre = BpePreTokenizer.forName("deepseek-llm");
    assertThat(pre.isImplemented()).isTrue();
    assertThat(pre.ignoresMerges())
        .describedAs("the deepseek-llm branch sets clean_spaces but not ignore_merges")
        .isFalse();
    // Oracle-derived, not reasoned: the sixth expression is \\p{N}+ WITH a plus, so digits stay
    // grouped here, unlike the single-digit GPT-2 group whose first expression is a bare \\p{N}.
    assertThat(pre.split("costs 100 dollars")).containsExactly("costs", " ", "100", " dollars");
    assertThat(pre.split("line\n\nbreak")).containsExactly("line", "\n", "\n", "break");
    assertThat(pre.split("\ud55c\uae00 \ud14c\uc2a4\uc6b0 42"))
        .containsExactly("\ud55c\uae00", " ", "\ud14c\uc2a4\uc6b0", " ", "42");
  }

  @Test
  void deepSeekLlmExpressionsAreVerbatimFromTheReference() throws Exception {
    // The long second expression is an explicit multi-script letter list containing SUPPLEMENTARY
    // code points. Generating it with \\uXXXX escapes silently corrupted it -- U+10400 became
    // U+1040
    // followed by a literal '0' -- so this pins the compiled patterns against their own lengths and
    // the code points that exposed the bug.
    java.lang.reflect.Field field = BpePreTokenizer.class.getDeclaredField("DEEPSEEK_LLM_PATTERNS");
    field.setAccessible(true);
    @SuppressWarnings("unchecked")
    java.util.List<java.util.regex.Pattern> patterns =
        (java.util.List<java.util.regex.Pattern>) field.get(null);

    assertThat(patterns).hasSize(6);
    assertThat(patterns.get(0).pattern()).isEqualTo("[\\r\\n]");
    assertThat(patterns.get(3).pattern()).isEqualTo("\\s+$");
    assertThat(patterns.get(5).pattern()).isEqualTo("\\p{N}+");
    String letters = patterns.get(1).pattern();
    // 223 CODE POINTS but 237 UTF-16 chars, because 14 of them are supplementary. That distinction
    // is
    // the bug this test exists for: the generated literal used \\uXXXX for every character, so each
    // supplementary code point became a wrong BMP character plus a stray digit.
    assertThat(letters.codePointCount(0, letters.length())).isEqualTo(223);
    assertThat(letters).hasSize(237);
    assertThat(letters.codePoints().anyMatch(cp -> cp == 0x10400))
        .describedAs("the Deseret range start must survive as one supplementary code point")
        .isTrue();
    assertThat(letters.codePoints().anyMatch(cp -> cp == 0x1E943))
        .describedAs("and the Adlam range end")
        .isTrue();
    assertThat(letters).doesNotContain("\u1040" + "0");
  }

  @Test
  void leavesUnknownPreTokenizerTextWhole() {
    assertThat(BpePreTokenizer.forName("vendor-specific").split("2024 code::foo"))
        .containsExactly("2024 code::foo");
  }
}
