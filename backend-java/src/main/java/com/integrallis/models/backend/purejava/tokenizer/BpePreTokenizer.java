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

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Selects and applies the byte-level BPE pre-tokenization declared by GGUF metadata.
 *
 * <p>Every pattern compiles with {@link Pattern#UNICODE_CHARACTER_CLASS} because the published Rust
 * regexes treat {@code \s} as Unicode White_Space: a no-break space is whitespace, not punctuation,
 * and llama.cpp agrees.
 */
final class BpePreTokenizer {

  private static final Pattern LLAMA3_PATTERN =
      Pattern.compile(
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
              + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
              + "|\\p{N}{1,3}"
              + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+",
          Pattern.UNICODE_CHARACTER_CLASS);
  private static final Pattern QWEN2_PATTERN =
      Pattern.compile(
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
              + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
              + "|\\p{N}"
              + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+",
          Pattern.UNICODE_CHARACTER_CLASS);
  private static final Pattern QWEN35_PATTERN =
      Pattern.compile(
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
              + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+"
              + "|\\p{N}"
              + "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+",
          Pattern.UNICODE_CHARACTER_CLASS);
  private static final Pattern GPT_OSS_PATTERN =
      Pattern.compile(
          "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
              + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
              + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
              + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
              + "|\\p{N}{1,3}"
              + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+",
          Pattern.UNICODE_CHARACTER_CLASS);

  /**
   * Mistral's tekken, transcribed from the {@code config.pattern} of the model's own {@code
   * tekken.json} (version v7, read from Mistral-Small-24B-Instruct-2501) and independently
   * confirmed against llama.cpp's {@code LLAMA_VOCAB_PRE_TYPE_TEKKEN}, whose comment quotes the
   * identical expression as the "original regex from tokenizer.json".
   *
   * <p>This is the o200k shape that {@link #GPT_OSS_PATTERN} uses, differing in exactly two places:
   * there are <b>no contraction groups</b>, and digits are taken <b>one at a time</b> ({@code
   * \p{N}}, not {@code \p{N}{1,3}}). Both change the token stream on ordinary prose, so borrowing
   * the gpt-oss pattern would have been a silent tokenization difference rather than an error:
   * {@code don't stop} is {@code don} {@code 't} {@code stop} here against gpt-oss's {@code don't}
   * {@code stop}, and {@code 2024} is four pieces against gpt-oss's {@code 202} {@code 4}.
   *
   * <p>Note where the apostrophe goes, because it is not obvious and was got wrong once by
   * reasoning about the expression instead of running it: {@code 't} stays together even with no
   * contraction group, because the first alternative's optional {@code [^\r\n\p{L}\p{N}]?} prefix
   * absorbs the apostrophe ahead of the lowercase run. The splits asserted in the tests were
   * produced by running this exact pattern string, taken from {@code tekken.json}, through an
   * independent regex engine -- not derived by hand.
   *
   * <p>llama.cpp's <i>active</i> expression rewrites {@code \p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}} into
   * {@code ((?=[\p{L}])([^a-z]))} lookahead form because {@code std::regex} lacks those classes.
   * That rewrite is a limitation workaround and is not equivalent for non-ASCII scripts -- "a
   * letter that is not a lowercase ASCII letter" is not "an uppercase, titlecase, modifier, other
   * or mark character". Java's {@link Pattern} supports the real classes, so the original form is
   * used.
   */
  private static final Pattern TEKKEN_PATTERN =
      Pattern.compile(
          "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
              + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
              + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
              + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
              + "|\\p{N}"
              + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+",
          Pattern.UNICODE_CHARACTER_CLASS);

  private static final Pattern GPT2_PATTERN =
      Pattern.compile(
          "'s|'t|'re|'ve|'m|'ll|'d"
              + "| ?\\p{L}+"
              + "| ?\\p{N}+"
              + "| ?[^\\s\\p{L}\\p{N}]+"
              + "|\\s+(?!\\S)"
              + "|\\s+",
          Pattern.UNICODE_CHARACTER_CLASS);

  /**
   * The single-digit GPT-2 split, as <b>two expressions applied in sequence</b>.
   *
   * <p>Transcribed from llama.cpp's {@code LLAMA_VOCAB_PRE_TYPE_SMOLLM} case, which lists {@code
   * \p{N}} first and the word/number/punctuation alternation second. Joining them with {@code |}
   * into one expression -- which is what this was until it was checked against the reference -- is
   * a different tokenizer: {@code ?\p{N}+} in the second expression swallows a leading space and a
   * whole digit run before the bare {@code \p{N}} of the first can isolate one digit, so {@code
   * "costs 100 dollars"} came out as {@code costs}/{@code 100}/{@code dollars} instead of {@code
   * costs}/{@code }/{@code 1}/{@code 0}/{@code 0}/{@code dollars}.
   *
   * <p>No model in the current catalogue declares one of these names, so nothing measured so far is
   * affected; it was found by verifying the mapping table rather than by a failure. Note also that
   * the second expression has no trailing {@code |\s+}: unmatched text is kept as its own piece, so
   * the reference does not need one.
   */
  private static final List<Pattern> SINGLE_DIGIT_GPT2_PATTERNS =
      List.of(
          Pattern.compile("\\p{N}", Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile(
              "'s|'t|'re|'ve|'m|'ll|'d"
                  + "| ?\\p{L}+"
                  + "| ?\\p{N}+"
                  + "| ?[^\\s\\p{L}\\p{N}]+"
                  + "|\\s+(?!\\S)",
              Pattern.UNICODE_CHARACTER_CLASS));

  /**
   * The DeepSeek-Coder split, as <b>five expressions applied in sequence</b>.
   *
   * <p>Transcribed in order from llama.cpp's {@code LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER} case. The
   * order is the whole point and cannot be expressed as one alternation: the letter run comes
   * before the CJK/Hangul range, so a piece like {@code " \uc5ec\uae30"} is first taken whole by
   * {@code \s?\p{L}+} and then <i>subdivided</i> by the range expression into the space and the
   * syllables. Joined with {@code |} -- which is what this was -- the second split can never
   * happen, and the space stays glued to the word.
   *
   * <p>Measured: joined against sequential, 4 of 32 strings differed, every one of them CJK, Hangul
   * or leading whitespace. No catalogue model declares this name, so nothing measured so far is
   * affected. Note there is no trailing {@code \s+} alternative: unmatched text is kept as its own
   * piece, so the reference does not need one.
   */
  private static final List<Pattern> DEEPSEEK_CODER_PATTERNS =
      List.of(
          Pattern.compile("[\\r\\n]", Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile("\\s?\\p{L}+", Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile("\\s?\\p{P}+", Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile("[一-龥ࠀ-一가-퟿]+", Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile("\\p{N}", Pattern.UNICODE_CHARACTER_CLASS));

  private static final Set<String> LLAMA3_IGNORE_MERGES_NAMES =
      Set.of(
          "llama3",
          "llama-v3",
          "llama-bpe",
          "falcon3",
          "falcon-h1",
          "pixtral",
          "midm-2.0",
          "lfm2",
          "jina-v5-nano",
          "minicpm5");

  /**
   * Llama-3 word boundaries with ordinary merge ranking. Granite 4.x GGUFs declare {@code dbrx};
   * their published tokenizer.json carries exactly this split regex with {@code ignore_merges}
   * false, and an unmapped name would otherwise skip pre-tokenization entirely.
   */
  private static final Set<String> LLAMA3_KEEP_MERGES_NAMES = Set.of("smaug-bpe", "dbrx");

  // "hunyuan" is here on evidence, not resemblance. Its own tokenizer.json publishes
  //   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
  // ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  // which is QWEN2_PATTERN alternative for alternative -- (?i:'s|...) and (?:'[sS]|...) are the
  // same
  // case-insensitive contraction set, and the number branch is the single-digit \p{N}, not the
  // \p{N}{1,3} of LLAMA3_PATTERN. A GGUF does not carry the regex, so the authoritative source is
  // the
  // model's tokenizer.json; recalling a tokenizer regex is how tokenization goes quietly wrong.
  private static final Set<String> QWEN2_NAMES =
      Set.of("qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "megrez", "hunyuan");
  private static final Set<String> GPT2_NAMES =
      Set.of(
          "gpt-2",
          "phi-2",
          "jina-es",
          "jina-de",
          "gigachat",
          "jina-v2-es",
          "jina-v2-de",
          "a.x-4.0",
          "mellum",
          "modern-bert",
          "exaone4");
  private static final Set<String> SINGLE_DIGIT_GPT2_NAMES =
      Set.of(
          "starcoder",
          "refact",
          "command-r",
          "smollm",
          "codeshell",
          "exaone",
          "minerva-7b",
          "mellum2");

  /**
   * Names selecting the o200k split, which {@link #GPT_OSS_PATTERN} already is.
   *
   * <p>{@code gpt-4o} is here on evidence: llama.cpp's own {@code LLAMA_VOCAB_PRE_TYPE_GPT4O} case
   * quotes the "original regex from tokenizer.json", and it is our {@code GPT_OSS_PATTERN}
   * alternative for alternative -- unsurprising, since gpt-oss's o200k_harmony inherits
   * o200k_base's split from GPT-4o. The same dispatch branch selects that type for {@code llama4},
   * {@code kanana2} and {@code talkie} with identical flags, and sets {@code clean_spaces = false}
   * but <b>not</b> {@code ignore_merges}, which is why this group keeps merges like {@code
   * gpt-oss}.
   *
   * <p>Found by surveying the declared pre-tokenizer of all 80 catalogue GGUFs: Phi-4-mini declares
   * {@code gpt-4o} on a byte-level BPE vocabulary, and phi3 became runnable earlier in this
   * campaign, so it would have been scheduled and silently tokenized with no word-boundary
   * splitting at all.
   */
  private static final Set<String> O200K_NAMES =
      Set.of("gpt-oss", "gpt-4o", "llama4", "kanana2", "talkie");

  /**
   * The DeepSeek-LLM split: <b>six expressions applied in sequence</b>.
   *
   * <p>Transcribed in order from llama.cpp's {@code LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM} case. The
   * long second expression is an explicit list of Latin, Greek, Cyrillic, Armenian, Georgian,
   * Cherokee and other letter ranges rather than {@code \\p{L}}, and it is reproduced verbatim --
   * the literal was generated from the reference source rather than retyped, and a test asserts
   * each compiled pattern still equals the reference string exactly.
   *
   * <p>This is the last pre-tokenizer any model in the catalogue declared and we did not implement.
   * It needed the sequential multi-expression support that {@link #split} gained for the
   * single-digit GPT-2 and DeepSeek-Coder corrections; before that there was no way to express it,
   * and the fallback was no word-boundary splitting at all.
   *
   * <p>Order matters here for a reason worth recording: expression four is {@code \\s+$}, and
   * Java's {@code $} also matches before a final line terminator where {@code std::regex} matches
   * only the very end. Expression one has already split every {@code \\r} and {@code \\n} into its
   * own piece by then, so no piece reaching expression four can contain one, and the difference
   * cannot arise.
   */
  private static final List<Pattern> DEEPSEEK_LLM_PATTERNS =
      List.of(
          Pattern.compile("[\\r\\n]", Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile(
              "\\s?[A-Za-z\u00b5\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u01ba\u01bc-\u01bf"
                  + "\u01c4-\u0293\u0295-\u02af\u0370-\u0373\u0376\u0377\u037b-\u037d"
                  + "\u037f\u0386\u0388-\u038a\u038c\u038e-\u03a1\u03a3-\u03f5\u03f7-"
                  + "\u0481\u048a-\u052f\u0531-\u0556\u10a0-\u10c5\u13a0-\u13f5\u13f8"
                  + "-\u13fd\u1c90-\u1cba\u1cbd-\u1cbf\u1d00-\u1d2b\u1d6b-\u1d77\u1d79"
                  + "-\u1d9a\u1e00-\u1f15\u1f18-\u1f1d\u1f20-\u1f45\u1f48-\u1f4d\u1f50"
                  + "-\u1f57\u1f59\u1f5b\u1f5d\u1f5f-\u1f7d\u1f80-\u1fb4\u1fb6-\u1fbc"
                  + "\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fcc\u1fd0-\u1fd3\u1fd6-\u1fdb\u1fe0"
                  + "-\u1fec\u1ff2-\u1ff4\u1ff6-\u1ffc\u2102\u2107\u210a-\u2113\u2115"
                  + "\u2119-\u211d\u2124\u2126\u2128\u212a-\u212d\u212f-\u2134\u2139\u213c"
                  + "-\u213f\u2145-\u2149\u214e\u2183\u2184\u2c00-\u2c7b\u2c7e-\u2ce4"
                  + "\u2ceb-\u2cee\u2cf2\u2cf3\ua640-\ua66d\ua680-\ua69b\ua722-\ua76f"
                  + "\ua771-\ua787\ua78b-\ua78e\uab70-\uabbf\ufb00-\ufb06\ufb13-\ufb17"
                  + "\uff21-\uff3a\uff41-\uff5a\ud801\udc00-\ud801\udc4f\ud801\udcb0-"
                  + "\ud801\udcd3\ud801\udcd8-\ud801\udcfb\ud803\udc80-\ud803\udcb2\ud803"
                  + "\udcc0-\ud803\udcf2\ud806\udca0-\ud806\udcdf\ud83a\udd00-\ud83a\udd43"
                  + "]+",
              Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile(
              "\\s?[!-/:-~\uff01-\uff0f\uff1a-\uff5e\u2018-\u201f\u3000-\u3002]+",
              Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile("\\s+$", Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile(
              "[\u4e00-\u9fa5\u0800-\u4e00\uac00-\ud7ff]+", Pattern.UNICODE_CHARACTER_CLASS),
          Pattern.compile("\\p{N}+", Pattern.UNICODE_CHARACTER_CLASS));

  /** No pre-splitting: no expressions at all, so every text stays whole. */
  private static final BpePreTokenizer NONE = new BpePreTokenizer(List.<Pattern>of(), false);

  private static final java.util.Set<String> UNRECOGNISED =
      java.util.concurrent.ConcurrentHashMap.newKeySet();
  private static final BpePreTokenizer LLAMA3 = new BpePreTokenizer(LLAMA3_PATTERN, true);
  private static final BpePreTokenizer SMAUG = new BpePreTokenizer(LLAMA3_PATTERN, false);
  private static final BpePreTokenizer QWEN2 = new BpePreTokenizer(QWEN2_PATTERN, false);
  private static final BpePreTokenizer QWEN35 = new BpePreTokenizer(QWEN35_PATTERN, false);
  private static final BpePreTokenizer GPT_OSS = new BpePreTokenizer(GPT_OSS_PATTERN, false);
  // ignoreMerges is true, not a guess: llama.cpp's tokenizer_pre == "tekken" branch sets
  // ignore_merges = true alongside add_bos = true. Our flag has the same meaning -- try the whole
  // pre-tokenized piece as one vocabulary entry before running the merge loop.
  private static final BpePreTokenizer TEKKEN = new BpePreTokenizer(TEKKEN_PATTERN, true);
  private static final BpePreTokenizer GPT2 = new BpePreTokenizer(GPT2_PATTERN, false);
  private static final BpePreTokenizer SINGLE_DIGIT_GPT2 =
      new BpePreTokenizer(SINGLE_DIGIT_GPT2_PATTERNS, false);
  private static final BpePreTokenizer DEEPSEEK_CODER =
      new BpePreTokenizer(DEEPSEEK_CODER_PATTERNS, false);
  private static final BpePreTokenizer DEEPSEEK_LLM =
      new BpePreTokenizer(DEEPSEEK_LLM_PATTERNS, false);

  private final List<Pattern> patterns;
  private final boolean ignoreMerges;
  private final String declaredName;
  private final boolean implemented;

  private BpePreTokenizer(Pattern pattern, boolean ignoreMerges) {
    this(List.of(pattern), ignoreMerges, "", true);
  }

  private BpePreTokenizer(List<Pattern> patterns, boolean ignoreMerges) {
    this(patterns, ignoreMerges, "", true);
  }

  private BpePreTokenizer(
      List<Pattern> patterns, boolean ignoreMerges, String declaredName, boolean implemented) {
    this.patterns = List.copyOf(patterns);
    this.ignoreMerges = ignoreMerges;
    this.declaredName = declaredName;
    this.implemented = implemented;
  }

  /** A copy of this pre-tokenizer tagged with the metadata name that selected it. */
  private BpePreTokenizer named(String name, boolean nameImplemented) {
    return new BpePreTokenizer(patterns, ignoreMerges, name, nameImplemented);
  }

  /**
   * The {@code tokenizer.ggml.pre} value this was selected by, or empty when the model declared
   * none.
   */
  String declaredName() {
    return declaredName;
  }

  /**
   * Whether the declared pre-tokenizer is one we implement.
   *
   * <p>True when the model declared none at all -- every SentencePiece model, and byte-level models
   * that genuinely pre-split nothing -- and true when the declared name mapped to a pattern. False
   * only for a name we fell back to no-splitting on, which is a silent tokenization difference.
   *
   * <p>This exists because {@link #unrecognisedNames()} could not answer the question in
   * production: it is public on a package-private class, so the only callers able to reach it were
   * tests in this package, and the fallback went unreported for every real model run.
   */
  boolean isImplemented() {
    return implemented;
  }

  static BpePreTokenizer forName(String name) {
    if (LLAMA3_IGNORE_MERGES_NAMES.contains(name)) {
      return LLAMA3.named(name, true);
    }
    if (LLAMA3_KEEP_MERGES_NAMES.contains(name)) {
      return SMAUG.named(name, true);
    }
    if (QWEN2_NAMES.contains(name)) {
      return QWEN2.named(name, true);
    }
    if ("qwen35".equals(name)) {
      return QWEN35.named(name, true);
    }
    if (O200K_NAMES.contains(name)) {
      return GPT_OSS.named(name, true);
    }
    if ("tekken".equals(name)) {
      return TEKKEN.named(name, true);
    }
    if (GPT2_NAMES.contains(name)) {
      return GPT2.named(name, true);
    }
    if (SINGLE_DIGIT_GPT2_NAMES.contains(name)) {
      return SINGLE_DIGIT_GPT2.named(name, true);
    }
    if ("deepseek-coder".equals(name)) {
      return DEEPSEEK_CODER.named(name, true);
    }
    if ("deepseek-llm".equals(name)) {
      return DEEPSEEK_LLM.named(name, true);
    }
    // An unrecognised NAME falls back to no pre-splitting. That is correct for the models that
    // genuinely have none -- an absent key, and every SentencePiece model, which never consults
    // this
    // -- but for a byte-level BPE model that does declare one it is a silent tokenization
    // difference
    // that shows up as slightly worse output, never as an error.
    //
    // Coverage is wide (38 names above, including dbrx for Granite, deepseek-r1-qwen and smollm),
    // so
    // this is not thrown: doing so would reject models that ship today and pass their pinned-token
    // oracles. Instead the name is recorded, so "this model declares a pre-tokenizer we do not
    // implement" is answerable rather than invisible. The authoritative regex for a missing one is
    // in
    // the model's own tokenizer.json on HuggingFace -- the GGUF does not carry it -- so a new entry
    // should be transcribed from there, never recalled.
    if (!name.isEmpty()) {
      UNRECOGNISED.add(name);
    }
    return NONE.named(name, name.isEmpty());
  }

  /**
   * Pre-tokenizer names seen but not implemented, for diagnostics.
   *
   * <p>Exposed so a caller can surface "this model declares a pre-tokenizer we do not implement"
   * instead of discovering it as unexplained quality loss.
   */
  public static java.util.Set<String> unrecognisedNames() {
    return java.util.Set.copyOf(UNRECOGNISED);
  }

  boolean ignoresMerges() {
    return ignoreMerges;
  }

  /**
   * Splits text into pre-tokenized pieces.
   *
   * <p>Several expressions are applied <b>in sequence, each subdividing the pieces the previous one
   * produced</b>, and text no expression matches is kept as its own piece. That is exactly what
   * llama.cpp's {@code unicode_regex_split} does: it seeds {@code bpe_offsets} with the whole text
   * and hands the current offsets to each expression in turn.
   *
   * <p>Sequential application is not the same as joining the expressions with {@code |}, and the
   * difference is not academic. The single-digit GPT-2 split is two expressions, {@code \p{N}} then
   * the word/number/punctuation alternation. On {@code "costs 100 dollars"} the sequential form
   * gives {@code costs}, {@code }, {@code 1}, {@code 0}, {@code 0}, {@code dollars}; the joined
   * form gives {@code costs}, {@code 100}, {@code dollars}, because {@code ?\p{N}+} consumes the
   * space and all three digits before the bare {@code \p{N}} ever gets a chance. Four of six
   * ordinary test strings differ.
   */
  List<String> split(String text) {
    if (patterns.isEmpty() || text.isEmpty()) {
      return List.of(text);
    }

    List<String> pieces = new ArrayList<>();
    pieces.add(text);
    for (Pattern pattern : patterns) {
      List<String> next = new ArrayList<>(pieces.size() * 2);
      for (String piece : pieces) {
        subdivide(pattern, piece, next);
      }
      pieces = next;
    }

    // The pieces must reconstruct the input exactly. This replaces the old requirement that one
    // expression tile the whole text, which cannot hold once unmatched gaps are legitimate pieces,
    // and it catches the same class of mistake -- a pattern that loses or duplicates input.
    int total = 0;
    for (String piece : pieces) {
      total += piece.length();
    }
    if (total != text.length()) {
      throw new IllegalStateException(
          "pre-tokenization lost input: " + total + " characters from " + text.length());
    }
    return pieces;
  }

  /** Splits one piece by one expression, keeping the gaps between matches. */
  private static void subdivide(Pattern pattern, String piece, List<String> out) {
    Matcher matcher = pattern.matcher(piece);
    int at = 0;
    while (matcher.find()) {
      if (matcher.end() == matcher.start()) {
        continue;
      }
      if (matcher.start() > at) {
        out.add(piece.substring(at, matcher.start()));
      }
      out.add(matcher.group());
      at = matcher.end();
    }
    if (at < piece.length()) {
      out.add(piece.substring(at));
    }
  }
}
