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

/** Selects and applies the byte-level BPE pre-tokenization declared by GGUF metadata. */
final class BpePreTokenizer {

  private static final Pattern LLAMA3_PATTERN =
      Pattern.compile(
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
              + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
              + "|\\p{N}{1,3}"
              + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+");
  private static final Pattern QWEN2_PATTERN =
      Pattern.compile(
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
              + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
              + "|\\p{N}"
              + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+");
  private static final Pattern QWEN35_PATTERN =
      Pattern.compile(
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
              + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+"
              + "|\\p{N}"
              + "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+");
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
              + "|\\s+");
  private static final Pattern GPT2_PATTERN =
      Pattern.compile(
          "'s|'t|'re|'ve|'m|'ll|'d"
              + "| ?\\p{L}+"
              + "| ?\\p{N}+"
              + "| ?[^\\s\\p{L}\\p{N}]+"
              + "|\\s+(?!\\S)"
              + "|\\s+");
  private static final Pattern SINGLE_DIGIT_GPT2_PATTERN =
      Pattern.compile(
          "\\p{N}"
              + "|'s|'t|'re|'ve|'m|'ll|'d"
              + "| ?\\p{L}+"
              + "| ?\\p{N}+"
              + "| ?[^\\s\\p{L}\\p{N}]+"
              + "|\\s+(?!\\S)"
              + "|\\s+");
  private static final Pattern DEEPSEEK_CODER_PATTERN =
      Pattern.compile(
          "[\\r\\n]" + "|\\s?\\p{L}+" + "|\\s?\\p{P}+" + "|[一-龥ࠀ-一가-퟿]+" + "|\\p{N}" + "|\\s+");

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
  private static final Set<String> QWEN2_NAMES =
      Set.of("qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "megrez");
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

  private static final BpePreTokenizer NONE = new BpePreTokenizer(null, false);
  private static final BpePreTokenizer LLAMA3 = new BpePreTokenizer(LLAMA3_PATTERN, true);
  private static final BpePreTokenizer SMAUG = new BpePreTokenizer(LLAMA3_PATTERN, false);
  private static final BpePreTokenizer QWEN2 = new BpePreTokenizer(QWEN2_PATTERN, false);
  private static final BpePreTokenizer QWEN35 = new BpePreTokenizer(QWEN35_PATTERN, false);
  private static final BpePreTokenizer GPT_OSS = new BpePreTokenizer(GPT_OSS_PATTERN, false);
  private static final BpePreTokenizer GPT2 = new BpePreTokenizer(GPT2_PATTERN, false);
  private static final BpePreTokenizer SINGLE_DIGIT_GPT2 =
      new BpePreTokenizer(SINGLE_DIGIT_GPT2_PATTERN, false);
  private static final BpePreTokenizer DEEPSEEK_CODER =
      new BpePreTokenizer(DEEPSEEK_CODER_PATTERN, false);

  private final Pattern pattern;
  private final boolean ignoreMerges;

  private BpePreTokenizer(Pattern pattern, boolean ignoreMerges) {
    this.pattern = pattern;
    this.ignoreMerges = ignoreMerges;
  }

  static BpePreTokenizer forName(String name) {
    if (LLAMA3_IGNORE_MERGES_NAMES.contains(name)) {
      return LLAMA3;
    }
    if ("smaug-bpe".equals(name)) {
      return SMAUG;
    }
    if (QWEN2_NAMES.contains(name)) {
      return QWEN2;
    }
    if ("qwen35".equals(name)) {
      return QWEN35;
    }
    if ("gpt-oss".equals(name)) {
      return GPT_OSS;
    }
    if (GPT2_NAMES.contains(name)) {
      return GPT2;
    }
    if (SINGLE_DIGIT_GPT2_NAMES.contains(name)) {
      return SINGLE_DIGIT_GPT2;
    }
    if ("deepseek-coder".equals(name)) {
      return DEEPSEEK_CODER;
    }
    return NONE;
  }

  boolean ignoresMerges() {
    return ignoreMerges;
  }

  List<String> split(String text) {
    if (pattern == null || text.isEmpty()) {
      return List.of(text);
    }

    List<String> pieces = new ArrayList<>();
    Matcher matcher = pattern.matcher(text);
    int matchedThrough = 0;
    while (matcher.find()) {
      if (matcher.start() != matchedThrough) {
        throw unmatchedInput(matchedThrough);
      }
      pieces.add(matcher.group());
      matchedThrough = matcher.end();
    }
    if (matchedThrough != text.length()) {
      throw unmatchedInput(matchedThrough);
    }
    return pieces;
  }

  private static IllegalArgumentException unmatchedInput(int index) {
    return new IllegalArgumentException(
        "Tokenizer pre-pattern did not match input at index " + index);
  }
}
