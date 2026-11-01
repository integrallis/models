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
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * English text to misaki-spelled IPA: a port of misaki's {@code en.G2P} pipeline over a {@link
 * MisakiLexicon}, with {@link EnglishTokenizer} in place of spaCy and no out-of-vocabulary
 * fallback. Words the lexicon cannot resolve become {@code unk} (the publisher of Paradee uses the
 * empty string) and are reported in {@link Result#unknownWords()}.
 */
public final class MisakiEnglishPhonemizer {

  /**
   * The phoneme string plus what the pipeline could not resolve.
   *
   * @param phonemes the phoneme string, with the unknown marker where a word was not resolved
   * @param unknownWords the words that had no phonemes, in text order
   * @param wordCount the number of words the pipeline tried to resolve
   */
  public record Result(String phonemes, List<String> unknownWords, int wordCount) {
    /** Defensive copy of the word list. */
    public Result {
      unknownWords = List.copyOf(unknownWords);
    }
  }

  private static final Pattern SUBTOKEN =
      Pattern.compile(
          "^['‘’]+|\\p{Lu}(?=\\p{Lu}\\p{Ll})|(?:^-)?(?:\\d?[,.]?\\d)+|[-_]+|['‘’]{2,}|\\p{L}*?(?:['‘’]\\p{L})*?\\p{Ll}(?=\\p{Lu})|\\p{L}+(?:['‘’]\\p{L})*|[^-_\\p{L}'‘’\\d]|['‘’]+$");
  private static final Pattern LINK = Pattern.compile("\\[([^\\]]+)\\]\\(([^\\)]*)\\)");
  private static final Set<String> PUNCT_TAGS =
      Set.of(".", ",", "-LRB-", "-RRB-", "\u0060\u0060", "\"\"", "''", ":", "$", "#", "NFP");
  private static final Map<String, String> PUNCT_TAG_PHONEMES =
      Map.of("-LRB-", "(", "-RRB-", ")", "\u0060\u0060", "“", "\"\"", "”", "''", "”");

  private final MisakiLexicon lexicon;
  private final String unk;

  /** Creates a phonemizer; {@code unk} replaces unresolved words and may be empty. */
  public MisakiEnglishPhonemizer(MisakiLexicon lexicon, String unk) {
    this.lexicon = Objects.requireNonNull(lexicon, "lexicon");
    this.unk = Objects.requireNonNull(unk, "unk");
  }

  /** Whether this phonemizer spells British English. */
  public boolean british() {
    return lexicon.british();
  }

  /** The phoneme string for {@code text}. */
  public String phonemize(String text) {
    return convert(text).phonemes();
  }

  /** Phonemizes {@code text} and reports the unresolved words. */
  public Result convert(String text) {
    List<MToken> tokens = EnglishTokenizer.tokenize(preprocess(text));
    tokens = foldLeft(tokens);
    List<Word> words = retokenize(tokens);
    MisakiLexicon.Context ctx = MisakiLexicon.Context.INITIAL;
    int wordCount = 0;
    for (int i = words.size() - 1; i >= 0; i--) {
      Word w = words.get(i);
      if (w.single != null) {
        wordCount++;
        if (w.single.phonemes == null) {
          MisakiLexicon.Result found = lexicon.phonemize(w.single, ctx);
          w.single.phonemes = found.phonemes();
          w.single.rating = found.rating();
        }
        ctx = tokenContext(ctx, w.single.phonemes, w.single);
        continue;
      }
      wordCount++;
      ctx = resolveGroup(w.group, ctx);
    }
    List<String> unknown = new ArrayList<>();
    List<MToken> merged = new ArrayList<>();
    for (Word w : words) {
      if (w.single != null) {
        if (w.single.phonemes == null && w.single.alias == null) {
          unknown.add(w.single.text);
        }
        merged.add(w.single);
      } else {
        for (MToken tk : w.group) {
          if (tk.phonemes == null) {
            unknown.add(tk.text);
          }
        }
        merged.add(mergeTokens(w.group, unk));
      }
    }
    StringBuilder result = new StringBuilder();
    for (MToken tk : merged) {
      String ps = tk.phonemes == null ? unk : tk.phonemes.replace('ɾ', 'T').replace('ʔ', 't');
      result.append(ps).append(tk.whitespace);
    }
    return new Result(result.toString(), List.copyOf(unknown), wordCount);
  }

  /** misaki's {@code preprocess} without feature links: {@code [text](anything)} keeps the text. */
  static String preprocess(String text) {
    String stripped = text.stripLeading();
    Matcher m = LINK.matcher(stripped);
    StringBuilder out = new StringBuilder();
    int last = 0;
    while (m.find()) {
      out.append(stripped, last, m.start()).append(m.group(1));
      last = m.end();
    }
    out.append(stripped.substring(last));
    return out.toString();
  }

  private List<MToken> foldLeft(List<MToken> tokens) {
    List<MToken> result = new ArrayList<>();
    for (MToken tk : tokens) {
      if (!result.isEmpty() && !tk.isHead) {
        MToken previous = result.remove(result.size() - 1);
        result.add(mergeTokens(List.of(previous, tk), unk));
      } else {
        result.add(tk);
      }
    }
    return result;
  }

  /** A pipeline item: one token, or a run of sub-tokens that form one written word. */
  static final class Word {
    final MToken single;
    final List<MToken> group;

    Word(MToken single) {
      this.single = single;
      this.group = null;
    }

    Word(List<MToken> group) {
      this.single = null;
      this.group = group;
    }
  }

  static List<String> subtokenize(String word) {
    List<String> parts = new ArrayList<>();
    Matcher m = SUBTOKEN.matcher(word);
    while (m.find()) {
      if (!m.group().isEmpty()) {
        parts.add(m.group());
      }
    }
    return parts;
  }

  private static List<Word> retokenize(List<MToken> tokens) {
    List<Object> words = new ArrayList<>();
    String currency = null;
    for (int i = 0; i < tokens.size(); i++) {
      MToken token = tokens.get(i);
      List<MToken> tks = new ArrayList<>();
      if (token.alias == null && token.phonemes == null) {
        for (String part : subtokenize(token.text)) {
          MToken sub = new MToken(part, token.tag, "");
          sub.numFlags = token.numFlags;
          sub.stress = token.stress;
          tks.add(sub);
        }
        if (tks.isEmpty()) {
          tks.add(new MToken(token.text, token.tag, ""));
        }
      } else {
        tks.add(token);
      }
      tks.get(tks.size() - 1).whitespace = token.whitespace;
      for (int j = 0; j < tks.size(); j++) {
        MToken tk = tks.get(j);
        if (tk.alias != null || tk.phonemes != null) {
          // already resolved
        } else if ("$".equals(tk.tag) && MisakiLexicon.CURRENCIES.containsKey(tk.text)) {
          currency = tk.text;
          tk.phonemes = "";
          tk.rating = 4;
        } else if (":".equals(tk.tag) && (tk.text.equals("-") || tk.text.equals("–"))) {
          tk.phonemes = "—";
          tk.rating = 3;
        } else if (PUNCT_TAGS.contains(tk.tag) && !allAsciiLetters(tk.text)) {
          String mapped = PUNCT_TAG_PHONEMES.get(tk.tag);
          if (mapped == null) {
            StringBuilder kept = new StringBuilder();
            for (char c : tk.text.toCharArray()) {
              if (MisakiStress.PUNCTS.contains(c)) {
                kept.append(c);
              }
            }
            mapped = kept.toString();
          }
          tk.phonemes = mapped;
          tk.rating = 4;
        } else if (currency != null) {
          if (!"CD".equals(tk.tag)) {
            currency = null;
          } else if (j + 1 == tks.size()
              && (i + 1 == tokens.size() || !"CD".equals(tokens.get(i + 1).tag))) {
            tk.currency = currency;
          }
        } else if (j > 0 && j < tks.size() - 1 && tk.text.equals("2")) {
          String before = tks.get(j - 1).text;
          String after = tks.get(j + 1).text;
          if (Character.isLetter(before.charAt(before.length() - 1))
              && Character.isLetter(after.charAt(0))) {
            tk.alias = "to";
          }
        }
        if (tk.alias != null || tk.phonemes != null) {
          words.add(tk);
        } else if (!words.isEmpty()
            && words.get(words.size() - 1) instanceof List<?> last
            && ((MToken) last.get(last.size() - 1)).whitespace.isEmpty()) {
          tk.isHead = false;
          @SuppressWarnings("unchecked")
          List<MToken> group = (List<MToken>) last;
          group.add(tk);
        } else if (!tk.whitespace.isEmpty()) {
          words.add(tk);
        } else {
          List<MToken> group = new ArrayList<>();
          group.add(tk);
          words.add(group);
        }
      }
    }
    List<Word> result = new ArrayList<>();
    for (Object w : words) {
      if (w instanceof MToken tk) {
        result.add(new Word(tk));
      } else {
        @SuppressWarnings("unchecked")
        List<MToken> group = (List<MToken>) w;
        result.add(group.size() == 1 ? new Word(group.get(0)) : new Word(group));
      }
    }
    return result;
  }

  private static boolean allAsciiLetters(String text) {
    for (char c : text.toCharArray()) {
      char lower = Character.toLowerCase(c);
      if (lower < 'a' || lower > 'z') {
        return false;
      }
    }
    return true;
  }

  private MisakiLexicon.Context resolveGroup(List<MToken> w, MisakiLexicon.Context ctx) {
    int left = 0;
    int right = w.size();
    while (left < right) {
      boolean anyResolved = false;
      for (int k = left; k < right; k++) {
        if (w.get(k).alias != null || w.get(k).phonemes != null) {
          anyResolved = true;
          break;
        }
      }
      MToken merged = anyResolved ? null : mergeTokens(w.subList(left, right), null);
      MisakiLexicon.Result found =
          merged == null ? MisakiLexicon.Result.NONE : lexicon.phonemize(merged, ctx);
      if (found.found()) {
        w.get(left).phonemes = found.phonemes();
        w.get(left).rating = found.rating();
        for (int k = left + 1; k < right; k++) {
          w.get(k).phonemes = "";
          w.get(k).rating = found.rating();
        }
        ctx = tokenContext(ctx, found.phonemes(), merged);
        right = left;
        left = 0;
      } else if (left + 1 < right) {
        left++;
      } else {
        right--;
        MToken tk = w.get(right);
        if (tk.phonemes == null && allJunk(tk.text)) {
          tk.phonemes = "";
          tk.rating = 3;
        }
        left = 0;
      }
    }
    resolveTokens(w);
    return ctx;
  }

  private static boolean allJunk(String text) {
    for (char c : text.toCharArray()) {
      if (!MisakiStress.SUBTOKEN_JUNKS.contains(c)) {
        return false;
      }
    }
    return true;
  }

  static MisakiLexicon.Context tokenContext(MisakiLexicon.Context ctx, String ps, MToken token) {
    Boolean vowel = ctx.futureVowel();
    if (ps != null && !ps.isEmpty()) {
      for (char c : ps.toCharArray()) {
        boolean isVowel = MisakiStress.VOWELS.contains(c);
        if (isVowel
            || MisakiStress.CONSONANTS.contains(c)
            || MisakiStress.NON_QUOTE_PUNCTS.contains(c)) {
          vowel = MisakiStress.NON_QUOTE_PUNCTS.contains(c) ? null : Boolean.valueOf(isVowel);
          break;
        }
      }
    }
    boolean futureTo =
        token.text.equals("to")
            || token.text.equals("To")
            || (token.text.equals("TO") && ("TO".equals(token.tag) || "IN".equals(token.tag)));
    return new MisakiLexicon.Context(vowel, futureTo);
  }

  /** misaki's {@code resolve_tokens}: compound-word stress and punctuation inside a word group. */
  static void resolveTokens(List<MToken> tokens) {
    StringBuilder textBuilder = new StringBuilder();
    for (int i = 0; i < tokens.size(); i++) {
      textBuilder.append(tokens.get(i).text);
      if (i < tokens.size() - 1) {
        textBuilder.append(tokens.get(i).whitespace);
      }
    }
    String text = textBuilder.toString();
    Set<Integer> classes = new TreeSet<>();
    for (char c : text.toCharArray()) {
      if (!MisakiStress.SUBTOKEN_JUNKS.contains(c)) {
        classes.add(Character.isLetter(c) ? 0 : MisakiLexicon.isDigit(String.valueOf(c)) ? 1 : 2);
      }
    }
    boolean prespace = text.indexOf(' ') >= 0 || text.indexOf('/') >= 0 || classes.size() > 1;
    for (int i = 0; i < tokens.size(); i++) {
      MToken tk = tokens.get(i);
      if (tk.phonemes == null) {
        if (i == tokens.size() - 1
            && tk.text.length() == 1
            && MisakiStress.NON_QUOTE_PUNCTS.contains(tk.text.charAt(0))) {
          tk.phonemes = tk.text;
          tk.rating = 3;
        } else if (allJunk(tk.text)) {
          tk.phonemes = "";
          tk.rating = 3;
        }
      } else if (i > 0) {
        tk.prespace = prespace;
      }
    }
    if (prespace) {
      return;
    }
    List<int[]> indices = new ArrayList<>();
    for (int i = 0; i < tokens.size(); i++) {
      String ps = tokens.get(i).phonemes;
      if (ps != null && !ps.isEmpty()) {
        indices.add(
            new int[] {MisakiStress.hasPrimary(ps) ? 1 : 0, MisakiStress.stressWeight(ps), i});
      }
    }
    if (indices.size() == 2 && tokens.get(indices.get(0)[2]).text.length() == 1) {
      int i = indices.get(1)[2];
      tokens.get(i).phonemes = MisakiStress.applyStress(tokens.get(i).phonemes, -0.5);
      return;
    }
    int primaries = 0;
    for (int[] entry : indices) {
      primaries += entry[0];
    }
    if (indices.size() < 2 || primaries <= (indices.size() + 1) / 2) {
      return;
    }
    indices.sort(
        (a, b) ->
            a[0] != b[0]
                ? Integer.compare(a[0], b[0])
                : a[1] != b[1] ? Integer.compare(a[1], b[1]) : Integer.compare(a[2], b[2]));
    for (int k = 0; k < indices.size() / 2; k++) {
      int i = indices.get(k)[2];
      tokens.get(i).phonemes = MisakiStress.applyStress(tokens.get(i).phonemes, -0.5);
    }
  }

  /** misaki's {@code merge_tokens}; {@code unk == null} leaves the merged phonemes unset. */
  static MToken mergeTokens(List<MToken> tokens, String unk) {
    Set<Double> stresses = new TreeSet<>();
    String currency = null;
    boolean anyNullRating = false;
    Integer minRating = null;
    StringBuilder numFlags = new StringBuilder();
    for (MToken tk : tokens) {
      if (tk.stress != null) {
        stresses.add(tk.stress);
      }
      if (tk.currency != null && (currency == null || tk.currency.compareTo(currency) > 0)) {
        currency = tk.currency;
      }
      if (tk.rating == null) {
        anyNullRating = true;
      } else if (minRating == null || tk.rating < minRating) {
        minRating = tk.rating;
      }
      for (char c : tk.numFlags.toCharArray()) {
        if (numFlags.indexOf(String.valueOf(c)) < 0) {
          numFlags.append(c);
        }
      }
    }
    String phonemes = null;
    if (unk != null) {
      StringBuilder ps = new StringBuilder();
      for (MToken tk : tokens) {
        if (tk.prespace
            && ps.length() > 0
            && !Character.isWhitespace(ps.charAt(ps.length() - 1))
            && tk.phonemes != null
            && !tk.phonemes.isEmpty()) {
          ps.append(' ');
        }
        ps.append(tk.phonemes == null ? unk : tk.phonemes);
      }
      phonemes = ps.toString();
    }
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < tokens.size() - 1; i++) {
      text.append(tokens.get(i).text).append(tokens.get(i).whitespace);
    }
    text.append(tokens.get(tokens.size() - 1).text);
    MToken best = tokens.get(0);
    int bestWeight = -1;
    for (MToken tk : tokens) {
      int weight = 0;
      for (char c : tk.text.toCharArray()) {
        weight += c == Character.toLowerCase(c) ? 1 : 2;
      }
      if (weight > bestWeight) {
        bestWeight = weight;
        best = tk;
      }
    }
    MToken merged = new MToken(text.toString(), best.tag, tokens.get(tokens.size() - 1).whitespace);
    merged.phonemes = phonemes;
    merged.isHead = tokens.get(0).isHead;
    merged.alias = null;
    merged.stress = stresses.size() == 1 ? stresses.iterator().next() : null;
    merged.currency = currency;
    char[] flags = numFlags.toString().toCharArray();
    java.util.Arrays.sort(flags);
    merged.numFlags = new String(flags);
    merged.prespace = tokens.get(0).prespace;
    merged.rating = anyNullRating ? null : minRating;
    return merged;
  }

  @Override
  public String toString() {
    return "MisakiEnglishPhonemizer[" + (british() ? "gb" : "us") + "]";
  }

  static String lower(String s) {
    return s.toLowerCase(Locale.ROOT);
  }
}
