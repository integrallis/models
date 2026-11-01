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
import java.util.Set;
import java.util.regex.Pattern;

/**
 * A rule-based stand-in for the spaCy tokenizer and tagger misaki runs on.
 *
 * <p>It splits on whitespace, peels bracketing punctuation off either end of each piece, keeps
 * contractions and common abbreviations whole, splits hyphens and dashes between words, and assigns
 * the handful of Penn Treebank tags the lexicon rules actually read: determiners, {@code TO},
 * prepositions, pronouns, auxiliaries, cardinals, proper nouns by capitalisation, and the
 * punctuation tags. Every other word is tagged {@code NN}. How far this falls short of spaCy is
 * measured by the phoneme-agreement number in the qualification record, never assumed.
 */
public final class EnglishTokenizer {

  private static final String OPENERS = "([{\"“‘'«";
  private static final String CLOSERS = ")]}\"”’'»,;:!?…—–";
  private static final Set<String> ABBREVIATIONS =
      Set.of(
          "Dr.", "Mr.", "Mrs.", "Ms.", "St.", "Jr.", "Sr.", "vs.", "etc.", "e.g.", "i.e.", "No.",
          "Inc.", "Ltd.", "Co.", "Mt.", "Prof.", "Capt.", "Sgt.", "Gen.", "Rev.", "Ave.", "Blvd.",
          "a.m.", "p.m.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.",
          "Oct.", "Nov.", "Dec.", "U.S.", "U.K.", "U.N.", "Ph.D.", "B.C.", "A.D.");
  private static final Pattern INITIALISM = Pattern.compile("^(?:\\p{L}\\.)+$");
  private static final Pattern NUMBER =
      Pattern.compile("^[+-]?(?:\\d[\\d,]*(?:\\.\\d+)?|\\.\\d+)(?:st|nd|rd|th|s)?$");
  private static final Pattern WORD_DASH_WORD =
      Pattern.compile("(?<=[\\p{L}\\d])([-–—])(?=[\\p{L}\\d])");
  private static final Set<String> DETERMINERS =
      Set.of(
          "a", "an", "the", "this", "that", "these", "those", "every", "some", "any", "no", "each",
          "another");
  private static final Set<String> PREPOSITIONS =
      Set.of(
          "in", "by", "of", "on", "at", "for", "with", "from", "as", "into", "over", "under",
          "about", "after", "before", "between", "through", "during", "without", "within", "upon",
          "than", "like");
  private static final Set<String> PRONOUNS =
      Set.of("i", "he", "she", "it", "we", "they", "you", "me", "him", "her", "us", "them");
  private static final Set<String> CONJUNCTIONS = Set.of("and", "or", "but", "nor", "yet", "so");

  /** Words that open a clause after a complementizer "that" rather than a noun phrase it owns. */
  private static final Set<String> CLAUSE_OPENERS =
      Set.of(
          "there", "if", "when", "while", "not", "all", "none", "both", "his", "her", "its",
          "their", "our", "my", "your", "what", "which", "who", "whom", "whose", "how", "where",
          "why");

  private EnglishTokenizer() {}

  /** Tokenizes {@code text}; each token records whether whitespace followed it. */
  public static List<MToken> tokenize(String text) {
    List<MToken> tokens = new ArrayList<>();
    int i = 0;
    int length = text.length();
    while (i < length) {
      while (i < length && Character.isWhitespace(text.charAt(i))) {
        i++;
      }
      if (i >= length) {
        break;
      }
      int start = i;
      while (i < length && !Character.isWhitespace(text.charAt(i))) {
        i++;
      }
      String piece = text.substring(start, i);
      boolean spaceAfter = i < length;
      splitPiece(piece, spaceAfter, tokens);
    }
    tagAll(tokens);
    return tokens;
  }

  private static void splitPiece(String piece, boolean spaceAfter, List<MToken> out) {
    List<String> prefixes = new ArrayList<>();
    List<String> suffixes = new ArrayList<>();
    String core = piece;
    while (core.length() > 1 && OPENERS.indexOf(core.charAt(0)) >= 0) {
      prefixes.add(core.substring(0, 1));
      core = core.substring(1);
    }
    while (core.length() > 1
        && (isCurrency(core.charAt(0)) || core.charAt(0) == '#')
        && Character.isDigit(core.charAt(1))) {
      prefixes.add(core.substring(0, 1));
      core = core.substring(1);
    }
    while (core.length() > 1) {
      char last = core.charAt(core.length() - 1);
      if (CLOSERS.indexOf(last) >= 0) {
        if ((last == '\'' || last == '’') && core.charAt(core.length() - 2) == 's') {
          break;
        }
        suffixes.add(0, String.valueOf(last));
        core = core.substring(0, core.length() - 1);
      } else if (last == '.'
          && !ABBREVIATIONS.contains(core)
          && !INITIALISM.matcher(core).matches()) {
        suffixes.add(0, ".");
        core = core.substring(0, core.length() - 1);
      } else {
        break;
      }
    }
    List<String> cores = new ArrayList<>();
    if (NUMBER.matcher(core).matches() || ABBREVIATIONS.contains(core)) {
      cores.add(core);
    } else {
      int from = 0;
      java.util.regex.Matcher dash = WORD_DASH_WORD.matcher(core);
      while (dash.find()) {
        cores.add(core.substring(from, dash.start()));
        cores.add(dash.group(1));
        from = dash.end();
      }
      cores.add(core.substring(from));
    }
    List<String> all = new ArrayList<>(prefixes);
    all.addAll(cores);
    all.addAll(suffixes);
    for (int k = 0; k < all.size(); k++) {
      boolean last = k == all.size() - 1;
      out.add(new MToken(all.get(k), null, last && spaceAfter ? " " : ""));
    }
  }

  private static boolean isCurrency(char c) {
    return c == '$' || c == '£' || c == '€';
  }

  private static void tagAll(List<MToken> tokens) {
    boolean sentenceStart = true;
    boolean quoteOpen = false;
    boolean shouting = allCaps(tokens);
    for (int i = 0; i < tokens.size(); i++) {
      MToken tk = tokens.get(i);
      String next = i + 1 < tokens.size() ? tokens.get(i + 1).text : "";
      String tag;
      String t = tk.text;
      if (t.equals("\"")) {
        tag = quoteOpen ? "''" : "\u0060\u0060";
        quoteOpen = !quoteOpen;
      } else {
        tag = tagOf(t, next, sentenceStart, shouting);
      }
      tk.tag = tag;
      sentenceStart =
          tag.equals(".") || (sentenceStart && (tag.equals("\u0060\u0060") || tag.equals("-LRB-")));
    }
  }

  /** Whether every word of two or more letters is upper-case, so capitals carry no information. */
  private static boolean allCaps(List<MToken> tokens) {
    boolean sawWord = false;
    for (MToken tk : tokens) {
      String t = tk.text;
      if (t.length() < 2 || t.chars().noneMatch(Character::isLetter)) {
        continue;
      }
      sawWord = true;
      if (!t.equals(t.toUpperCase(Locale.ROOT))) {
        return false;
      }
    }
    return sawWord;
  }

  static String tagOf(String t, String next, boolean sentenceStart) {
    return tagOf(t, next, sentenceStart, false);
  }

  /**
   * Whether the token after "that" opens a clause (complementizer or relative "that", unstressed in
   * the lexicon) rather than a noun phrase the determiner owns.
   */
  private static boolean complementizerFollows(String next) {
    if (next.isEmpty()) {
      return true;
    }
    char c = next.charAt(0);
    if (!Character.isLetterOrDigit(c) || Character.isUpperCase(c)) {
      return true;
    }
    String lower = next.toLowerCase(Locale.ROOT);
    if (DETERMINERS.contains(lower)
        || PRONOUNS.contains(lower)
        || CLAUSE_OPENERS.contains(lower)
        || NUMBER.matcher(next).matches()) {
      return true;
    }
    return lower.length() > 3
        && (lower.endsWith("ed")
            || lower.endsWith("ing")
            || (lower.endsWith("s") && !lower.endsWith("ss")));
  }

  static String tagOf(String t, String next, boolean sentenceStart, boolean shouting) {
    if (t.length() == 1) {
      char c = t.charAt(0);
      switch (c) {
        case '.', '!', '?' -> {
          return ".";
        }
        case ',' -> {
          return ",";
        }
        case ';', ':', '—', '–', '…' -> {
          return ":";
        }
        case '-' -> {
          return "HYPH";
        }
        case '“', '‘' -> {
          return "\u0060\u0060";
        }
        case '”', '’' -> {
          return "''";
        }
        case '(', '[', '{' -> {
          return "-LRB-";
        }
        case ')', ']', '}' -> {
          return "-RRB-";
        }
        case '$', '£', '€' -> {
          return "$";
        }
        case '#' -> {
          return "#";
        }
        case '\'' -> {
          return "''";
        }
        default -> {}
      }
    }
    if (NUMBER.matcher(t).matches()) {
      return "CD";
    }
    boolean letters = t.chars().anyMatch(Character::isLetter);
    if (!letters) {
      return "NFP";
    }
    String lower = t.toLowerCase(Locale.ROOT);
    if (t.equals("I")) {
      return "PRP";
    }
    if (t.length() > 1 && t.equals(t.toUpperCase(Locale.ROOT)) && !shouting) {
      // An all-caps word in mixed-case text is an initialism or a proper noun to the tagger misaki
      // runs ("US", "NASA"), even when its lowercase spelling is a function word; the lexicon's
      // own entries then decide whether to spell it out.
      return "NNP";
    }
    if (lower.equals("that") && !sentenceStart && complementizerFollows(next)) {
      return "IN";
    }
    if (DETERMINERS.contains(lower)) {
      return "DT";
    }
    if (lower.equals("to")) {
      return "TO";
    }
    if (PREPOSITIONS.contains(lower)) {
      return "IN";
    }
    if (PRONOUNS.contains(lower)) {
      return "PRP";
    }
    if (CONJUNCTIONS.contains(lower)) {
      return "CC";
    }
    switch (lower) {
      case "am" -> {
        return "VBP";
      }
      case "is", "has", "does" -> {
        return "VBZ";
      }
      case "are", "have", "do" -> {
        return "VBP";
      }
      case "was", "were", "did", "had" -> {
        return "VBD";
      }
      case "be" -> {
        return "VB";
      }
      case "been" -> {
        return "VBN";
      }
      case "used" -> {
        return next.equalsIgnoreCase("to") ? "VBD" : "VBN";
      }
      case "not", "very", "also", "just", "only", "too", "then", "there", "here", "now" -> {
        return "RB";
      }
      default -> {}
    }
    if (ABBREVIATIONS.contains(t) || INITIALISM.matcher(t).matches()) {
      return "NNP";
    }
    boolean allUpper = t.equals(t.toUpperCase(Locale.ROOT)) && t.length() > 1;
    boolean capitalized = Character.isUpperCase(t.charAt(0)) && !allUpper;
    if (allUpper || (capitalized && !sentenceStart)) {
      return "NNP";
    }
    if (lower.endsWith("ly") && lower.length() > 4) {
      return "RB";
    }
    if (lower.endsWith("ing") && lower.length() > 5) {
      return "VBG";
    }
    return "NN";
  }
}
