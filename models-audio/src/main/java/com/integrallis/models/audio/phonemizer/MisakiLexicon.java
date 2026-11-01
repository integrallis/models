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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * misaki's English lexicon: the gold and silver dictionaries plus the word-level rules that sit on
 * top of them (capitalisation, stems, special cases, numbers and currency).
 *
 * <p>The dictionaries are the Apache-2.0 files {@code us_gold.json}, {@code us_silver.json}, {@code
 * gb_gold.json} and {@code gb_silver.json} from misaki at revision {@code fba1236}; they are
 * supplied as paths and are not bundled. Method names follow the original so the port can be read
 * beside it.
 */
public final class MisakiLexicon {

  /** A dictionary value: either one spelling, or spellings by part-of-speech tag with a default. */
  record Entry(String single, Map<String, String> byTag) {
    boolean isMap() {
      return byTag != null;
    }
  }

  /** What a lookup returns: the phonemes and misaki's confidence rating, or nulls. */
  record Result(String phonemes, Integer rating) {
    static final Result NONE = new Result(null, null);

    boolean found() {
      return phonemes != null;
    }
  }

  /** The two context flags the pipeline threads backwards through a sentence. */
  record Context(Boolean futureVowel, boolean futureTo) {
    static final Context INITIAL = new Context(null, false);
  }

  static final Map<String, String[]> CURRENCIES =
      Map.of(
          "$",
          new String[] {"dollar", "cent"},
          "£",
          new String[] {"pound", "pence"},
          "€",
          new String[] {"euro", "cent"});
  static final Set<String> ORDINALS = Set.of("st", "nd", "rd", "th");
  static final Map<String, String> ADD_SYMBOLS = Map.of(".", "dot", "/", "slash");
  static final Map<String, String> SYMBOLS =
      Map.of("%", "percent", "&", "and", "+", "plus", "@", "at");
  private static final Pattern DIGITS = Pattern.compile("^[0-9]+$");
  private static final Pattern VS = Pattern.compile("(?i)vs\\.?$");
  private static final Pattern NUMBER_SUFFIX = Pattern.compile("[a-z']+$");
  private static final Pattern NON_LETTERS = Pattern.compile("[^a-z]+");
  private static final Pattern DOUBLED_CONSONANT_ING =
      Pattern.compile("([bcdgklmnprstvxz])\\1ing$|cking$");

  private final boolean british;
  private final Map<String, Entry> golds;
  private final Map<String, Entry> silvers;
  private final double[] capStresses = {0.5, 2};

  private MisakiLexicon(boolean british, Map<String, Entry> golds, Map<String, Entry> silvers) {
    this.british = british;
    this.golds = golds;
    this.silvers = silvers;
  }

  /** Loads a gold and a silver dictionary file for one accent. */
  public static MisakiLexicon load(Path gold, Path silver, boolean british) throws IOException {
    return of(
        Json.parseObject(Files.readString(gold, StandardCharsets.UTF_8)),
        Json.parseObject(Files.readString(silver, StandardCharsets.UTF_8)),
        british);
  }

  /** Builds a lexicon from already-parsed dictionary objects; tests use this with small maps. */
  @SuppressWarnings("unchecked")
  public static MisakiLexicon of(
      Map<String, Object> gold, Map<String, Object> silver, boolean british) {
    Map<String, Entry> golds = new HashMap<>();
    for (Map.Entry<String, Object> e : gold.entrySet()) {
      golds.put(e.getKey(), toEntry(e.getKey(), e.getValue()));
    }
    Map<String, Entry> silvers = new HashMap<>();
    for (Map.Entry<String, Object> e : silver.entrySet()) {
      silvers.put(e.getKey(), toEntry(e.getKey(), e.getValue()));
    }
    return new MisakiLexicon(british, growDictionary(golds), growDictionary(silvers));
  }

  @SuppressWarnings("unchecked")
  private static Entry toEntry(String key, Object value) {
    if (value instanceof String s) {
      return new Entry(s, null);
    }
    if (value instanceof Map<?, ?> map) {
      Map<String, String> byTag = new HashMap<>();
      for (Map.Entry<?, ?> e : map.entrySet()) {
        byTag.put((String) e.getKey(), (String) e.getValue());
      }
      if (!byTag.containsKey("DEFAULT")) {
        throw new IllegalArgumentException("lexicon entry '" + key + "' has no DEFAULT");
      }
      return new Entry(null, byTag);
    }
    throw new IllegalArgumentException(
        "lexicon entry '" + key + "' is neither a string nor an object");
  }

  /**
   * misaki's {@code grow_dictionary}: adds the capitalised form of lower-case keys and vice versa.
   */
  static Map<String, Entry> growDictionary(Map<String, Entry> d) {
    Map<String, Entry> grown = new HashMap<>();
    for (Map.Entry<String, Entry> e : d.entrySet()) {
      String k = e.getKey();
      if (k.length() < 2) {
        continue;
      }
      String lower = k.toLowerCase(Locale.ROOT);
      if (k.equals(lower)) {
        if (!k.equals(capitalize(k))) {
          grown.put(capitalize(k), e.getValue());
        }
      } else if (k.equals(capitalize(lower))) {
        grown.put(lower, e.getValue());
      }
    }
    grown.putAll(d);
    return grown;
  }

  /** Python's {@code str.capitalize()}: first character upper, the rest lower. */
  static String capitalize(String word) {
    if (word.isEmpty()) {
      return word;
    }
    return word.substring(0, 1).toUpperCase(Locale.ROOT)
        + word.substring(1).toLowerCase(Locale.ROOT);
  }

  boolean british() {
    return british;
  }

  // --- character classes ---

  static boolean isDigit(String text) {
    return DIGITS.matcher(text).matches();
  }

  static boolean isAlpha(String word) {
    if (word.isEmpty()) {
      return false;
    }
    for (int i = 0; i < word.length(); i++) {
      if (!Character.isLetter(word.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  static boolean allLexiconOrds(String word) {
    for (char c : word.toCharArray()) {
      boolean ok = c == 39 || c == 45 || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
      if (!ok) {
        return false;
      }
    }
    return true;
  }

  private static boolean isUpper(String word) {
    return word.equals(word.toUpperCase(Locale.ROOT));
  }

  private static boolean isLower(String word) {
    return word.equals(word.toLowerCase(Locale.ROOT));
  }

  // --- lookups ---

  Result getNnp(String word) {
    StringBuilder ps = new StringBuilder();
    for (char c : word.toCharArray()) {
      if (!Character.isLetter(c)) {
        continue;
      }
      Entry letter = golds.get(String.valueOf(c).toUpperCase(Locale.ROOT));
      if (letter == null) {
        return Result.NONE;
      }
      ps.append(letter.isMap() ? letter.byTag().get("DEFAULT") : letter.single());
    }
    String stressed = MisakiStress.applyStress(ps.toString(), 0.0);
    int last = stressed.lastIndexOf(MisakiStress.SECONDARY);
    if (last >= 0) {
      stressed = stressed.substring(0, last) + MisakiStress.PRIMARY + stressed.substring(last + 1);
    }
    return new Result(stressed, 3);
  }

  Result getSpecialCase(String word, String tag, Double stress, Context ctx) {
    if ("ADD".equals(tag) && ADD_SYMBOLS.containsKey(word)) {
      return lookup(ADD_SYMBOLS.get(word), null, -0.5, ctx);
    }
    if (SYMBOLS.containsKey(word)) {
      return lookup(SYMBOLS.get(word), null, null, ctx);
    }
    String stripped = word.replaceAll("^\\.+|\\.+$", "");
    if (stripped.indexOf('.') >= 0 && isAlpha(word.replace(".", "")) && longestPiece(word) < 3) {
      return getNnp(word);
    }
    if (word.equals("a") || word.equals("A")) {
      return new Result("DT".equals(tag) ? "ɐ" : "ˈA", 4);
    }
    if (word.equals("am") || word.equals("Am") || word.equals("AM")) {
      if (tag != null && tag.startsWith("NN")) {
        return getNnp(word);
      }
      if (ctx.futureVowel() == null || !word.equals("am") || (stress != null && stress > 0)) {
        return new Result(single(golds.get("am")), 4);
      }
      return new Result("ɐm", 4);
    }
    if (word.equals("an") || word.equals("An") || word.equals("AN")) {
      if (word.equals("AN") && tag != null && tag.startsWith("NN")) {
        return getNnp(word);
      }
      return new Result("ɐn", 4);
    }
    if (word.equals("I") && "PRP".equals(tag)) {
      return new Result(MisakiStress.SECONDARY + "I", 4);
    }
    if ((word.equals("by") || word.equals("By") || word.equals("BY"))
        && "ADV".equals(getParentTag(tag))) {
      return new Result("bˈI", 4);
    }
    if (word.equals("to")
        || word.equals("To")
        || (word.equals("TO") && ("TO".equals(tag) || "IN".equals(tag)))) {
      String ps =
          ctx.futureVowel() == null ? single(golds.get("to")) : ctx.futureVowel() ? "tʊ" : "tə";
      return new Result(ps, 4);
    }
    if (word.equals("in") || word.equals("In") || (word.equals("IN") && !"NNP".equals(tag))) {
      String prefix =
          ctx.futureVowel() == null || !"IN".equals(tag)
              ? String.valueOf(MisakiStress.PRIMARY)
              : "";
      return new Result(prefix + "ɪn", 4);
    }
    if (word.equals("the") || word.equals("The") || (word.equals("THE") && "DT".equals(tag))) {
      return new Result(Boolean.TRUE.equals(ctx.futureVowel()) ? "ði" : "ðə", 4);
    }
    if ("IN".equals(tag) && VS.matcher(word).matches()) {
      return lookup("versus", null, null, ctx);
    }
    if (word.equals("used") || word.equals("Used") || word.equals("USED")) {
      Entry used = golds.get("used");
      if (("VBD".equals(tag) || "JJ".equals(tag)) && ctx.futureTo()) {
        return new Result(used.byTag().get("VBD"), 4);
      }
      return new Result(used.byTag().get("DEFAULT"), 4);
    }
    return Result.NONE;
  }

  private static int longestPiece(String word) {
    int longest = 0;
    for (String piece : word.split("\\.", -1)) {
      longest = Math.max(longest, piece.length());
    }
    return longest;
  }

  private static String single(Entry entry) {
    return entry.isMap() ? entry.byTag().get("DEFAULT") : entry.single();
  }

  static String getParentTag(String tag) {
    if (tag == null) {
      return null;
    }
    if (tag.startsWith("VB")) {
      return "VERB";
    }
    if (tag.startsWith("NN")) {
      return "NOUN";
    }
    if (tag.startsWith("ADV") || tag.startsWith("RB")) {
      return "ADV";
    }
    if (tag.startsWith("ADJ") || tag.startsWith("JJ")) {
      return "ADJ";
    }
    return tag;
  }

  boolean isKnown(String word, String tag) {
    if (golds.containsKey(word) || SYMBOLS.containsKey(word) || silvers.containsKey(word)) {
      return true;
    }
    if (!isAlpha(word) || !allLexiconOrds(word)) {
      return false;
    }
    if (word.length() == 1) {
      return true;
    }
    if (isUpper(word) && golds.containsKey(word.toLowerCase(Locale.ROOT))) {
      return true;
    }
    String rest = word.substring(1);
    return rest.equals(rest.toUpperCase(Locale.ROOT));
  }

  Result lookup(String word, String tag, Double stress, Context ctx) {
    Boolean isNnp = null;
    if (isUpper(word) && !golds.containsKey(word)) {
      word = word.toLowerCase(Locale.ROOT);
      isNnp = "NNP".equals(tag);
    }
    Entry entry = golds.get(word);
    int rating = 4;
    if (entry == null && !Boolean.TRUE.equals(isNnp)) {
      entry = silvers.get(word);
      rating = 3;
    }
    String ps = null;
    if (entry != null) {
      if (entry.isMap()) {
        Map<String, String> byTag = entry.byTag();
        String key = tag;
        if (ctx != null && ctx.futureVowel() == null && byTag.containsKey("None")) {
          key = "None";
        } else if (!byTag.containsKey(key)) {
          key = getParentTag(tag);
        }
        ps = byTag.containsKey(key) ? byTag.get(key) : byTag.get("DEFAULT");
      } else {
        ps = entry.single();
      }
    }
    if (ps == null || (Boolean.TRUE.equals(isNnp) && !MisakiStress.hasPrimary(ps))) {
      Result nnp = getNnp(word);
      if (nnp.found()) {
        return nnp;
      }
    }
    return new Result(MisakiStress.applyStress(ps, stress), rating);
  }

  String pluralS(String stem) {
    if (stem == null || stem.isEmpty()) {
      return null;
    }
    char last = stem.charAt(stem.length() - 1);
    if ("ptkfθ".indexOf(last) >= 0) {
      return stem + "s";
    }
    if ("szʃʒʧʤ".indexOf(last) >= 0) {
      return stem + (british ? "ɪ" : "ᵻ") + "z";
    }
    return stem + "z";
  }

  Result stemS(String word, String tag, Double stress, Context ctx) {
    if (word.length() < 3 || !word.endsWith("s")) {
      return Result.NONE;
    }
    String stem;
    if (!word.endsWith("ss") && isKnown(word.substring(0, word.length() - 1), tag)) {
      stem = word.substring(0, word.length() - 1);
    } else if ((word.endsWith("'s")
            || (word.length() > 4 && word.endsWith("es") && !word.endsWith("ies")))
        && isKnown(word.substring(0, word.length() - 2), tag)) {
      stem = word.substring(0, word.length() - 2);
    } else if (word.length() > 4
        && word.endsWith("ies")
        && isKnown(word.substring(0, word.length() - 3) + "y", tag)) {
      stem = word.substring(0, word.length() - 3) + "y";
    } else {
      return Result.NONE;
    }
    Result base = lookup(stem, tag, stress, ctx);
    return new Result(pluralS(base.phonemes()), base.rating());
  }

  String pastEd(String stem) {
    if (stem == null || stem.isEmpty()) {
      return null;
    }
    char last = stem.charAt(stem.length() - 1);
    if ("pkfθʃsʧ".indexOf(last) >= 0) {
      return stem + "t";
    }
    if (last == 'd') {
      return stem + (british ? "ɪ" : "ᵻ") + "d";
    }
    if (last != 't') {
      return stem + "d";
    }
    if (british || stem.length() < 2) {
      return stem + "ɪd";
    }
    if (MisakiStress.US_TAUS.contains(stem.charAt(stem.length() - 2))) {
      return stem.substring(0, stem.length() - 1) + "ɾᵻd";
    }
    return stem + "ᵻd";
  }

  Result stemEd(String word, String tag, Double stress, Context ctx) {
    if (word.length() < 4 || !word.endsWith("d")) {
      return Result.NONE;
    }
    String stem;
    if (!word.endsWith("dd") && isKnown(word.substring(0, word.length() - 1), tag)) {
      stem = word.substring(0, word.length() - 1);
    } else if (word.length() > 4
        && word.endsWith("ed")
        && !word.endsWith("eed")
        && isKnown(word.substring(0, word.length() - 2), tag)) {
      stem = word.substring(0, word.length() - 2);
    } else {
      return Result.NONE;
    }
    Result base = lookup(stem, tag, stress, ctx);
    return new Result(pastEd(base.phonemes()), base.rating());
  }

  String progressiveIng(String stem) {
    if (stem == null || stem.isEmpty()) {
      return null;
    }
    char last = stem.charAt(stem.length() - 1);
    if (british) {
      if (last == 'ə' || last == 'ː') {
        return null;
      }
    } else if (stem.length() > 1
        && last == 't'
        && MisakiStress.US_TAUS.contains(stem.charAt(stem.length() - 2))) {
      return stem.substring(0, stem.length() - 1) + "ɾɪŋ";
    }
    return stem + "ɪŋ";
  }

  Result stemIng(String word, String tag, Double stress, Context ctx) {
    if (word.length() < 5 || !word.endsWith("ing")) {
      return Result.NONE;
    }
    String stem;
    if (word.length() > 5 && isKnown(word.substring(0, word.length() - 3), tag)) {
      stem = word.substring(0, word.length() - 3);
    } else if (isKnown(word.substring(0, word.length() - 3) + "e", tag)) {
      stem = word.substring(0, word.length() - 3) + "e";
    } else if (word.length() > 5
        && DOUBLED_CONSONANT_ING.matcher(word).find()
        && isKnown(word.substring(0, word.length() - 4), tag)) {
      stem = word.substring(0, word.length() - 4);
    } else {
      return Result.NONE;
    }
    Result base = lookup(stem, tag, stress, ctx);
    return new Result(progressiveIng(base.phonemes()), base.rating());
  }

  Result getWord(String word, String tag, Double stress, Context ctx) {
    Result special = getSpecialCase(word, tag, stress, ctx);
    if (special.found()) {
      return special;
    }
    String wl = word.toLowerCase(Locale.ROOT);
    if (word.length() > 1
        && isAlpha(word.replace("'", ""))
        && !word.equals(wl)
        && (!"NNP".equals(tag) || word.length() > 7)
        && !golds.containsKey(word)
        && !silvers.containsKey(word)
        && (isUpper(word) || isLower(word.substring(1)))
        && (golds.containsKey(wl)
            || silvers.containsKey(wl)
            || stemS(wl, tag, stress, ctx).found()
            || stemEd(wl, tag, stress, ctx).found()
            || stemIng(wl, tag, stress, ctx).found())) {
      word = wl;
    }
    if (isKnown(word, tag)) {
      return lookup(word, tag, stress, ctx);
    }
    if (word.endsWith("s'") && isKnown(word.substring(0, word.length() - 2) + "'s", tag)) {
      return lookup(word.substring(0, word.length() - 2) + "'s", tag, stress, ctx);
    }
    if (word.endsWith("'") && isKnown(word.substring(0, word.length() - 1), tag)) {
      return lookup(word.substring(0, word.length() - 1), tag, stress, ctx);
    }
    Result s = stemS(word, tag, stress, ctx);
    if (s.found()) {
      return s;
    }
    Result ed = stemEd(word, tag, stress, ctx);
    if (ed.found()) {
      return ed;
    }
    Double ingStress = stress == null ? Double.valueOf(0.5) : stress;
    Result ing = stemIng(word, tag, ingStress, ctx);
    if (ing.found()) {
      return ing;
    }
    return Result.NONE;
  }

  static boolean isCurrency(String word) {
    if (word.indexOf('.') < 0) {
      return true;
    }
    if (word.chars().filter(c -> c == '.').count() > 1) {
      return false;
    }
    String cents = word.substring(word.indexOf('.') + 1);
    return cents.length() < 3 || cents.chars().allMatch(c -> c == '0');
  }

  Result getNumber(String word, String currency, boolean isHead, String numFlags) {
    Matcher suffixMatch = NUMBER_SUFFIX.matcher(word);
    String suffix = suffixMatch.find() ? suffixMatch.group() : null;
    if (suffix != null) {
      word = word.substring(0, word.length() - suffix.length());
    }
    List<Result> result = new ArrayList<>();
    if (word.startsWith("-")) {
      result.add(lookup("minus", null, null, null));
      word = word.substring(1);
    }
    String digitsOnly = word;
    if (isDigit(word) && suffix != null && ORDINALS.contains(suffix)) {
      extendNumber(result, EnglishNumbers.ordinal(Long.parseLong(word)), true, true, numFlags);
    } else if (result.isEmpty()
        && word.length() == 4
        && (currency == null || !CURRENCIES.containsKey(currency))
        && isDigit(word)) {
      extendNumber(result, EnglishNumbers.year(Long.parseLong(word)), true, true, numFlags);
    } else if (!isHead && word.indexOf('.') < 0) {
      String num = word.replace(",", "");
      if (num.charAt(0) == '0' || num.length() > 3) {
        for (char n : num.toCharArray()) {
          extendNumber(result, String.valueOf(n), false, false, numFlags);
        }
      } else if (num.length() == 3 && !num.endsWith("00")) {
        extendNumber(result, num.substring(0, 1), true, false, numFlags);
        if (num.charAt(1) == '0') {
          result.add(lookup("O", null, -2.0, null));
          extendNumber(result, num.substring(2), false, false, numFlags);
        } else {
          extendNumber(result, num.substring(1), false, false, numFlags);
        }
      } else {
        extendNumber(result, num, true, false, numFlags);
      }
    } else if (word.chars().filter(c -> c == '.').count() > 1 || !isHead) {
      boolean first = true;
      for (String num : word.replace(",", "").split("\\.", -1)) {
        if (num.isEmpty()) {
          continue;
        }
        boolean spellOut =
            num.charAt(0) == '0'
                || (num.length() != 2 && num.substring(1).chars().anyMatch(c -> c != '0'));
        if (spellOut) {
          for (char n : num.toCharArray()) {
            extendNumber(result, String.valueOf(n), false, false, numFlags);
          }
        } else {
          extendNumber(result, num, first, false, numFlags);
        }
        first = false;
      }
    } else if (currency != null && CURRENCIES.containsKey(currency) && isCurrency(word)) {
      String[] units = CURRENCIES.get(currency);
      String[] parts = word.replace(",", "").split("\\.", -1);
      List<long[]> pairs = new ArrayList<>();
      for (int i = 0; i < Math.min(parts.length, units.length); i++) {
        pairs.add(new long[] {parts[i].isEmpty() ? 0 : Long.parseLong(parts[i]), i});
      }
      if (pairs.size() > 1) {
        if (pairs.get(1)[0] == 0) {
          pairs = pairs.subList(0, 1);
        } else if (pairs.get(0)[0] == 0) {
          pairs = pairs.subList(1, 2);
        }
      }
      for (int i = 0; i < pairs.size(); i++) {
        long num = pairs.get(i)[0];
        String unit = units[(int) pairs.get(i)[1]];
        if (i > 0) {
          result.add(lookup("and", null, null, null));
        }
        extendNumber(result, EnglishNumbers.cardinal(num), i == 0, true, numFlags);
        result.add(
            Math.abs(num) != 1 && !unit.equals("pence")
                ? stemS(unit + "s", null, null, null)
                : lookup(unit, null, null, null));
      }
    } else {
      String words;
      if (isDigit(word)) {
        words = EnglishNumbers.cardinal(Long.parseLong(word));
      } else if (word.indexOf('.') < 0) {
        long value = Long.parseLong(word.replace(",", ""));
        words =
            suffix != null && ORDINALS.contains(suffix)
                ? EnglishNumbers.ordinal(value)
                : EnglishNumbers.cardinal(value);
      } else {
        String plain = word.replace(",", "");
        if (plain.charAt(0) == '.') {
          StringBuilder spelled = new StringBuilder("point");
          for (char n : plain.substring(1).toCharArray()) {
            spelled.append(' ').append(EnglishNumbers.cardinal(n - '0'));
          }
          words = spelled.toString();
        } else {
          words = EnglishNumbers.decimal(plain);
        }
      }
      extendNumber(result, words, true, true, numFlags);
    }
    if (result.isEmpty()) {
      return Result.NONE;
    }
    StringBuilder joined = new StringBuilder();
    Integer rating = null;
    for (Result r : result) {
      if (r.phonemes() == null) {
        return Result.NONE;
      }
      if (joined.length() > 0) {
        joined.append(' ');
      }
      joined.append(r.phonemes());
      rating = rating == null || (r.rating() != null && r.rating() < rating) ? r.rating() : rating;
    }
    String ps = joined.toString();
    if ("s".equals(suffix) || "'s".equals(suffix)) {
      return new Result(pluralS(ps), rating);
    }
    if ("ed".equals(suffix) || "'d".equals(suffix)) {
      return new Result(pastEd(ps), rating);
    }
    if ("ing".equals(suffix)) {
      return new Result(progressiveIng(ps), rating);
    }
    return new Result(ps, rating);
  }

  /** misaki's {@code extend_num}: one number word at a time through the lexicon. */
  private void extendNumber(
      List<Result> result, String number, boolean first, boolean escape, String numFlags) {
    String words = escape ? number : EnglishNumbers.cardinal(Long.parseLong(number));
    String[] splits = NON_LETTERS.split(words);
    for (int i = 0; i < splits.length; i++) {
      String w = splits[i];
      if (w.isEmpty()) {
        continue;
      }
      if (!w.equals("and") || numFlags.indexOf('&') >= 0) {
        if (first && i == 0 && splits.length > 1 && w.equals("one") && numFlags.indexOf('a') >= 0) {
          result.add(new Result("ə", 4));
        } else {
          result.add(lookup(w, null, w.equals("point") ? -2.0 : null, null));
        }
      } else if (numFlags.indexOf('n') >= 0 && !result.isEmpty()) {
        Result last = result.remove(result.size() - 1);
        result.add(new Result(last.phonemes() + "ən", last.rating()));
      }
    }
  }

  String appendCurrency(String ps, String currency) {
    if (currency == null) {
      return ps;
    }
    String[] units = CURRENCIES.get(currency);
    String name = units == null ? null : stemS(units[0] + "s", null, null, null).phonemes();
    return name == null ? ps : ps + " " + name;
  }

  static String numericIfNeeded(char c) {
    if (!Character.isDigit(c)) {
      return String.valueOf(c);
    }
    int value = Character.getNumericValue(c);
    return value >= 0 ? String.valueOf(value) : String.valueOf(c);
  }

  static boolean isNumber(String word, boolean isHead) {
    boolean anyDigit = false;
    for (char c : word.toCharArray()) {
      anyDigit |= isDigit(String.valueOf(c));
    }
    if (!anyDigit) {
      return false;
    }
    for (String s : new String[] {"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"}) {
      if (word.endsWith(s)) {
        word = word.substring(0, word.length() - s.length());
        break;
      }
    }
    for (int i = 0; i < word.length(); i++) {
      char c = word.charAt(i);
      boolean ok =
          isDigit(String.valueOf(c)) || c == ',' || c == '.' || (isHead && i == 0 && c == '-');
      if (!ok) {
        return false;
      }
    }
    return true;
  }

  /** misaki's {@code Lexicon.__call__}: one token to phonemes, or {@link Result#NONE}. */
  Result phonemize(MToken tk, Context ctx) {
    String word = (tk.alias == null ? tk.text : tk.alias).replace('‘', '\'').replace('’', '\'');
    word = Normalizer.normalize(word, Normalizer.Form.NFKC);
    StringBuilder normalized = new StringBuilder();
    for (char c : word.toCharArray()) {
      normalized.append(numericIfNeeded(c));
    }
    word = normalized.toString();
    Double stress =
        word.equals(word.toLowerCase(Locale.ROOT)) ? null : capStresses[isUpper(word) ? 1 : 0];
    Result found = getWord(word, tk.tag, stress, ctx);
    if (found.found()) {
      return new Result(
          MisakiStress.applyStress(appendCurrency(found.phonemes(), tk.currency), tk.stress),
          found.rating());
    }
    if (isNumber(word, tk.isHead)) {
      Result number = getNumber(word, tk.currency, tk.isHead, tk.numFlags);
      return new Result(MisakiStress.applyStress(number.phonemes(), tk.stress), number.rating());
    }
    return Result.NONE;
  }
}
