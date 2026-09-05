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
package com.integrallis.models.backend.purejava.soprano;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/** Soprano's lowercase byte-level BPE prompt tokenizer. */
public final class SopranoTokenizer {

  private static final JsonFactory JSON = new JsonFactory();
  private static final Pattern WHITESPACE = Pattern.compile("\\s", Pattern.UNICODE_CHARACTER_CLASS);
  private static final Pattern WORD = Pattern.compile("\\w", Pattern.UNICODE_CHARACTER_CLASS);

  private final Map<String, Integer> vocabulary;
  private final List<Merge> merges;
  private final int unknownToken;
  private final int textToken;
  private final int startToken;
  private final int stopToken;

  private SopranoTokenizer(Map<String, Integer> vocabulary, List<Merge> merges) {
    this.vocabulary = Map.copyOf(vocabulary);
    this.merges = List.copyOf(merges);
    unknownToken = requireToken("[UNK]");
    textToken = requireToken("[TEXT]");
    startToken = requireToken("[START]");
    stopToken = requireToken("[STOP]");
  }

  /** Parses the tokenizer JSON embedded in a Soprano GGUF package. */
  public static SopranoTokenizer fromJson(String json) {
    Map<String, Integer> vocabulary = new HashMap<>();
    List<String[]> mergeTexts = new ArrayList<>();
    try (JsonParser parser = JSON.createParser(json)) {
      if (parser.nextToken() != JsonToken.START_OBJECT) {
        throw new IllegalArgumentException("Soprano tokenizer must be a JSON object");
      }
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        if ("added_tokens".equals(name)) {
          readAddedTokens(parser, value, vocabulary);
        } else if ("model".equals(name)) {
          readModel(parser, value, vocabulary, mergeTexts);
        } else {
          parser.skipChildren();
        }
      }
    } catch (IOException failure) {
      throw new IllegalArgumentException("Cannot parse Soprano tokenizer", failure);
    }
    List<Merge> merges = new ArrayList<>(mergeTexts.size());
    for (String[] pair : mergeTexts) {
      Integer left = vocabulary.get(pair[0]);
      Integer right = vocabulary.get(pair[1]);
      Integer merged = vocabulary.get(pair[0] + pair[1]);
      if (left != null && right != null && merged != null) {
        merges.add(new Merge(left, right, merged));
      }
    }
    return new SopranoTokenizer(vocabulary, merges);
  }

  /** Normalizes text, applies BPE, and wraps it in Soprano's speech-generation prompt. */
  public int[] encodePrompt(String rawText) {
    if (rawText == null) {
      throw new NullPointerException("rawText");
    }
    String text = normalize(rawText);
    List<Integer> encoded = new ArrayList<>();
    encoded.add(stopToken);
    encoded.add(textToken);
    for (String piece : preTokenize(text)) {
      List<Integer> word = new ArrayList<>(piece.codePointCount(0, piece.length()));
      for (int index = 0; index < piece.length(); ) {
        int codePoint = piece.codePointAt(index);
        String character = new String(Character.toChars(codePoint));
        word.add(vocabulary.getOrDefault(character, unknownToken));
        index += Character.charCount(codePoint);
      }
      applyMerges(word);
      encoded.addAll(word);
    }
    encoded.add(startToken);
    return encoded.stream().mapToInt(Integer::intValue).toArray();
  }

  /** End-of-speech token used by the autoregressive model. */
  public int eosToken() {
    return stopToken;
  }

  private void applyMerges(List<Integer> word) {
    while (word.size() > 1) {
      int bestRank = -1;
      int bestPosition = -1;
      for (int position = 0; position + 1 < word.size(); position++) {
        for (int rank = 0; rank < merges.size(); rank++) {
          Merge merge = merges.get(rank);
          if (merge.left == word.get(position) && merge.right == word.get(position + 1)) {
            if (bestRank < 0 || rank < bestRank) {
              bestRank = rank;
              bestPosition = position;
            }
            break;
          }
        }
      }
      if (bestRank < 0) {
        return;
      }
      word.set(bestPosition, merges.get(bestRank).result);
      word.remove(bestPosition + 1);
    }
  }

  private int requireToken(String text) {
    Integer token = vocabulary.get(text);
    if (token == null) {
      throw new IllegalArgumentException("Soprano tokenizer is missing " + text);
    }
    return token;
  }

  private static String normalize(String text) {
    String lower = text.toLowerCase(Locale.ROOT);
    StringBuilder result = new StringBuilder(lower.length());
    boolean previousWhitespace = false;
    for (int index = 0; index < lower.length(); ) {
      int codePoint = lower.codePointAt(index);
      if (matches(WHITESPACE, codePoint)) {
        if (!previousWhitespace) {
          result.append(' ');
        }
        previousWhitespace = true;
      } else {
        result.appendCodePoint(codePoint);
        previousWhitespace = false;
      }
      index += Character.charCount(codePoint);
    }
    return result.toString();
  }

  private static List<String> preTokenize(String text) {
    List<String> pieces = new ArrayList<>();
    int start = 0;
    while (start < text.length()) {
      int first = text.codePointAt(start);
      int end = start + Character.charCount(first);
      if (Character.isDigit(first)) {
        pieces.add(text.substring(start, end));
        start = end;
        continue;
      }

      int category = category(first);
      while (end < text.length()) {
        int next = text.codePointAt(end);
        if (Character.isDigit(next) || category(next) != category) {
          break;
        }
        end += Character.charCount(next);
      }
      pieces.add(text.substring(start, end));
      start = end;
    }
    return pieces;
  }

  private static int category(int codePoint) {
    if (matches(WHITESPACE, codePoint)) {
      return 0;
    }
    return matches(WORD, codePoint) ? 1 : 2;
  }

  private static boolean matches(Pattern pattern, int codePoint) {
    return pattern.matcher(new String(Character.toChars(codePoint))).matches();
  }

  private static void readAddedTokens(
      JsonParser parser, JsonToken token, Map<String, Integer> vocabulary) throws IOException {
    require(token, JsonToken.START_ARRAY, "added_tokens");
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      require(parser.currentToken(), JsonToken.START_OBJECT, "added token");
      Integer id = null;
      String content = null;
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        String field = parser.currentName();
        JsonToken value = parser.nextToken();
        if ("id".equals(field)) {
          id = parser.getIntValue();
        } else if ("content".equals(field)) {
          content = parser.getValueAsString();
        } else {
          parser.skipChildren();
        }
      }
      if (id != null && content != null) {
        vocabulary.put(content, id);
      }
    }
  }

  private static void readModel(
      JsonParser parser, JsonToken token, Map<String, Integer> vocabulary, List<String[]> merges)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "model");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      String field = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("vocab".equals(field)) {
        require(value, JsonToken.START_OBJECT, "vocab");
        while (parser.nextToken() != JsonToken.END_OBJECT) {
          String text = parser.currentName();
          parser.nextToken();
          vocabulary.put(text, parser.getIntValue());
        }
      } else if ("merges".equals(field)) {
        require(value, JsonToken.START_ARRAY, "merges");
        while (parser.nextToken() != JsonToken.END_ARRAY) {
          require(parser.currentToken(), JsonToken.START_ARRAY, "merge");
          parser.nextToken();
          String left = parser.getValueAsString();
          parser.nextToken();
          String right = parser.getValueAsString();
          require(parser.nextToken(), JsonToken.END_ARRAY, "merge");
          merges.add(new String[] {left, right});
        }
      } else {
        parser.skipChildren();
      }
    }
  }

  private static void require(JsonToken actual, JsonToken expected, String description) {
    if (actual != expected) {
      throw new IllegalArgumentException(
          "Soprano tokenizer " + description + " must be " + expected + ", got " + actual);
    }
  }

  private record Merge(int left, int right, int result) {}
}
