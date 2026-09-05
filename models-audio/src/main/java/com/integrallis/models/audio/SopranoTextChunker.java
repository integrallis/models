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
package com.integrallis.models.audio;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Java port of audio.cpp's default Unicode-aware text chunking policy. */
final class SopranoTextChunker {

  private SopranoTextChunker() {}

  static List<String> split(String text, int codePointBudget) {
    Objects.requireNonNull(text, "text");
    if (codePointBudget <= 0) {
      throw new IllegalArgumentException("text chunk budget must be positive");
    }
    String trimmed = trimAsciiWhitespace(text);
    if (trimmed.isEmpty()) {
      return List.of();
    }
    if (trimmed.codePointCount(0, trimmed.length()) <= codePointBudget) {
      return List.of(trimmed);
    }

    List<Word> words = words(trimmed);
    List<String> chunks = new ArrayList<>();
    int wordStart = 0;
    while (wordStart < words.size()) {
      int hardEnd = wordStart;
      while (hardEnd < words.size()
          && words.get(hardEnd).codePointEnd() - words.get(wordStart).codePointStart()
              <= codePointBudget) {
        hardEnd++;
      }
      if (hardEnd == wordStart) {
        hardEnd++;
      }

      int chunkEnd = hardEnd;
      if (hardEnd < words.size() && hardEnd > wordStart + 1) {
        for (int index = hardEnd; index > wordStart + 1; index--) {
          if (words.get(index - 1).sentenceBreak()) {
            chunkEnd = index;
            break;
          }
        }
        if (chunkEnd == hardEnd) {
          for (int index = hardEnd; index > wordStart + 1; index--) {
            if (words.get(index - 1).clauseBreak()) {
              chunkEnd = index;
              break;
            }
          }
        }
      }

      Word first = words.get(wordStart);
      Word last = words.get(chunkEnd - 1);
      String chunk = trimAsciiWhitespace(trimmed.substring(first.charStart(), last.charEnd()));
      if (!chunk.isEmpty()) {
        chunks.add(chunk);
      }
      wordStart = chunkEnd;
    }
    return List.copyOf(chunks);
  }

  private static List<Word> words(String text) {
    List<Word> words = new ArrayList<>();
    int charIndex = 0;
    int codePointIndex = 0;
    while (charIndex < text.length()) {
      while (charIndex < text.length()) {
        int codePoint = text.codePointAt(charIndex);
        if (!isAsciiWhitespace(codePoint)) {
          break;
        }
        charIndex += Character.charCount(codePoint);
        codePointIndex++;
      }
      if (charIndex == text.length()) {
        break;
      }

      int charStart = charIndex;
      int codePointStart = codePointIndex;
      int lastCodePoint = -1;
      while (charIndex < text.length()) {
        int codePoint = text.codePointAt(charIndex);
        if (isAsciiWhitespace(codePoint)) {
          break;
        }
        lastCodePoint = codePoint;
        charIndex += Character.charCount(codePoint);
        codePointIndex++;
      }
      words.add(
          new Word(
              codePointStart,
              codePointIndex,
              charStart,
              charIndex,
              isSentenceBreak(lastCodePoint),
              isClauseBreak(lastCodePoint)));
    }
    return words;
  }

  private static String trimAsciiWhitespace(String text) {
    int start = 0;
    while (start < text.length()) {
      int codePoint = text.codePointAt(start);
      if (!isAsciiWhitespace(codePoint)) {
        break;
      }
      start += Character.charCount(codePoint);
    }
    int end = text.length();
    while (end > start) {
      int codePoint = text.codePointBefore(end);
      if (!isAsciiWhitespace(codePoint)) {
        break;
      }
      end -= Character.charCount(codePoint);
    }
    return text.substring(start, end);
  }

  private static boolean isAsciiWhitespace(int codePoint) {
    return codePoint == ' '
        || codePoint == '\t'
        || codePoint == '\n'
        || codePoint == '\r'
        || codePoint == '\f'
        || codePoint == 0x0B;
  }

  private static boolean isSentenceBreak(int codePoint) {
    return codePoint == '.'
        || codePoint == '!'
        || codePoint == '?'
        || codePoint == '。'
        || codePoint == '！'
        || codePoint == '？';
  }

  private static boolean isClauseBreak(int codePoint) {
    return codePoint == ','
        || codePoint == ';'
        || codePoint == ':'
        || codePoint == '，'
        || codePoint == '、'
        || codePoint == '；'
        || codePoint == '：';
  }

  private record Word(
      int codePointStart,
      int codePointEnd,
      int charStart,
      int charEnd,
      boolean sentenceBreak,
      boolean clauseBreak) {}
}
