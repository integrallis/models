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

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** SentencePiece unigram/Viterbi tokenizer used by T5-family GGUF artifacts. */
final class UnigramTokenizer {

  private static final byte[] ESCAPED_SPACE = "\u2581".getBytes(StandardCharsets.UTF_8);
  private static final double UNKNOWN_SCORE_PENALTY = 10.0;

  private final String[] vocab;
  private final float[] scores;
  private final int[] tokenTypes;
  private final Map<String, Integer> tokenToId;
  private final List<byte[]> userDefinedTokens;
  private final byte[] charsMap;
  private final int[] xcda;
  private final int replacementsOffset;
  private final boolean addSpacePrefix;
  private final boolean removeExtraWhitespaces;
  private final int unknownTokenId;
  private final double unknownTokenScore;
  private final int maxTokenChars;

  UnigramTokenizer(
      String[] vocab,
      float[] scores,
      List<Integer> tokenTypes,
      byte[] charsMap,
      boolean addSpacePrefix,
      boolean removeExtraWhitespaces,
      int unknownTokenId) {
    this.vocab = vocab;
    this.scores = scores;
    this.tokenTypes = new int[vocab.length];
    Arrays.fill(this.tokenTypes, 1);
    for (int index = 0; index < Math.min(vocab.length, tokenTypes.size()); index++) {
      this.tokenTypes[index] = tokenTypes.get(index);
    }
    this.tokenToId = new HashMap<>(vocab.length * 2);
    List<byte[]> userDefined = new ArrayList<>();
    double minimumScore = Double.POSITIVE_INFINITY;
    int longestToken = 1;
    for (int token = 0; token < vocab.length; token++) {
      int type = this.tokenTypes[token];
      if (type == 1 || type == 4 || type == 5) {
        tokenToId.put(vocab[token], token);
        longestToken = Math.max(longestToken, vocab[token].length());
      }
      if (type == 1) {
        minimumScore = Math.min(minimumScore, scores[token]);
      } else if (type == 4 && !vocab[token].isEmpty()) {
        userDefined.add(vocab[token].getBytes(StandardCharsets.UTF_8));
      }
    }
    userDefined.sort((left, right) -> Integer.compare(right.length, left.length));
    this.userDefinedTokens = List.copyOf(userDefined);
    this.charsMap = charsMap.clone();
    if (charsMap.length == 0) {
      this.xcda = new int[0];
      this.replacementsOffset = 0;
    } else {
      if (charsMap.length < Integer.BYTES) {
        throw new IllegalArgumentException("precompiled charsmap is smaller than its header");
      }
      int xcdaBytes = littleEndianInt(charsMap, 0);
      if (xcdaBytes < 0
          || xcdaBytes % Integer.BYTES != 0
          || Integer.BYTES + xcdaBytes >= charsMap.length) {
        throw new IllegalArgumentException("invalid precompiled charsmap XCDA length");
      }
      this.xcda = new int[xcdaBytes / Integer.BYTES];
      for (int index = 0; index < xcda.length; index++) {
        xcda[index] = littleEndianInt(charsMap, Integer.BYTES + index * Integer.BYTES);
      }
      this.replacementsOffset = Integer.BYTES + xcdaBytes;
    }
    this.addSpacePrefix = addSpacePrefix;
    this.removeExtraWhitespaces = removeExtraWhitespaces;
    this.unknownTokenId = unknownTokenId;
    this.unknownTokenScore =
        (Double.isFinite(minimumScore) ? minimumScore : 0.0) - UNKNOWN_SCORE_PENALTY;
    this.maxTokenChars = longestToken;
  }

  int[] encode(String text) {
    String normalized = new String(normalize(text), StandardCharsets.UTF_8);
    if (normalized.isEmpty()) {
      return new int[0];
    }

    int length = normalized.length();
    double[] bestScore = new double[length + 1];
    Arrays.fill(bestScore, Double.NEGATIVE_INFINITY);
    int[] bestToken = new int[length + 1];
    int[] previous = new int[length + 1];
    Arrays.fill(bestToken, unknownTokenId);
    bestScore[0] = 0.0;

    for (int offset = 0; offset < length; ) {
      int nextCodePoint = offset + Character.charCount(normalized.codePointAt(offset));
      if (Double.isFinite(bestScore[offset])) {
        boolean singleCodePointTokenFound = false;
        int maximumEnd = Math.min(length, offset + maxTokenChars);
        for (int end = nextCodePoint; end <= maximumEnd; ) {
          Integer token = tokenToId.get(normalized.substring(offset, end));
          if (token != null) {
            if (end == nextCodePoint) {
              singleCodePointTokenFound = true;
            }
            double tokenScore = tokenTypes[token] == 4 ? 0.0 : scores[token];
            double candidateScore = bestScore[offset] + tokenScore;
            if (candidateScore > bestScore[end]) {
              bestScore[end] = candidateScore;
              bestToken[end] = token;
              previous[end] = offset;
            }
          }
          if (end == length) {
            break;
          }
          end += Character.charCount(normalized.codePointAt(end));
        }
        if (!singleCodePointTokenFound) {
          double candidateScore = bestScore[offset] + unknownTokenScore;
          if (candidateScore > bestScore[nextCodePoint]) {
            bestScore[nextCodePoint] = candidateScore;
            bestToken[nextCodePoint] = unknownTokenId;
            previous[nextCodePoint] = offset;
          }
        }
      }
      offset = nextCodePoint;
    }

    List<Integer> reversed = new ArrayList<>();
    boolean previousWasUnknown = false;
    for (int offset = length; offset > 0; offset = previous[offset]) {
      int token = bestToken[offset];
      boolean unknown = token == unknownTokenId;
      if (!(previousWasUnknown && unknown)) {
        reversed.add(token);
      }
      previousWasUnknown = unknown;
    }
    Collections.reverse(reversed);
    return reversed.stream().mapToInt(Integer::intValue).toArray();
  }

  private byte[] normalize(String text) {
    byte[] input = text.getBytes(StandardCharsets.UTF_8);
    ByteArrayOutputStream normalized = new ByteArrayOutputStream(input.length * 2);
    boolean spacePrepended = false;
    boolean processingNonWhitespace = false;
    for (int offset = 0; offset < input.length; ) {
      Prefix replacement = normalizePrefix(input, offset);
      for (byte value : replacement.bytes()) {
        if (value != ' ') {
          if (!processingNonWhitespace) {
            processingNonWhitespace = true;
            if ((addSpacePrefix && !spacePrepended) || removeExtraWhitespaces) {
              normalized.writeBytes(ESCAPED_SPACE);
              spacePrepended = true;
            }
          }
          normalized.write(value);
        } else {
          processingNonWhitespace = false;
          if (!removeExtraWhitespaces) {
            normalized.writeBytes(ESCAPED_SPACE);
          }
        }
      }
      offset += replacement.consumedBytes();
    }
    return normalized.toByteArray();
  }

  private Prefix normalizePrefix(byte[] input, int offset) {
    for (byte[] userDefined : userDefinedTokens) {
      if (startsWith(input, offset, userDefined)) {
        return new Prefix(userDefined, userDefined.length);
      }
    }

    int longestLength = 0;
    int longestReplacement = 0;
    if (xcda.length > 0) {
      int node = base(0);
      for (int current = offset; current < input.length; current++) {
        int value = Byte.toUnsignedInt(input[current]);
        if (value == 0) {
          break;
        }
        node ^= value;
        if (lcheck(node) != value) {
          break;
        }
        boolean leaf = leaf(node);
        node ^= base(node);
        if (leaf) {
          longestLength = current - offset + 1;
          longestReplacement = xcdaValue(node);
        }
      }
    }
    if (longestLength > 0) {
      int replacementStart = replacementsOffset + longestReplacement;
      if (replacementStart < replacementsOffset || replacementStart >= charsMap.length) {
        throw new IllegalArgumentException("precompiled charsmap replacement is out of bounds");
      }
      int replacementEnd = replacementStart;
      while (replacementEnd < charsMap.length && charsMap[replacementEnd] != 0) {
        replacementEnd++;
      }
      if (replacementEnd == charsMap.length) {
        throw new IllegalArgumentException("unterminated precompiled charsmap replacement");
      }
      return new Prefix(
          Arrays.copyOfRange(charsMap, replacementStart, replacementEnd), longestLength);
    }

    int codePointBytes = utf8CodePointLength(input[offset]);
    codePointBytes = Math.min(codePointBytes, input.length - offset);
    return new Prefix(Arrays.copyOfRange(input, offset, offset + codePointBytes), codePointBytes);
  }

  private int base(int index) {
    int packed = node(index);
    return (packed >>> 10) << (((packed & (1 << 9)) >>> 6));
  }

  private int lcheck(int index) {
    return node(index) & (0x80000000 | 0xff);
  }

  private boolean leaf(int index) {
    return ((node(index) >>> 8) & 1) != 0;
  }

  private int xcdaValue(int index) {
    return node(index) & 0x7fffffff;
  }

  private int node(int index) {
    if (index < 0 || index >= xcda.length) {
      throw new IllegalArgumentException("precompiled charsmap node is out of bounds: " + index);
    }
    return xcda[index];
  }

  private static int littleEndianInt(byte[] bytes, int offset) {
    return Byte.toUnsignedInt(bytes[offset])
        | Byte.toUnsignedInt(bytes[offset + 1]) << 8
        | Byte.toUnsignedInt(bytes[offset + 2]) << 16
        | Byte.toUnsignedInt(bytes[offset + 3]) << 24;
  }

  private static int utf8CodePointLength(byte first) {
    int value = Byte.toUnsignedInt(first);
    if ((value & 0x80) == 0) {
      return 1;
    }
    if ((value & 0xe0) == 0xc0) {
      return 2;
    }
    if ((value & 0xf0) == 0xe0) {
      return 3;
    }
    if ((value & 0xf8) == 0xf0) {
      return 4;
    }
    return 1;
  }

  private static boolean startsWith(byte[] input, int offset, byte[] prefix) {
    if (offset + prefix.length > input.length) {
      return false;
    }
    for (int index = 0; index < prefix.length; index++) {
      if (input[offset + index] != prefix[index]) {
        return false;
      }
    }
    return true;
  }

  private record Prefix(byte[] bytes, int consumedBytes) {}
}
