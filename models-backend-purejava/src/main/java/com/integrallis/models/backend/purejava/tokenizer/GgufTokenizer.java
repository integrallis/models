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

import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * BPE tokenizer loaded from GGUF metadata. Reads vocabulary, scores, and merges from standard GGUF
 * keys. Supports GPT-2 style byte-level BPE where bytes 0-255 are mapped to printable Unicode
 * characters in the vocabulary.
 */
public final class GgufTokenizer implements Tokenizer {

  private final String[] vocab;
  private final float[] scores;
  private final Map<String, Integer> tokenToId;
  private final Map<Long, Integer> mergeRanks;
  private final int bosTokenId;
  private final int eosTokenId;
  private final boolean useByteLevel;
  private final char[] byteToChar;
  private final int[] charToByte;

  private GgufTokenizer(
      String[] vocab,
      float[] scores,
      Map<String, Integer> tokenToId,
      Map<Long, Integer> mergeRanks,
      int bosTokenId,
      int eosTokenId,
      boolean useByteLevel) {
    this.vocab = vocab;
    this.scores = scores;
    this.tokenToId = tokenToId;
    this.mergeRanks = mergeRanks;
    this.bosTokenId = bosTokenId;
    this.eosTokenId = eosTokenId;
    this.useByteLevel = useByteLevel;
    this.byteToChar = buildBytesToUnicode();
    this.charToByte = buildUnicodeToBytes(byteToChar);
  }

  /**
   * Builds the GPT-2 bytes_to_unicode mapping. Printable bytes (33-126, 161-172, 174-255) map to
   * themselves as Unicode chars; remaining bytes (0-32, 127-160, 173) map to chars starting at
   * U+0100.
   */
  private static char[] buildBytesToUnicode() {
    char[] mapping = new char[256];
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        mapping[b] = (char) b;
      } else {
        mapping[b] = (char) (256 + n);
        n++;
      }
    }
    return mapping;
  }

  /** Builds inverse mapping: Unicode char → byte value. */
  private static int[] buildUnicodeToBytes(char[] byteToChar) {
    // Find max char value to size the array
    int maxChar = 0;
    for (char c : byteToChar) {
      if (c > maxChar) maxChar = c;
    }
    int[] inverse = new int[maxChar + 1];
    java.util.Arrays.fill(inverse, -1);
    for (int b = 0; b < 256; b++) {
      inverse[byteToChar[b]] = b;
    }
    return inverse;
  }

  /** Creates a tokenizer from GGUF metadata. */
  public static GgufTokenizer fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");

    List<String> tokenList =
        metadata
            .getStringArray("tokenizer.ggml.tokens")
            .orElseThrow(() -> new IllegalArgumentException("Missing tokenizer.ggml.tokens"));

    String[] vocab = tokenList.toArray(new String[0]);

    float[] scores = new float[vocab.length];
    metadata
        .getFloat32Array("tokenizer.ggml.scores")
        .ifPresent(
            s -> {
              for (int i = 0; i < Math.min(s.size(), scores.length); i++) {
                scores[i] = s.get(i);
              }
            });

    Map<String, Integer> tokenToId = new HashMap<>(vocab.length * 2);
    for (int i = 0; i < vocab.length; i++) {
      tokenToId.put(vocab[i], i);
    }

    Map<Long, Integer> mergeRanks = new HashMap<>();
    metadata
        .getStringArray("tokenizer.ggml.merges")
        .ifPresent(
            merges -> {
              for (int i = 0; i < merges.size(); i++) {
                mergeRanks.put(mergeKey(merges.get(i)), i);
              }
            });

    int bosTokenId = metadata.getUint32("tokenizer.ggml.bos_token_id").orElse(1);
    int eosTokenId = metadata.getUint32("tokenizer.ggml.eos_token_id").orElse(2);

    // Detect GPT-2 style byte-level BPE: check if the model uses "gpt2" or "bpe" tokenizer type,
    // or if the vocabulary contains the characteristic Ġ (U+0120) characters that indicate
    // the bytes_to_unicode mapping is in use
    Optional<String> tokenizerModel = metadata.getString("tokenizer.ggml.model");
    boolean useByteLevel = detectByteLevel(tokenizerModel.orElse(""), vocab, tokenToId);

    return new GgufTokenizer(
        vocab, scores, tokenToId, mergeRanks, bosTokenId, eosTokenId, useByteLevel);
  }

  /**
   * Detect if this tokenizer uses GPT-2 byte-level BPE. Indicators:
   *
   * <ul>
   *   <li>tokenizer model is "gpt2"
   *   <li>vocabulary contains 'Ġ' (U+0120) which is the byte-level mapping for space (0x20)
   * </ul>
   */
  private static boolean detectByteLevel(
      String model, String[] vocab, Map<String, Integer> tokenToId) {
    if ("gpt2".equals(model)) {
      return true;
    }
    // Check if vocab contains the characteristic Ġ (bytes_to_unicode mapping for space)
    // This is the most reliable heuristic for byte-level BPE tokenizers
    return tokenToId.containsKey("\u0120");
  }

  @Override
  public int[] encode(String text) {
    if (text == null || text.isEmpty()) {
      return new int[0];
    }

    if (useByteLevel) {
      return encodeByteLevelBpe(text);
    } else {
      return encodePlainBpe(text);
    }
  }

  /**
   * GPT-2 style byte-level BPE encoding. Each byte of the UTF-8 input is mapped through
   * bytes_to_unicode to produce a Unicode string that the BPE vocabulary operates on.
   */
  private int[] encodeByteLevelBpe(String text) {
    byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);

    // Map each byte to its Unicode character representation
    StringBuilder mapped = new StringBuilder(textBytes.length);
    for (byte b : textBytes) {
      mapped.append(byteToChar[b & 0xFF]);
    }
    String unicodeText = mapped.toString();

    // Initial tokenization: each character is a separate token
    List<Integer> tokens = new ArrayList<>();
    for (int i = 0; i < unicodeText.length(); i++) {
      String ch = unicodeText.substring(i, i + 1);
      Integer id = tokenToId.get(ch);
      if (id != null) {
        tokens.add(id);
      } else {
        // Fallback: try <0xNN> byte token for the original byte
        String byteToken = String.format("<0x%02X>", textBytes[i] & 0xFF);
        Integer byteId = tokenToId.get(byteToken);
        tokens.add(byteId != null ? byteId : 0);
      }
    }

    // Apply BPE merges
    if (!mergeRanks.isEmpty()) {
      tokens = applyMerges(tokens);
    }

    return tokens.stream().mapToInt(Integer::intValue).toArray();
  }

  /**
   * Plain BPE encoding (non byte-level). Greedy longest-match against vocabulary, with byte token
   * fallback.
   */
  private int[] encodePlainBpe(String text) {
    byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
    List<Integer> tokens = new ArrayList<>();

    // Try to match longest tokens first (greedy), falling back to byte tokens
    int i = 0;
    while (i < textBytes.length) {
      int bestLen = 0;
      int bestId = -1;

      // Try matching substrings of decreasing length
      for (int len = Math.min(textBytes.length - i, 64); len >= 1; len--) {
        String candidate = new String(textBytes, i, len, StandardCharsets.UTF_8);
        Integer id = tokenToId.get(candidate);
        if (id != null) {
          bestLen = len;
          bestId = id;
          break;
        }
      }

      if (bestId >= 0) {
        tokens.add(bestId);
        i += bestLen;
      } else {
        // Fall back to byte-level token <0xNN>
        String byteToken = String.format("<0x%02X>", textBytes[i] & 0xFF);
        Integer id = tokenToId.get(byteToken);
        tokens.add(id != null ? id : 0);
        i += 1;
      }
    }

    // Apply BPE merges
    if (!mergeRanks.isEmpty()) {
      tokens = applyMerges(tokens);
    }

    return tokens.stream().mapToInt(Integer::intValue).toArray();
  }

  private List<Integer> applyMerges(List<Integer> tokens) {
    if (tokens.size() < 2) {
      return tokens;
    }

    List<Integer> result = new ArrayList<>(tokens);

    while (true) {
      int bestRank = Integer.MAX_VALUE;
      int bestIdx = -1;

      for (int i = 0; i < result.size() - 1; i++) {
        String pair = vocab[result.get(i)] + " " + vocab[result.get(i + 1)];
        Integer rank = mergeRanks.get(mergeKey(pair));
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }

      if (bestIdx < 0) {
        break;
      }

      // Merge the pair
      String merged = vocab[result.get(bestIdx)] + vocab[result.get(bestIdx + 1)];
      Integer mergedId = tokenToId.get(merged);
      if (mergedId == null) {
        break;
      }

      result.set(bestIdx, mergedId);
      result.remove(bestIdx + 1);
    }

    return result;
  }

  @Override
  public String decode(int[] tokens) {
    if (!useByteLevel) {
      StringBuilder sb = new StringBuilder();
      for (int token : tokens) {
        sb.append(decode(token));
      }
      return sb.toString();
    }

    // For byte-level BPE, we need to collect all bytes first and then decode as UTF-8
    List<Byte> byteList = new ArrayList<>();
    for (int token : tokens) {
      if (token < 0 || token >= vocab.length) {
        continue;
      }
      String piece = vocab[token];

      // Handle <0xNN> byte tokens
      if (piece.startsWith("<0x") && piece.endsWith(">") && piece.length() == 6) {
        try {
          int byteVal = Integer.parseInt(piece.substring(3, 5), 16);
          byteList.add((byte) byteVal);
          continue;
        } catch (NumberFormatException e) {
          // fall through to normal processing
        }
      }

      // Map each character back through unicode_to_bytes
      for (int i = 0; i < piece.length(); i++) {
        char c = piece.charAt(i);
        if (c < charToByte.length && charToByte[c] >= 0) {
          byteList.add((byte) charToByte[c]);
        } else {
          // Unknown char — encode it as UTF-8 directly
          byte[] charBytes = String.valueOf(c).getBytes(StandardCharsets.UTF_8);
          for (byte b : charBytes) {
            byteList.add(b);
          }
        }
      }
    }

    byte[] bytes = new byte[byteList.size()];
    for (int i = 0; i < byteList.size(); i++) {
      bytes[i] = byteList.get(i);
    }
    return new String(bytes, StandardCharsets.UTF_8);
  }

  @Override
  public String decode(int token) {
    if (token < 0 || token >= vocab.length) {
      return "";
    }
    String piece = vocab[token];

    // Handle byte-level fallback tokens like <0xNN>
    if (piece.startsWith("<0x") && piece.endsWith(">") && piece.length() == 6) {
      try {
        int byteVal = Integer.parseInt(piece.substring(3, 5), 16);
        return new String(new byte[] {(byte) byteVal}, StandardCharsets.UTF_8);
      } catch (NumberFormatException e) {
        return piece;
      }
    }

    if (useByteLevel) {
      // Map characters back through unicode_to_bytes
      List<Byte> byteList = new ArrayList<>();
      for (int i = 0; i < piece.length(); i++) {
        char c = piece.charAt(i);
        if (c < charToByte.length && charToByte[c] >= 0) {
          byteList.add((byte) charToByte[c]);
        } else {
          byte[] charBytes = String.valueOf(c).getBytes(StandardCharsets.UTF_8);
          for (byte b : charBytes) {
            byteList.add(b);
          }
        }
      }
      byte[] bytes = new byte[byteList.size()];
      for (int i = 0; i < byteList.size(); i++) {
        bytes[i] = byteList.get(i);
      }
      return new String(bytes, StandardCharsets.UTF_8);
    }

    return piece;
  }

  @Override
  public int vocabSize() {
    return vocab.length;
  }

  @Override
  public int bosToken() {
    return bosTokenId;
  }

  @Override
  public int eosToken() {
    return eosTokenId;
  }

  private static long mergeKey(String mergeStr) {
    return (long) mergeStr.hashCode();
  }
}
