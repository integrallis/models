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

/**
 * BPE tokenizer loaded from GGUF metadata. Reads vocabulary, scores, and merges from standard GGUF
 * keys.
 */
public final class GgufTokenizer implements Tokenizer {

  private final String[] vocab;
  private final float[] scores;
  private final Map<String, Integer> tokenToId;
  private final Map<Long, Integer> mergeRanks;
  private final int bosTokenId;
  private final int eosTokenId;

  private GgufTokenizer(
      String[] vocab,
      float[] scores,
      Map<String, Integer> tokenToId,
      Map<Long, Integer> mergeRanks,
      int bosTokenId,
      int eosTokenId) {
    this.vocab = vocab;
    this.scores = scores;
    this.tokenToId = tokenToId;
    this.mergeRanks = mergeRanks;
    this.bosTokenId = bosTokenId;
    this.eosTokenId = eosTokenId;
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

    return new GgufTokenizer(vocab, scores, tokenToId, mergeRanks, bosTokenId, eosTokenId);
  }

  @Override
  public int[] encode(String text) {
    if (text == null || text.isEmpty()) {
      return new int[0];
    }

    // Convert text to UTF-8 bytes and then to initial token sequence
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
    StringBuilder sb = new StringBuilder();
    for (int token : tokens) {
      sb.append(decode(token));
    }
    return sb.toString();
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
        return new String(new byte[] {(byte) byteVal}, StandardCharsets.ISO_8859_1);
      } catch (NumberFormatException e) {
        return piece;
      }
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
