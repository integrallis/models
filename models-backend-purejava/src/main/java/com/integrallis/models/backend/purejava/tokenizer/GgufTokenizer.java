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
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tokenizer loaded from GGUF metadata. Supports GPT-2 byte-level BPE, Llama SentencePiece-style
 * score-ordered merges, and the legacy plain-BPE fallback.
 */
public final class GgufTokenizer implements Tokenizer {

  private static final Pattern LLAMA3_PRETOKEN_PATTERN =
      Pattern.compile(
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
              + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
              + "|\\p{N}{1,3}"
              + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
              + "|\\s*[\\r\\n]+"
              + "|\\s+(?!\\S)"
              + "|\\s+");
  private static final Set<String> LLAMA3_PRETOKENIZERS =
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
          "smaug-bpe");
  private static final Set<String> END_OF_GENERATION_TOKEN_TEXTS =
      Set.of(
          "<|eot_id|>",
          "<|im_end|>",
          "<|end|>",
          "<|return|>",
          "<|call|>",
          "<|flush|>",
          "<|calls|>",
          "<end_of_turn>",
          "<|endoftext|>",
          "</s>",
          "<|eom_id|>",
          "<EOT>",
          "_<EOT>",
          "[EOT]",
          "[EOS]",
          "<|end_of_text|>",
          "<end_of_utterance>",
          "<eos>",
          "<turn|>",
          "<|tool_response>",
          "<｜end▁of▁sentence｜>");
  private static final List<String> END_OF_GENERATION_METADATA_KEYS =
      List.of(
          "tokenizer.ggml.eos_token_id",
          "tokenizer.ggml.eot_token_id",
          "tokenizer.ggml.eom_token_id",
          "tokenizer.ggml.fim_pad_token_id",
          "tokenizer.ggml.fim_rep_token_id",
          "tokenizer.ggml.fim_sep_token_id");

  private final String[] vocab;
  private final float[] scores;
  private final Map<String, Integer> tokenToId;
  private final Map<String, Integer> mergeRanks;
  private final int bosTokenId;
  private final int eosTokenId;
  private final boolean[] endOfGenerationTokens;
  private final List<SpecialToken> specialTokens;
  private final boolean useByteLevel;
  private final boolean useSentencePiece;
  private final boolean addBosToken;
  private final boolean addEosToken;
  private final boolean addSpacePrefix;
  private final boolean useLlama3PreTokenizer;
  private final int unknownTokenId;
  private final char[] byteToChar;
  private final int[] charToByte;

  private GgufTokenizer(
      String[] vocab,
      float[] scores,
      Map<String, Integer> tokenToId,
      Map<String, Integer> mergeRanks,
      int bosTokenId,
      int eosTokenId,
      boolean[] endOfGenerationTokens,
      List<SpecialToken> specialTokens,
      boolean useByteLevel,
      boolean useSentencePiece,
      boolean addBosToken,
      boolean addEosToken,
      boolean addSpacePrefix,
      boolean useLlama3PreTokenizer,
      int unknownTokenId) {
    this.vocab = vocab;
    this.scores = scores;
    this.tokenToId = tokenToId;
    this.mergeRanks = mergeRanks;
    this.bosTokenId = bosTokenId;
    this.eosTokenId = eosTokenId;
    this.endOfGenerationTokens = endOfGenerationTokens;
    this.specialTokens = specialTokens;
    this.useByteLevel = useByteLevel;
    this.useSentencePiece = useSentencePiece;
    this.addBosToken = addBosToken;
    this.addEosToken = addEosToken;
    this.addSpacePrefix = addSpacePrefix;
    this.useLlama3PreTokenizer = useLlama3PreTokenizer;
    this.unknownTokenId = unknownTokenId;
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

    Map<String, Integer> mergeRanks = new HashMap<>();
    metadata
        .getStringArray("tokenizer.ggml.merges")
        .ifPresent(
            merges -> {
              for (int i = 0; i < merges.size(); i++) {
                mergeRanks.put(merges.get(i), i);
              }
            });

    int bosTokenId = metadata.getUint32("tokenizer.ggml.bos_token_id").orElse(1);
    int eosTokenId = metadata.getUint32("tokenizer.ggml.eos_token_id").orElse(2);
    boolean[] endOfGenerationTokens =
        buildEndOfGenerationTokens(metadata, vocab, tokenToId, eosTokenId);
    List<SpecialToken> specialTokens = buildSpecialTokens(metadata, vocab);

    // Detect GPT-2 style byte-level BPE: check if the model uses "gpt2" or "bpe" tokenizer type,
    // or if the vocabulary contains the characteristic Ġ (U+0120) characters that indicate
    // the bytes_to_unicode mapping is in use
    String tokenizerModel = metadata.getString("tokenizer.ggml.model").orElse("");
    boolean useSentencePiece = "llama".equals(tokenizerModel);
    boolean useByteLevel = !useSentencePiece && detectByteLevel(tokenizerModel, vocab, tokenToId);
    boolean addBosToken = metadata.getBool("tokenizer.ggml.add_bos_token").orElse(useSentencePiece);
    boolean addEosToken = metadata.getBool("tokenizer.ggml.add_eos_token").orElse(false);
    boolean addSpacePrefix =
        metadata.getBool("tokenizer.ggml.add_space_prefix").orElse(useSentencePiece);
    String preTokenizer = metadata.getString("tokenizer.ggml.pre").orElse("");
    boolean useLlama3PreTokenizer = LLAMA3_PRETOKENIZERS.contains(preTokenizer);
    int unknownTokenId = metadata.getUint32("tokenizer.ggml.unknown_token_id").orElse(0);

    return new GgufTokenizer(
        vocab,
        scores,
        tokenToId,
        mergeRanks,
        bosTokenId,
        eosTokenId,
        endOfGenerationTokens,
        specialTokens,
        useByteLevel,
        useSentencePiece,
        addBosToken,
        addEosToken,
        addSpacePrefix,
        useLlama3PreTokenizer,
        unknownTokenId);
  }

  private static boolean[] buildEndOfGenerationTokens(
      GgufMetadata metadata, String[] vocab, Map<String, Integer> tokenToId, int eosTokenId) {
    boolean[] result = new boolean[vocab.length];
    markEndOfGeneration(result, eosTokenId);
    for (String key : END_OF_GENERATION_METADATA_KEYS) {
      metadata.getUint32(key).ifPresent(token -> markEndOfGeneration(result, token));
    }
    for (int token = 0; token < vocab.length; token++) {
      if (END_OF_GENERATION_TOKEN_TEXTS.contains(vocab[token])) {
        result[token] = true;
      }
    }

    boolean harmonyOrSolar =
        (tokenToId.containsKey("<|return|>") && tokenToId.containsKey("<|call|>"))
            || (tokenToId.containsKey("<|calls|>") && tokenToId.containsKey("<|flush|>"));
    if (harmonyOrSolar) {
      clearEndOfGeneration(result, tokenToId.get("<|end|>"));
    }
    if (tokenToId.containsKey("<|tool_response>")) {
      clearEndOfGeneration(result, tokenToId.get("</s>"));
    }
    return result;
  }

  private static List<SpecialToken> buildSpecialTokens(GgufMetadata metadata, String[] vocab) {
    boolean[] special = new boolean[vocab.length];
    for (int token = 0; token < vocab.length; token++) {
      special[token] = END_OF_GENERATION_TOKEN_TEXTS.contains(vocab[token]);
    }
    metadata
        .getInt32Array("tokenizer.ggml.token_type")
        .ifPresent(
            tokenTypes -> {
              int count = Math.min(tokenTypes.size(), special.length);
              for (int token = 0; token < count; token++) {
                int type = tokenTypes.get(token);
                if (type == 2 || type == 3 || type == 4) {
                  special[token] = true;
                }
              }
            });

    List<SpecialToken> result = new ArrayList<>();
    for (int token = 0; token < special.length; token++) {
      if (special[token] && !vocab[token].isEmpty()) {
        result.add(new SpecialToken(vocab[token], token));
      }
    }
    result.sort(Comparator.comparingInt((SpecialToken token) -> token.text().length()).reversed());
    return List.copyOf(result);
  }

  private static void markEndOfGeneration(boolean[] tokens, int token) {
    if (token >= 0 && token < tokens.length) {
      tokens[token] = true;
    }
  }

  private static void clearEndOfGeneration(boolean[] tokens, Integer token) {
    if (token != null && token >= 0 && token < tokens.length) {
      tokens[token] = false;
    }
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
    if (text == null) {
      return new int[0];
    }

    int[] encoded;
    if (text.isEmpty()) {
      encoded = new int[0];
    } else if (specialTokens.isEmpty()) {
      encoded = encodeOrdinaryText(text);
    } else {
      encoded = encodeWithSpecialTokens(text);
    }
    return addConfiguredBoundaryTokens(encoded);
  }

  private int[] encodeWithSpecialTokens(String text) {
    List<Integer> encoded = new ArrayList<>();
    int position = 0;
    while (position < text.length()) {
      SpecialToken nextSpecial = null;
      int nextSpecialIndex = -1;
      for (SpecialToken candidate : specialTokens) {
        int candidateIndex = text.indexOf(candidate.text(), position);
        if (candidateIndex >= 0 && (nextSpecialIndex < 0 || candidateIndex < nextSpecialIndex)) {
          nextSpecial = candidate;
          nextSpecialIndex = candidateIndex;
        }
      }

      if (nextSpecial == null) {
        append(encoded, encodeOrdinaryText(text.substring(position)));
        break;
      }
      if (nextSpecialIndex > position) {
        append(encoded, encodeOrdinaryText(text.substring(position, nextSpecialIndex)));
      }
      encoded.add(nextSpecial.id());
      position = nextSpecialIndex + nextSpecial.text().length();
    }
    return encoded.stream().mapToInt(Integer::intValue).toArray();
  }

  private int[] encodeOrdinaryText(String text) {
    if (text.isEmpty()) {
      return new int[0];
    }
    if (useSentencePiece) {
      return encodeSentencePiece(text);
    }
    return useByteLevel ? encodeByteLevelBpe(text) : encodePlainBpe(text);
  }

  private static void append(List<Integer> destination, int[] source) {
    for (int token : source) {
      destination.add(token);
    }
  }

  private int[] addConfiguredBoundaryTokens(int[] encoded) {
    int prefixLength = addBosToken ? 1 : 0;
    int suffixLength = addEosToken ? 1 : 0;
    if (prefixLength == 0 && suffixLength == 0) {
      return encoded;
    }

    int[] tokens = new int[prefixLength + encoded.length + suffixLength];
    if (addBosToken) {
      tokens[0] = bosTokenId;
    }
    System.arraycopy(encoded, 0, tokens, prefixLength, encoded.length);
    if (addEosToken) {
      tokens[tokens.length - 1] = eosTokenId;
    }
    return tokens;
  }

  private int[] encodeSentencePiece(String text) {
    List<Integer> tokens = new ArrayList<>();

    if (!text.isEmpty()) {
      String normalized = (addSpacePrefix ? " " : "") + text;
      normalized = normalized.replace(' ', '\u2581');

      List<String> symbols =
          normalized
              .codePoints()
              .mapToObj(codePoint -> new String(Character.toChars(codePoint)))
              .toList();
      symbols = applySentencePieceMerges(symbols);

      for (String symbol : symbols) {
        Integer token = tokenToId.get(symbol);
        if (token != null) {
          tokens.add(token);
        } else {
          appendByteFallback(tokens, symbol);
        }
      }
    }

    return tokens.stream().mapToInt(Integer::intValue).toArray();
  }

  private List<String> applySentencePieceMerges(List<String> initialSymbols) {
    if (initialSymbols.size() < 2) {
      return initialSymbols;
    }

    List<SentencePieceSymbol> symbols = new ArrayList<>(initialSymbols.size());
    for (int index = 0; index < initialSymbols.size(); index++) {
      symbols.add(
          new SentencePieceSymbol(
              initialSymbols.get(index),
              index - 1,
              index + 1 < initialSymbols.size() ? index + 1 : -1));
    }

    PriorityQueue<SentencePieceBigram> workQueue =
        new PriorityQueue<>(
            Comparator.comparingDouble(SentencePieceBigram::score)
                .reversed()
                .thenComparingInt(SentencePieceBigram::left));
    for (int right = 1; right < symbols.size(); right++) {
      addSentencePieceBigram(workQueue, symbols, right - 1, right);
    }

    while (!workQueue.isEmpty()) {
      SentencePieceBigram bigram = workQueue.remove();
      SentencePieceSymbol left = symbols.get(bigram.left());
      SentencePieceSymbol right = symbols.get(bigram.right());
      if (!left.active
          || !right.active
          || left.next != bigram.right()
          || left.version != bigram.leftVersion()
          || right.version != bigram.rightVersion()) {
        continue;
      }

      left.text += right.text;
      left.version++;
      left.next = right.next;
      right.active = false;
      if (right.next >= 0) {
        symbols.get(right.next).previous = bigram.left();
      }

      addSentencePieceBigram(workQueue, symbols, left.previous, bigram.left());
      addSentencePieceBigram(workQueue, symbols, bigram.left(), left.next);
    }

    List<String> merged = new ArrayList<>();
    for (int index = 0; index >= 0; index = symbols.get(index).next) {
      merged.add(symbols.get(index).text);
    }
    return merged;
  }

  private void addSentencePieceBigram(
      PriorityQueue<SentencePieceBigram> workQueue,
      List<SentencePieceSymbol> symbols,
      int leftIndex,
      int rightIndex) {
    if (leftIndex < 0 || rightIndex < 0) {
      return;
    }
    SentencePieceSymbol left = symbols.get(leftIndex);
    SentencePieceSymbol right = symbols.get(rightIndex);
    Integer token = tokenToId.get(left.text + right.text);
    if (token != null) {
      workQueue.add(
          new SentencePieceBigram(
              leftIndex, rightIndex, scores[token], left.version, right.version));
    }
  }

  private void appendByteFallback(List<Integer> tokens, String symbol) {
    for (byte value : symbol.getBytes(StandardCharsets.UTF_8)) {
      Integer token = tokenToId.get(String.format("<0x%02X>", value & 0xFF));
      tokens.add(token != null ? token : unknownTokenId);
    }
  }

  /**
   * GPT-2 style byte-level BPE encoding. Each byte of the UTF-8 input is mapped through
   * bytes_to_unicode to produce a Unicode string that the BPE vocabulary operates on.
   */
  private int[] encodeByteLevelBpe(String text) {
    if (!useLlama3PreTokenizer) {
      return encodeByteLevelBpePiece(text);
    }

    List<Integer> tokens = new ArrayList<>();
    Matcher matcher = LLAMA3_PRETOKEN_PATTERN.matcher(text);
    int matchedThrough = 0;
    while (matcher.find()) {
      if (matcher.start() != matchedThrough) {
        throw new IllegalArgumentException(
            "Tokenizer pre-pattern did not match input at index " + matchedThrough);
      }
      for (int token : encodeByteLevelBpePiece(matcher.group())) {
        tokens.add(token);
      }
      matchedThrough = matcher.end();
    }
    if (matchedThrough != text.length()) {
      throw new IllegalArgumentException(
          "Tokenizer pre-pattern did not match input at index " + matchedThrough);
    }
    return tokens.stream().mapToInt(Integer::intValue).toArray();
  }

  private int[] encodeByteLevelBpePiece(String text) {
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
        Integer rank = mergeRanks.get(pair);
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
    if (useSentencePiece) {
      return decodeSentencePiece(tokens);
    }
    if (!useByteLevel) {
      StringBuilder sb = new StringBuilder();
      for (int token : tokens) {
        if (token != bosTokenId && !isEndOfGeneration(token)) {
          sb.append(decode(token));
        }
      }
      return sb.toString();
    }

    // For byte-level BPE, we need to collect all bytes first and then decode as UTF-8
    List<Byte> byteList = new ArrayList<>();
    for (int token : tokens) {
      if (token < 0 || token >= vocab.length || token == bosTokenId || isEndOfGeneration(token)) {
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

  private String decodeSentencePiece(int[] tokens) {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    for (int token : tokens) {
      if (token < 0 || token >= vocab.length || token == bosTokenId || isEndOfGeneration(token)) {
        continue;
      }
      String piece = vocab[token];
      Integer byteValue = explicitByteValue(piece);
      if (byteValue != null) {
        bytes.write(byteValue);
      } else {
        bytes.writeBytes(piece.replace('\u2581', ' ').getBytes(StandardCharsets.UTF_8));
      }
    }

    String decoded = bytes.toString(StandardCharsets.UTF_8);
    if (addSpacePrefix && decoded.startsWith(" ")) {
      return decoded.substring(1);
    }
    return decoded;
  }

  @Override
  public String decode(int token) {
    if (token < 0 || token >= vocab.length || token == bosTokenId || isEndOfGeneration(token)) {
      return "";
    }
    String piece = vocab[token];

    // Handle byte-level fallback tokens like <0xNN>
    Integer byteValue = explicitByteValue(piece);
    if (byteValue != null) {
      return new String(new byte[] {byteValue.byteValue()}, StandardCharsets.UTF_8);
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

    if (useSentencePiece) {
      return piece.replace('\u2581', ' ');
    }

    return piece;
  }

  private static Integer explicitByteValue(String piece) {
    if (!piece.startsWith("<0x") || !piece.endsWith(">") || piece.length() != 6) {
      return null;
    }
    try {
      return Integer.parseInt(piece.substring(3, 5), 16);
    } catch (NumberFormatException e) {
      return null;
    }
  }

  private static final class SentencePieceSymbol {
    private String text;
    private int previous;
    private int next;
    private int version;
    private boolean active = true;

    private SentencePieceSymbol(String text, int previous, int next) {
      this.text = text;
      this.previous = previous;
      this.next = next;
    }
  }

  private record SentencePieceBigram(
      int left, int right, float score, int leftVersion, int rightVersion) {}

  private record SpecialToken(String text, int id) {}

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

  @Override
  public boolean isEndOfGeneration(int token) {
    return token >= 0 && token < endOfGenerationTokens.length && endOfGenerationTokens[token];
  }
}
