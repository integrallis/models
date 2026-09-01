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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Tokenizer loaded from GGUF metadata. Supports GPT-2 byte-level BPE, Llama SentencePiece-style
 * score-ordered merges, and the legacy plain-BPE fallback.
 */
public final class GgufTokenizer implements Tokenizer {

  /** BERT sentence-pair tokens and their matching segment ids. Arrays are caller-owned. */
  public record TokenizedPair(int[] tokens, int[] tokenTypes) {

    public TokenizedPair {
      tokens = Objects.requireNonNull(tokens, "tokens").clone();
      tokenTypes = Objects.requireNonNull(tokenTypes, "tokenTypes").clone();
      if (tokens.length != tokenTypes.length) {
        throw new IllegalArgumentException("tokens and tokenTypes must have the same length");
      }
    }

    @Override
    public int[] tokens() {
      return tokens.clone();
    }

    @Override
    public int[] tokenTypes() {
      return tokenTypes.clone();
    }
  }

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
  private final boolean useGemma4Bpe;
  private final boolean useWordPiece;
  private final UnigramTokenizer unigramTokenizer;
  private final boolean addBosToken;
  private final boolean addEosToken;
  private final boolean addSpacePrefix;
  private final BpePreTokenizer bpePreTokenizer;
  private final int unknownTokenId;
  private final char[] byteToChar;
  private final int[] charToByte;
  private final boolean normalizeNfc;

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
      boolean useGemma4Bpe,
      boolean useWordPiece,
      UnigramTokenizer unigramTokenizer,
      boolean addBosToken,
      boolean addEosToken,
      boolean addSpacePrefix,
      BpePreTokenizer bpePreTokenizer,
      int unknownTokenId,
      boolean normalizeNfc) {
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
    this.useGemma4Bpe = useGemma4Bpe;
    this.useWordPiece = useWordPiece;
    this.unigramTokenizer = unigramTokenizer;
    this.addBosToken = addBosToken;
    this.addEosToken = addEosToken;
    this.addSpacePrefix = addSpacePrefix;
    this.bpePreTokenizer = bpePreTokenizer;
    this.unknownTokenId = unknownTokenId;
    this.normalizeNfc = normalizeNfc;
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

    int bosTokenId =
        metadata
            .getUint32("tokenizer.ggml.bos_token_id")
            .orElse(metadata.getUint32("tokenizer.ggml.cls_token_id").orElse(1));
    int eosTokenId =
        metadata
            .getUint32("tokenizer.ggml.eos_token_id")
            .orElse(
                metadata
                    .getUint32("tokenizer.ggml.seperator_token_id")
                    .orElse(metadata.getUint32("tokenizer.ggml.separator_token_id").orElse(2)));
    boolean[] endOfGenerationTokens =
        buildEndOfGenerationTokens(metadata, vocab, tokenToId, eosTokenId);
    List<SpecialToken> specialTokens = buildSpecialTokens(metadata, vocab);

    // Detect GPT-2 style byte-level BPE: check if the model uses "gpt2" or "bpe" tokenizer type,
    // or if the vocabulary contains the characteristic Ġ (U+0120) characters that indicate
    // the bytes_to_unicode mapping is in use
    String tokenizerModel = metadata.getString("tokenizer.ggml.model").orElse("");
    boolean useSentencePiece = "llama".equals(tokenizerModel);
    boolean useGemma4Bpe = "gemma4".equals(tokenizerModel);
    boolean useWordPiece = "bert".equals(tokenizerModel);
    boolean useUnigram = "t5".equals(tokenizerModel);
    boolean useByteLevel =
        !useSentencePiece
            && !useGemma4Bpe
            && !useWordPiece
            && !useUnigram
            && detectByteLevel(tokenizerModel, vocab, tokenToId);
    boolean addBosToken =
        metadata.getBool("tokenizer.ggml.add_bos_token").orElse(useSentencePiece || useWordPiece);
    boolean addEosToken =
        metadata.getBool("tokenizer.ggml.add_eos_token").orElse(useWordPiece || useUnigram);
    boolean addSpacePrefix =
        metadata.getBool("tokenizer.ggml.add_space_prefix").orElse(useSentencePiece);
    int unknownTokenId = metadata.getUint32("tokenizer.ggml.unknown_token_id").orElse(0);
    UnigramTokenizer unigramTokenizer =
        useUnigram
            ? new UnigramTokenizer(
                vocab,
                scores,
                metadata.getInt32Array("tokenizer.ggml.token_type").orElse(List.of()),
                metadata.getByteArray("tokenizer.ggml.precompiled_charsmap").orElse(new byte[0]),
                addSpacePrefix,
                metadata.getBool("tokenizer.ggml.remove_extra_whitespaces").orElse(false),
                unknownTokenId)
            : null;
    String preTokenizer = metadata.getString("tokenizer.ggml.pre").orElse("");
    BpePreTokenizer bpePreTokenizer = BpePreTokenizer.forName(preTokenizer);

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
        useGemma4Bpe,
        useWordPiece,
        unigramTokenizer,
        addBosToken,
        addEosToken,
        addSpacePrefix,
        bpePreTokenizer,
        unknownTokenId,
        false);
  }

  static GgufTokenizer fromByteLevelBpe(
      String[] vocab,
      List<String> merges,
      Set<Integer> controlTokenIds,
      int bosTokenId,
      int eosTokenId,
      boolean addBosToken,
      boolean addEosToken,
      int unknownTokenId,
      boolean normalizeNfc) {
    return fromByteLevelBpe(
        vocab,
        merges,
        controlTokenIds,
        bosTokenId,
        eosTokenId,
        addBosToken,
        addEosToken,
        unknownTokenId,
        normalizeNfc,
        "qwen2");
  }

  static GgufTokenizer fromByteLevelBpe(
      String[] vocab,
      List<String> merges,
      Set<Integer> controlTokenIds,
      int bosTokenId,
      int eosTokenId,
      boolean addBosToken,
      boolean addEosToken,
      int unknownTokenId,
      boolean normalizeNfc,
      String preTokenizerName) {
    Objects.requireNonNull(vocab, "vocab");
    Objects.requireNonNull(merges, "merges");
    Objects.requireNonNull(controlTokenIds, "controlTokenIds");
    String[] copiedVocab = vocab.clone();
    Map<String, Integer> tokenToId = new HashMap<>(copiedVocab.length * 2);
    for (int token = 0; token < copiedVocab.length; token++) {
      String text = Objects.requireNonNull(copiedVocab[token], "vocabulary token " + token);
      if (text.isEmpty()) {
        continue;
      }
      Integer previous = tokenToId.put(text, token);
      if (previous != null) {
        throw new IllegalArgumentException(
            "duplicate vocabulary token " + text + " at ids " + previous + " and " + token);
      }
    }
    Map<String, Integer> mergeRanks = new HashMap<>(merges.size() * 2);
    for (int rank = 0; rank < merges.size(); rank++) {
      mergeRanks.put(merges.get(rank), rank);
    }
    boolean[] endOfGenerationTokens = new boolean[copiedVocab.length];
    markEndOfGeneration(endOfGenerationTokens, eosTokenId);
    for (int token = 0; token < copiedVocab.length; token++) {
      if (END_OF_GENERATION_TOKEN_TEXTS.contains(copiedVocab[token])) {
        endOfGenerationTokens[token] = true;
      }
    }
    clearMessageBoundaryEndTokens(endOfGenerationTokens, tokenToId);
    List<SpecialToken> controlTokens =
        controlTokenIds.stream()
            .map(
                token -> {
                  if (token < 0 || token >= copiedVocab.length) {
                    throw new IllegalArgumentException(
                        "control token id is outside vocabulary: " + token);
                  }
                  if (copiedVocab[token].isEmpty()) {
                    throw new IllegalArgumentException(
                        "control token has no vocabulary text: " + token);
                  }
                  return new SpecialToken(copiedVocab[token], token);
                })
            .sorted(
                Comparator.comparingInt((SpecialToken token) -> token.text().length()).reversed())
            .toList();
    return new GgufTokenizer(
        copiedVocab,
        new float[copiedVocab.length],
        tokenToId,
        mergeRanks,
        bosTokenId,
        eosTokenId,
        endOfGenerationTokens,
        controlTokens,
        true,
        false,
        false,
        false,
        null,
        addBosToken,
        addEosToken,
        false,
        BpePreTokenizer.forName(Objects.requireNonNull(preTokenizerName, "preTokenizerName")),
        unknownTokenId,
        normalizeNfc);
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

    clearMessageBoundaryEndTokens(result, tokenToId);
    if (tokenToId.containsKey("<|tool_response>")) {
      clearEndOfGeneration(result, tokenToId.get("</s>"));
    }
    return result;
  }

  private static void clearMessageBoundaryEndTokens(
      boolean[] endOfGenerationTokens, Map<String, Integer> tokenToId) {
    boolean harmonyOrSolar =
        (tokenToId.containsKey("<|return|>") && tokenToId.containsKey("<|call|>"))
            || (tokenToId.containsKey("<|calls|>") && tokenToId.containsKey("<|flush|>"));
    if (harmonyOrSolar) {
      clearEndOfGeneration(endOfGenerationTokens, tokenToId.get("<|end|>"));
    }
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
    return encode(ModelPrompt.text(Objects.requireNonNull(text, "text")));
  }

  @Override
  public int[] encode(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    List<Integer> encoded = new ArrayList<>();
    StringBuilder ordinaryText = new StringBuilder();
    for (ModelPrompt.Segment segment : prompt.segments()) {
      if (segment.kind() == ModelPrompt.SegmentKind.CONTROL) {
        appendTrustedControl(encoded, ordinaryText, segment.text());
      } else {
        ordinaryText.append(segment.text());
      }
    }
    flushOrdinaryText(encoded, ordinaryText);
    return addConfiguredBoundaryTokens(encoded.stream().mapToInt(Integer::intValue).toArray());
  }

  /**
   * Encodes a BERT cross-encoder pair as {@code [CLS] first [SEP] second [SEP]}.
   *
   * <p>When the pair exceeds {@code maxTokens}, tokens are removed from the longer side first. This
   * is the longest-first truncation used by the model's reference tokenizer. Segment id {@code 0}
   * covers the first sentence and its separator; id {@code 1} covers the second sentence and its
   * separator.
   */
  public TokenizedPair encodePair(String first, String second, int maxTokens) {
    Objects.requireNonNull(first, "first");
    Objects.requireNonNull(second, "second");
    if (!useWordPiece) {
      throw new IllegalStateException(
          "sentence-pair encoding currently requires a WordPiece model");
    }
    if (maxTokens < 3) {
      throw new IllegalArgumentException(
          "maxTokens must leave room for [CLS] and two [SEP] tokens");
    }

    List<Integer> firstTokens = mutableTokens(encodeWordPiece(first));
    List<Integer> secondTokens = mutableTokens(encodeWordPiece(second));
    int contentBudget = maxTokens - 3;
    while (firstTokens.size() + secondTokens.size() > contentBudget) {
      List<Integer> longer = firstTokens.size() >= secondTokens.size() ? firstTokens : secondTokens;
      longer.remove(longer.size() - 1);
    }

    int length = firstTokens.size() + secondTokens.size() + 3;
    int[] tokens = new int[length];
    int[] tokenTypes = new int[length];
    int position = 0;
    tokens[position++] = bosTokenId;
    for (int token : firstTokens) {
      tokens[position++] = token;
    }
    tokens[position++] = eosTokenId;
    for (int token : secondTokens) {
      tokens[position] = token;
      tokenTypes[position++] = 1;
    }
    tokens[position] = eosTokenId;
    tokenTypes[position] = 1;
    return new TokenizedPair(tokens, tokenTypes);
  }

  private static List<Integer> mutableTokens(int[] tokens) {
    List<Integer> result = new ArrayList<>(tokens.length);
    for (int token : tokens) {
      result.add(token);
    }
    return result;
  }

  private void appendTrustedControl(
      List<Integer> encoded, StringBuilder ordinaryText, String controlText) {
    int position = 0;
    while (position < controlText.length()) {
      SpecialToken nextSpecial = null;
      int nextSpecialIndex = -1;
      for (SpecialToken candidate : specialTokens) {
        int candidateIndex = controlText.indexOf(candidate.text(), position);
        if (candidateIndex >= 0 && (nextSpecialIndex < 0 || candidateIndex < nextSpecialIndex)) {
          nextSpecial = candidate;
          nextSpecialIndex = candidateIndex;
        }
      }

      if (nextSpecial == null) {
        ordinaryText.append(controlText, position, controlText.length());
        break;
      }
      if (nextSpecialIndex > position) {
        ordinaryText.append(controlText, position, nextSpecialIndex);
      }
      flushOrdinaryText(encoded, ordinaryText);
      encoded.add(nextSpecial.id());
      position = nextSpecialIndex + nextSpecial.text().length();
    }
  }

  private void flushOrdinaryText(List<Integer> encoded, StringBuilder ordinaryText) {
    if (ordinaryText.isEmpty()) {
      return;
    }
    append(encoded, encodeOrdinaryText(ordinaryText.toString()));
    ordinaryText.setLength(0);
  }

  private int[] encodeOrdinaryText(String text) {
    if (text.isEmpty()) {
      return new int[0];
    }
    String normalized = normalizeNfc ? Normalizer.normalize(text, Normalizer.Form.NFC) : text;
    if (useSentencePiece) {
      return encodeSentencePiece(normalized);
    }
    if (useGemma4Bpe) {
      return encodeGemma4Bpe(normalized);
    }
    if (useWordPiece) {
      return encodeWordPiece(normalized);
    }
    if (unigramTokenizer != null) {
      return unigramTokenizer.encode(normalized);
    }
    return useByteLevel ? encodeByteLevelBpe(normalized) : encodePlainBpe(normalized);
  }

  private int[] encodeWordPiece(String text) {
    List<Integer> encoded = new ArrayList<>();
    for (String word : wordPieceWords(text)) {
      int before = encoded.size();
      String candidate = "\u2581" + word;
      int[] boundaries =
          candidate.codePoints().map(codePoint -> Character.charCount(codePoint)).toArray();
      int[] offsets = new int[boundaries.length + 1];
      for (int index = 0; index < boundaries.length; index++) {
        offsets[index + 1] = offsets[index] + boundaries[index];
      }

      for (int start = 0; start < boundaries.length; ) {
        Integer token = null;
        int matchedEnd = -1;
        for (int end = boundaries.length; end > start; end--) {
          token = tokenToId.get(candidate.substring(offsets[start], offsets[end]));
          if (token != null) {
            matchedEnd = end;
            break;
          }
        }
        if (matchedEnd < 0) {
          while (encoded.size() > before) {
            encoded.remove(encoded.size() - 1);
          }
          encoded.add(unknownTokenId);
          break;
        }
        encoded.add(token);
        start = matchedEnd;
      }
    }
    return encoded.stream().mapToInt(Integer::intValue).toArray();
  }

  private static List<String> wordPieceWords(String text) {
    String decomposed = Normalizer.normalize(text, Normalizer.Form.NFD).toLowerCase(Locale.ROOT);
    List<String> words = new ArrayList<>();
    StringBuilder word = new StringBuilder();
    decomposed
        .codePoints()
        .forEach(
            codePoint -> {
              int type = Character.getType(codePoint);
              if (isAccentMark(type) || codePoint == 0 || codePoint == 0xfffd || isControl(type)) {
                return;
              }
              if (Character.isWhitespace(codePoint)) {
                finishWord(words, word);
              } else if (isWordPieceBoundary(codePoint, type)) {
                finishWord(words, word);
                words.add(new String(Character.toChars(codePoint)));
              } else {
                word.appendCodePoint(codePoint);
              }
            });
    finishWord(words, word);
    return words;
  }

  private static boolean isAccentMark(int type) {
    return type == Character.NON_SPACING_MARK
        || type == Character.COMBINING_SPACING_MARK
        || type == Character.ENCLOSING_MARK;
  }

  private static boolean isControl(int type) {
    return type == Character.CONTROL || type == Character.FORMAT;
  }

  private static boolean isWordPieceBoundary(int codePoint, int type) {
    boolean punctuation =
        type == Character.CONNECTOR_PUNCTUATION
            || type == Character.DASH_PUNCTUATION
            || type == Character.START_PUNCTUATION
            || type == Character.END_PUNCTUATION
            || type == Character.INITIAL_QUOTE_PUNCTUATION
            || type == Character.FINAL_QUOTE_PUNCTUATION
            || type == Character.OTHER_PUNCTUATION;
    boolean asciiSymbol =
        codePoint < 0x7f
            && (type == Character.MATH_SYMBOL
                || type == Character.CURRENCY_SYMBOL
                || type == Character.MODIFIER_SYMBOL
                || type == Character.OTHER_SYMBOL);
    return punctuation || asciiSymbol || isChineseCharacter(codePoint);
  }

  private static boolean isChineseCharacter(int codePoint) {
    return (codePoint >= 0x4e00 && codePoint <= 0x9fff)
        || (codePoint >= 0x3400 && codePoint <= 0x4dbf)
        || (codePoint >= 0x20000 && codePoint <= 0x2a6df)
        || (codePoint >= 0x2a700 && codePoint <= 0x2b73f)
        || (codePoint >= 0x2b740 && codePoint <= 0x2b81f)
        || (codePoint >= 0x2b920 && codePoint <= 0x2ceaf)
        || (codePoint >= 0xf900 && codePoint <= 0xfaff)
        || (codePoint >= 0x2f800 && codePoint <= 0x2fa1f);
  }

  private static void finishWord(List<String> words, StringBuilder word) {
    if (!word.isEmpty()) {
      words.add(word.toString());
      word.setLength(0);
    }
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

  private int[] encodeGemma4Bpe(String text) {
    String normalized = text.replace(' ', '\u2581');
    List<Integer> tokens = new ArrayList<>();
    int pieceStart = 0;
    while (pieceStart < normalized.length()) {
      boolean newlineRun = normalized.charAt(pieceStart) == '\n';
      int pieceEnd = pieceStart + 1;
      while (pieceEnd < normalized.length()
          && (normalized.charAt(pieceEnd) == '\n') == newlineRun) {
        pieceEnd++;
      }
      encodeGemma4Piece(normalized.substring(pieceStart, pieceEnd), newlineRun, tokens);
      pieceStart = pieceEnd;
    }
    return tokens.stream().mapToInt(Integer::intValue).toArray();
  }

  private void encodeGemma4Piece(String piece, boolean newlineRun, List<Integer> output) {
    if (newlineRun) {
      Integer wholeRun = tokenToId.get(piece);
      if (wholeRun != null) {
        output.add(wholeRun);
        return;
      }
    }

    List<RawBpeSymbol> symbols = new ArrayList<>();
    piece
        .codePoints()
        .forEach(
            codePoint -> {
              int index = symbols.size();
              symbols.add(
                  new RawBpeSymbol(new String(Character.toChars(codePoint)), index - 1, -1));
              if (index > 0) {
                symbols.get(index - 1).next = index;
              }
            });
    if (symbols.isEmpty()) {
      return;
    }

    PriorityQueue<RawBpeMerge> workQueue =
        new PriorityQueue<>(
            Comparator.comparingInt(RawBpeMerge::rank).thenComparingInt(RawBpeMerge::left));
    for (int right = 1; right < symbols.size(); right++) {
      addRawBpeMerge(workQueue, symbols, right - 1, right);
    }

    while (!workQueue.isEmpty()) {
      RawBpeMerge merge = workQueue.remove();
      RawBpeSymbol left = symbols.get(merge.left());
      RawBpeSymbol right = symbols.get(merge.right());
      if (!left.active
          || !right.active
          || left.next != merge.right()
          || left.version != merge.leftVersion()
          || right.version != merge.rightVersion()) {
        continue;
      }

      left.text += right.text;
      left.version++;
      left.next = right.next;
      right.active = false;
      if (right.next >= 0) {
        symbols.get(right.next).previous = merge.left();
      }
      addRawBpeMerge(workQueue, symbols, left.previous, merge.left());
      addRawBpeMerge(workQueue, symbols, merge.left(), left.next);
    }

    for (int index = 0; index >= 0; index = symbols.get(index).next) {
      String symbol = symbols.get(index).text;
      Integer token = tokenToId.get(symbol);
      if (token != null) {
        output.add(token);
      } else {
        appendByteFallback(output, symbol);
      }
    }
  }

  private void addRawBpeMerge(
      PriorityQueue<RawBpeMerge> workQueue,
      List<RawBpeSymbol> symbols,
      int leftIndex,
      int rightIndex) {
    if (leftIndex < 0 || rightIndex < 0) {
      return;
    }
    RawBpeSymbol left = symbols.get(leftIndex);
    RawBpeSymbol right = symbols.get(rightIndex);
    Integer rank = mergeRanks.get(left.text + " " + right.text);
    if (rank != null) {
      workQueue.add(new RawBpeMerge(leftIndex, rightIndex, rank, left.version, right.version));
    }
  }

  /**
   * GPT-2 style byte-level BPE encoding. Each byte of the UTF-8 input is mapped through
   * bytes_to_unicode to produce a Unicode string that the BPE vocabulary operates on.
   */
  private int[] encodeByteLevelBpe(String text) {
    List<Integer> tokens = new ArrayList<>();
    for (String piece : bpePreTokenizer.split(text)) {
      for (int token : encodeByteLevelBpePiece(piece)) {
        tokens.add(token);
      }
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

    if (bpePreTokenizer.ignoresMerges()) {
      Integer token = tokenToId.get(unicodeText);
      if (token != null) {
        return new int[] {token};
      }
    }

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
        tokens.add(byteId != null ? byteId : unknownTokenId);
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
        tokens.add(id != null ? id : unknownTokenId);
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

    List<BpeSymbol> symbols = new ArrayList<>(tokens.size());
    for (int index = 0; index < tokens.size(); index++) {
      symbols.add(
          new BpeSymbol(tokens.get(index), index - 1, index + 1 < tokens.size() ? index + 1 : -1));
    }
    PriorityQueue<BpeMerge> workQueue =
        new PriorityQueue<>(
            Comparator.comparingInt(BpeMerge::rank).thenComparingInt(BpeMerge::left));
    for (int right = 1; right < symbols.size(); right++) {
      addBpeMerge(workQueue, symbols, right - 1, right);
    }

    while (!workQueue.isEmpty()) {
      BpeMerge merge = workQueue.remove();
      BpeSymbol left = symbols.get(merge.left());
      BpeSymbol right = symbols.get(merge.right());
      if (!left.active
          || !right.active
          || left.next != merge.right()
          || left.version != merge.leftVersion()
          || right.version != merge.rightVersion()) {
        continue;
      }

      left.token = merge.mergedToken();
      left.version++;
      left.next = right.next;
      right.active = false;
      if (right.next >= 0) {
        symbols.get(right.next).previous = merge.left();
      }

      addBpeMerge(workQueue, symbols, left.previous, merge.left());
      addBpeMerge(workQueue, symbols, merge.left(), left.next);
    }

    List<Integer> result = new ArrayList<>();
    for (int index = 0; index >= 0; index = symbols.get(index).next) {
      result.add(symbols.get(index).token);
    }
    return result;
  }

  private void addBpeMerge(
      PriorityQueue<BpeMerge> workQueue, List<BpeSymbol> symbols, int leftIndex, int rightIndex) {
    if (leftIndex < 0 || rightIndex < 0) {
      return;
    }
    BpeSymbol left = symbols.get(leftIndex);
    BpeSymbol right = symbols.get(rightIndex);
    String leftText = vocab[left.token];
    String rightText = vocab[right.token];
    Integer rank = mergeRanks.get(leftText + " " + rightText);
    Integer mergedToken = tokenToId.get(leftText + rightText);
    if (rank != null && mergedToken != null) {
      workQueue.add(
          new BpeMerge(leftIndex, rightIndex, rank, mergedToken, left.version, right.version));
    }
  }

  @Override
  public String decode(int[] tokens) {
    if (useWordPiece) {
      return decodeWordPiece(tokens);
    }
    if (useSentencePiece) {
      return decodeSentencePiece(tokens);
    }
    if (unigramTokenizer != null) {
      return decodeSentencePiece(tokens);
    }
    if (useGemma4Bpe) {
      return decodeGemma4Bpe(tokens);
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

  private String decodeWordPiece(int[] tokens) {
    StringBuilder decoded = new StringBuilder();
    for (int token : tokens) {
      if (token < 0 || token >= vocab.length || token == bosTokenId || isEndOfGeneration(token)) {
        continue;
      }
      decoded.append(vocab[token].replace('\u2581', ' '));
    }
    return !decoded.isEmpty() && decoded.charAt(0) == ' '
        ? decoded.substring(1)
        : decoded.toString();
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

  private String decodeGemma4Bpe(int[] tokens) {
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
    return bytes.toString(StandardCharsets.UTF_8);
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

    if (useSentencePiece || useGemma4Bpe || useWordPiece || unigramTokenizer != null) {
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

  private static final class RawBpeSymbol {
    private String text;
    private int previous;
    private int next;
    private int version;
    private boolean active = true;

    private RawBpeSymbol(String text, int previous, int next) {
      this.text = text;
      this.previous = previous;
      this.next = next;
    }
  }

  private record RawBpeMerge(int left, int right, int rank, int leftVersion, int rightVersion) {}

  private static final class BpeSymbol {
    private int token;
    private int previous;
    private int next;
    private int version;
    private boolean active = true;

    private BpeSymbol(int token, int previous, int next) {
      this.token = token;
      this.previous = previous;
      this.next = next;
    }
  }

  private record BpeMerge(
      int left, int right, int rank, int mergedToken, int leftVersion, int rightVersion) {}

  private record SpecialToken(String text, int id) {}

  @Override
  public int tokenId(String text) {
    if (text == null) {
      return -1;
    }
    Integer id = tokenToId.get(text);
    return id == null ? -1 : id;
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

  @Override
  public boolean isEndOfGeneration(int token) {
    return token >= 0 && token < endOfGenerationTokens.length && endOfGenerationTokens[token];
  }
}
