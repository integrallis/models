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

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.StreamReadFeature;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Objects;

/** Reads the standard Hugging Face DeBERTa-v2 Unigram sentence-pair tokenizer. */
public final class DebertaV2Tokenizer {

  /** Pair tokens and segment identifiers in the model's expected template. */
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

  private static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();

  private final UnigramTokenizer unigram;
  private final int clsToken;
  private final int separatorToken;
  private final int maxTokens;

  private DebertaV2Tokenizer(
      UnigramTokenizer unigram, int clsToken, int separatorToken, int maxTokens) {
    this.unigram = Objects.requireNonNull(unigram, "unigram");
    this.clsToken = clsToken;
    this.separatorToken = separatorToken;
    if (maxTokens < 3) {
      throw new IllegalArgumentException("maxTokens must leave room for pair boundary tokens");
    }
    this.maxTokens = maxTokens;
  }

  /** Loads and validates the tokenizer pipeline encoded in {@code tokenizer.json}. */
  public static DebertaV2Tokenizer fromJson(Path path, int maxTokens) throws IOException {
    Objects.requireNonNull(path, "path");
    Fields fields = new Fields();
    try (JsonParser parser = JSON.createParser(path.toFile())) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "tokenizer root");
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "tokenizer field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "added_tokens" -> readAddedTokens(parser, value, fields);
          case "model" -> readModel(parser, value, fields);
          case "normalizer" -> readNormalizer(parser, value, fields);
          case "pre_tokenizer" -> readPreTokenizer(parser, value, fields);
          default -> parser.skipChildren();
        }
      }
      if (parser.nextToken() != null) {
        throw malformed("tokenizer has content after its root object");
      }
    }
    return fields.build(maxTokens);
  }

  /** Encodes {@code [CLS] query [SEP] document [SEP]} with longest-first truncation. */
  public TokenizedPair encodePair(String query, String document) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(document, "document");
    List<Integer> first = mutable(unigram.encode(query));
    List<Integer> second = mutable(unigram.encode(document));
    int contentBudget = maxTokens - 3;
    while (first.size() + second.size() > contentBudget) {
      List<Integer> longer = first.size() >= second.size() ? first : second;
      longer.remove(longer.size() - 1);
    }

    int[] tokens = new int[first.size() + second.size() + 3];
    int[] tokenTypes = new int[tokens.length];
    int position = 0;
    tokens[position++] = clsToken;
    for (int token : first) {
      tokens[position++] = token;
    }
    tokens[position++] = separatorToken;
    for (int token : second) {
      tokens[position] = token;
      tokenTypes[position++] = 1;
    }
    tokens[position] = separatorToken;
    tokenTypes[position] = 1;
    return new TokenizedPair(tokens, tokenTypes);
  }

  private static List<Integer> mutable(int[] tokens) {
    List<Integer> result = new ArrayList<>(tokens.length);
    for (int token : tokens) {
      result.add(token);
    }
    return result;
  }

  private static void readModel(JsonParser parser, JsonToken token, Fields fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "tokenizer model");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "tokenizer model field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "byte_fallback" -> fields.byteFallback = readBoolean(parser, value, name);
        case "type" -> fields.modelType = readString(parser, value, name);
        case "unk_id" -> fields.unknownToken = readInt(parser, value, name);
        case "vocab" -> readVocabulary(parser, value, fields);
        default -> parser.skipChildren();
      }
    }
  }

  private static void readVocabulary(JsonParser parser, JsonToken token, Fields fields)
      throws IOException {
    require(token, JsonToken.START_ARRAY, "Unigram vocabulary");
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      require(parser.currentToken(), JsonToken.START_ARRAY, "Unigram vocabulary entry");
      fields.vocabulary.add(readString(parser, parser.nextToken(), "Unigram token"));
      JsonToken score = parser.nextToken();
      if (!score.isNumeric()) {
        throw malformed("Unigram score must be numeric");
      }
      fields.scores.add(parser.getFloatValue());
      require(parser.nextToken(), JsonToken.END_ARRAY, "Unigram vocabulary entry");
    }
  }

  private static void readAddedTokens(JsonParser parser, JsonToken token, Fields fields)
      throws IOException {
    require(token, JsonToken.START_ARRAY, "added_tokens");
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      require(parser.currentToken(), JsonToken.START_OBJECT, "added token");
      Integer id = null;
      String content = null;
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "added token field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        if ("id".equals(name)) {
          id = readInt(parser, value, name);
        } else if ("content".equals(name)) {
          content = readString(parser, value, name);
        } else {
          parser.skipChildren();
        }
      }
      if ("[CLS]".equals(content)) {
        fields.clsToken = id;
      } else if ("[SEP]".equals(content)) {
        fields.separatorToken = id;
      } else if ("[UNK]".equals(content)) {
        fields.declaredUnknownToken = id;
      }
    }
  }

  private static void readNormalizer(JsonParser parser, JsonToken token, Fields fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "normalizer");
    String type = null;
    int supportedParts = 0;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "normalizer field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("type".equals(name)) {
        type = readString(parser, value, name);
      } else if ("normalizers".equals(name)) {
        require(value, JsonToken.START_ARRAY, "normalizers");
        while (parser.nextToken() != JsonToken.END_ARRAY) {
          supportedParts += readNormalizerPart(parser, fields);
        }
      } else {
        parser.skipChildren();
      }
    }
    fields.normalizerSupported = "Sequence".equals(type) && supportedParts == 3;
  }

  private static int readNormalizerPart(JsonParser parser, Fields fields) throws IOException {
    require(parser.currentToken(), JsonToken.START_OBJECT, "normalizer part");
    String type = null;
    Boolean stripLeft = null;
    Boolean stripRight = null;
    String charsMap = null;
    String replacementRegex = null;
    String replacementContent = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "normalizer part field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "content" -> replacementContent = readString(parser, value, name);
        case "precompiled_charsmap" -> charsMap = readString(parser, value, name);
        case "strip_left" -> stripLeft = readBoolean(parser, value, name);
        case "strip_right" -> stripRight = readBoolean(parser, value, name);
        case "type" -> type = readString(parser, value, name);
        case "pattern" -> replacementRegex = readRegex(parser, value);
        default -> parser.skipChildren();
      }
    }
    if ("Strip".equals(type) && Boolean.TRUE.equals(stripLeft) && Boolean.TRUE.equals(stripRight)) {
      return 1;
    }
    if ("Precompiled".equals(type) && charsMap != null) {
      fields.charsMap = Base64.getDecoder().decode(charsMap);
      return 1;
    }
    if ("Replace".equals(type)
        && " {2,}".equals(replacementRegex)
        && " ".equals(replacementContent)) {
      return 1;
    }
    return 0;
  }

  private static String readRegex(JsonParser parser, JsonToken token) throws IOException {
    require(token, JsonToken.START_OBJECT, "normalizer regex");
    String regex = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "normalizer regex field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("Regex".equals(name)) {
        regex = readString(parser, value, name);
      } else {
        parser.skipChildren();
      }
    }
    return regex;
  }

  private static void readPreTokenizer(JsonParser parser, JsonToken token, Fields fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "pre_tokenizer");
    String type = null;
    boolean metaspace = false;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "pre_tokenizer field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("type".equals(name)) {
        type = readString(parser, value, name);
      } else if ("pretokenizers".equals(name)) {
        require(value, JsonToken.START_ARRAY, "pretokenizers");
        int parts = 0;
        while (parser.nextToken() != JsonToken.END_ARRAY) {
          parts++;
          metaspace = readMetaspace(parser);
        }
        metaspace &= parts == 1;
      } else {
        parser.skipChildren();
      }
    }
    fields.preTokenizerSupported = "Sequence".equals(type) && metaspace;
  }

  private static boolean readMetaspace(JsonParser parser) throws IOException {
    require(parser.currentToken(), JsonToken.START_OBJECT, "Metaspace pre-tokenizer");
    String type = null;
    String replacement = null;
    String prependScheme = null;
    boolean addPrefixSpace = false;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "Metaspace field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "add_prefix_space" -> addPrefixSpace = readBoolean(parser, value, name);
        case "prepend_scheme" -> prependScheme = readString(parser, value, name);
        case "replacement" -> replacement = readString(parser, value, name);
        case "type" -> type = readString(parser, value, name);
        default -> parser.skipChildren();
      }
    }
    return "Metaspace".equals(type)
        && "▁".equals(replacement)
        && "always".equals(prependScheme)
        && addPrefixSpace;
  }

  private static String readString(JsonParser parser, JsonToken token, String description)
      throws IOException {
    require(token, JsonToken.VALUE_STRING, description);
    return parser.getText();
  }

  private static int readInt(JsonParser parser, JsonToken token, String description)
      throws IOException {
    if (!token.isNumeric()) {
      throw malformed(description + " must be an integer");
    }
    return parser.getIntValue();
  }

  private static boolean readBoolean(JsonParser parser, JsonToken token, String description) {
    if (token != JsonToken.VALUE_TRUE && token != JsonToken.VALUE_FALSE) {
      throw malformed(description + " must be boolean");
    }
    return token == JsonToken.VALUE_TRUE;
  }

  private static void require(JsonToken actual, JsonToken expected, String description) {
    if (actual != expected) {
      throw malformed(description + " must be " + expected + "; got " + actual);
    }
  }

  private static IllegalArgumentException malformed(String message) {
    return new IllegalArgumentException(message);
  }

  private static final class Fields {
    private final List<String> vocabulary = new ArrayList<>();
    private final List<Float> scores = new ArrayList<>();
    private boolean byteFallback;
    private Integer clsToken;
    private byte[] charsMap = new byte[0];
    private Integer declaredUnknownToken;
    private String modelType;
    private boolean normalizerSupported;
    private boolean preTokenizerSupported;
    private Integer separatorToken;
    private int unknownToken = -1;

    private DebertaV2Tokenizer build(int maxTokens) {
      if (!"Unigram".equals(modelType)
          || byteFallback
          || vocabulary.isEmpty()
          || vocabulary.size() != scores.size()) {
        throw malformed("unsupported DeBERTa-v2 Unigram model options");
      }
      if (!normalizerSupported || charsMap.length == 0 || !preTokenizerSupported) {
        throw malformed("unsupported DeBERTa-v2 normalization or pre-tokenizer pipeline");
      }
      if (clsToken == null || separatorToken == null || unknownToken < 0) {
        throw malformed("DeBERTa-v2 tokenizer is missing CLS, SEP, or unknown token ids");
      }
      if (!Objects.equals(declaredUnknownToken, unknownToken)) {
        throw malformed("added and model unknown token ids differ");
      }
      float[] tokenScores = new float[scores.size()];
      for (int index = 0; index < tokenScores.length; index++) {
        tokenScores[index] = scores.get(index);
      }
      UnigramTokenizer tokenizer =
          new UnigramTokenizer(
              vocabulary.toArray(String[]::new),
              tokenScores,
              List.of(),
              charsMap,
              true,
              true,
              unknownToken);
      return new DebertaV2Tokenizer(tokenizer, clsToken, separatorToken, maxTokens);
    }
  }
}
