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
import com.fasterxml.jackson.core.exc.StreamReadException;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.gptoss.GptOssHuggingFaceConfig;
import com.integrallis.models.backend.purejava.huggingface.Qwen2HuggingFaceConfig;
import com.integrallis.models.backend.purejava.mobilemoe.MobileMoeHuggingFaceConfig;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Reads a supported Hugging Face {@code tokenizer.json} without a Python runtime. */
public final class HuggingFaceTokenizer {

  private static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();
  private static final String QWEN2_REGEX =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
          + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  private static final String GPT_OSS_REGEX =
      "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
          + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
          + "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
          + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
          + "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|"
          + "\\s+(?!\\S)|\\s+";
  private static final String LLAMA3_REGEX =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
          + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

  private HuggingFaceTokenizer() {}

  /** Creates the tokenizer declared beside a Hugging Face Qwen 2 Safetensors checkpoint. */
  public static Tokenizer fromQwen2(
      Path tokenizerJson, Path tokenizerConfigJson, Qwen2HuggingFaceConfig modelConfig)
      throws IOException {
    Objects.requireNonNull(tokenizerJson, "tokenizerJson");
    Objects.requireNonNull(tokenizerConfigJson, "tokenizerConfigJson");
    Objects.requireNonNull(modelConfig, "modelConfig");
    Definition definition =
        readDefinition(tokenizerJson, modelConfig.model().vocabSize(), Family.QWEN2);
    Settings settings = readSettings(tokenizerConfigJson);
    int bos = resolveToken(settings.bosToken(), modelConfig.bosTokenId(), definition, "BOS");
    int eos = resolveToken(settings.eosToken(), modelConfig.eosTokenId(), definition, "EOS");
    String unknownText =
        definition.unknownToken() != null ? definition.unknownToken() : settings.unknownToken();
    int unknown = unknownText == null ? 0 : resolveToken(unknownText, -1, definition, "unknown");
    return GgufTokenizer.fromByteLevelBpe(
        definition.vocabulary(),
        definition.merges(),
        definition.controlTokenIds(),
        bos,
        eos,
        settings.addBosToken(),
        settings.addEosToken(),
        unknown,
        definition.normalizeNfc(),
        definition.preTokenizerName());
  }

  /** Creates the tokenizer and Harmony control vocabulary declared by a GPT-OSS checkpoint. */
  public static Tokenizer fromGptOss(
      Path tokenizerJson, Path tokenizerConfigJson, GptOssHuggingFaceConfig modelConfig)
      throws IOException {
    Objects.requireNonNull(tokenizerJson, "tokenizerJson");
    Objects.requireNonNull(tokenizerConfigJson, "tokenizerConfigJson");
    Objects.requireNonNull(modelConfig, "modelConfig");
    Definition definition = readDefinition(tokenizerJson, modelConfig.vocabSize(), Family.GPT_OSS);
    Settings settings = readSettings(tokenizerConfigJson);
    int bos = resolveToken(settings.bosToken(), -1, definition, "BOS");
    int eos = resolveToken(settings.eosToken(), modelConfig.eosTokenId(), definition, "EOS");
    String unknownText =
        definition.unknownToken() != null ? definition.unknownToken() : settings.unknownToken();
    int unknown = unknownText == null ? 0 : resolveToken(unknownText, -1, definition, "unknown");
    return GgufTokenizer.fromByteLevelBpe(
        definition.vocabulary(),
        definition.merges(),
        definition.controlTokenIds(),
        bos,
        eos,
        settings.addBosToken(),
        settings.addEosToken(),
        unknown,
        definition.normalizeNfc(),
        definition.preTokenizerName());
  }

  /** Creates the Llama-3 byte-level tokenizer declared beside a MobileMoE checkpoint. */
  public static Tokenizer fromMobileMoe(
      Path tokenizerJson, Path tokenizerConfigJson, MobileMoeHuggingFaceConfig modelConfig)
      throws IOException {
    Objects.requireNonNull(tokenizerJson, "tokenizerJson");
    Objects.requireNonNull(tokenizerConfigJson, "tokenizerConfigJson");
    Objects.requireNonNull(modelConfig, "modelConfig");
    Definition definition =
        readDefinition(tokenizerJson, modelConfig.vocabSize(), Family.MOBILE_MOE);
    Settings settings = readSettings(tokenizerConfigJson);
    int bos = resolveToken(settings.bosToken(), -1, definition, "BOS");
    int eos = resolveToken(settings.eosToken(), -1, definition, "EOS");
    String unknownText =
        definition.unknownToken() != null ? definition.unknownToken() : settings.unknownToken();
    int unknown = unknownText == null ? 0 : resolveToken(unknownText, -1, definition, "unknown");
    return GgufTokenizer.fromByteLevelBpe(
        definition.vocabulary(),
        definition.merges(),
        definition.controlTokenIds(),
        bos,
        eos,
        settings.addBosToken(),
        settings.addEosToken(),
        unknown,
        definition.normalizeNfc(),
        definition.preTokenizerName());
  }

  private static Definition readDefinition(Path path, int modelVocabularySize, Family family)
      throws IOException {
    DefinitionFields fields = new DefinitionFields(modelVocabularySize, family);
    try (JsonParser parser = JSON.createParser(path.toFile())) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "tokenizer root");
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "tokenizer field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "added_tokens" -> readAddedTokens(parser, value, fields);
          case "decoder" -> fields.decoderType = readTypeObject(parser, value, "decoder");
          case "model" -> readModel(parser, value, fields);
          case "normalizer" ->
              fields.normalizerType = readNullableTypeObject(parser, value, "normalizer");
          case "pre_tokenizer" -> readPreTokenizer(parser, value, fields);
          default -> parser.skipChildren();
        }
      }
      if (parser.nextToken() != null) {
        throw malformed("tokenizer has content after its root object");
      }
    } catch (StreamReadException malformed) {
      throw malformed("invalid or duplicate tokenizer JSON: " + malformed.getMessage(), malformed);
    }
    return fields.build();
  }

  private static Settings readSettings(Path path) throws IOException {
    String bos = null;
    String eos = null;
    String unknown = null;
    boolean addBos = false;
    boolean addEos = false;
    try (JsonParser parser = JSON.createParser(path.toFile())) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "tokenizer config root");
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "tokenizer config field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "add_bos_token" -> addBos = readNullableBoolean(parser, value, name);
          case "add_eos_token" -> addEos = readNullableBoolean(parser, value, name);
          case "bos_token" -> bos = readTokenText(parser, value, name);
          case "eos_token" -> eos = readTokenText(parser, value, name);
          case "unk_token" -> unknown = readTokenText(parser, value, name);
          default -> parser.skipChildren();
        }
      }
    } catch (StreamReadException malformed) {
      throw malformed(
          "invalid or duplicate tokenizer config JSON: " + malformed.getMessage(), malformed);
    }
    return new Settings(bos, eos, unknown, addBos, addEos);
  }

  private static void readModel(JsonParser parser, JsonToken token, DefinitionFields fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "tokenizer model");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "tokenizer model field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "byte_fallback" -> fields.byteFallback = readNullableBoolean(parser, value, name);
        case "continuing_subword_prefix" ->
            fields.continuingPrefix = readNullableString(parser, value, name);
        case "dropout" -> fields.hasDropout = value != JsonToken.VALUE_NULL;
        case "end_of_word_suffix" -> fields.endSuffix = readNullableString(parser, value, name);
        case "merges" -> fields.merges = readMerges(parser, value);
        case "type" -> fields.modelType = readString(parser, value, name);
        case "unk_token" -> fields.unknownToken = readNullableString(parser, value, name);
        case "vocab" -> readVocabulary(parser, value, fields);
        default -> parser.skipChildren();
      }
    }
  }

  private static void readVocabulary(JsonParser parser, JsonToken token, DefinitionFields fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "tokenizer vocabulary");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "vocabulary token");
      String text = parser.currentName();
      int id = readInt(parser, parser.nextToken(), "vocabulary id for " + text);
      fields.addToken(id, text);
    }
  }

  private static void readAddedTokens(JsonParser parser, JsonToken token, DefinitionFields fields)
      throws IOException {
    require(token, JsonToken.START_ARRAY, "added_tokens");
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      require(parser.currentToken(), JsonToken.START_OBJECT, "added token");
      Integer id = null;
      String content = null;
      boolean lstrip = false;
      boolean rstrip = false;
      boolean singleWord = false;
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "added token field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "content" -> content = readString(parser, value, name);
          case "id" -> id = readInt(parser, value, name);
          case "lstrip" -> lstrip = readNullableBoolean(parser, value, name);
          case "rstrip" -> rstrip = readNullableBoolean(parser, value, name);
          case "single_word" -> singleWord = readNullableBoolean(parser, value, name);
          default -> parser.skipChildren();
        }
      }
      if (id == null || content == null || content.isEmpty()) {
        throw malformed("added token requires a non-empty content and integer id");
      }
      if (lstrip || rstrip || singleWord) {
        throw malformed("Qwen 2 added-token whitespace and single-word modifiers are unsupported");
      }
      fields.addToken(id, content);
      // Added tokens are atomic in trusted templates even when tokenizers marks special=false.
      fields.controlTokenIds.add(id);
    }
  }

  private static void readPreTokenizer(JsonParser parser, JsonToken token, DefinitionFields fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "pre_tokenizer");
    String type = null;
    List<PreTokenizerPart> parts = List.of();
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "pre_tokenizer field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("type".equals(name)) {
        type = readString(parser, value, name);
      } else if ("pretokenizers".equals(name)) {
        parts = readPreTokenizerParts(parser, value);
      } else {
        parser.skipChildren();
      }
    }
    fields.acceptPreTokenizer(type, parts);
  }

  private static List<PreTokenizerPart> readPreTokenizerParts(JsonParser parser, JsonToken token)
      throws IOException {
    require(token, JsonToken.START_ARRAY, "pretokenizers");
    List<PreTokenizerPart> parts = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      require(parser.currentToken(), JsonToken.START_OBJECT, "pretokenizer");
      String type = null;
      String regex = null;
      String behavior = null;
      boolean invert = false;
      boolean addPrefixSpace = false;
      boolean trimOffsets = false;
      boolean useRegex = false;
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "pretokenizer field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "add_prefix_space" -> addPrefixSpace = readNullableBoolean(parser, value, name);
          case "behavior" -> behavior = readString(parser, value, name);
          case "invert" -> invert = readNullableBoolean(parser, value, name);
          case "pattern" -> regex = readRegex(parser, value);
          case "trim_offsets" -> trimOffsets = readNullableBoolean(parser, value, name);
          case "type" -> type = readString(parser, value, name);
          case "use_regex" -> useRegex = readNullableBoolean(parser, value, name);
          default -> parser.skipChildren();
        }
      }
      parts.add(
          new PreTokenizerPart(
              type, regex, behavior, invert, addPrefixSpace, trimOffsets, useRegex));
    }
    return List.copyOf(parts);
  }

  private static String readRegex(JsonParser parser, JsonToken token) throws IOException {
    require(token, JsonToken.START_OBJECT, "pretokenizer pattern");
    String regex = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "pretokenizer pattern field");
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

  private static String readNullableTypeObject(
      JsonParser parser, JsonToken token, String description) throws IOException {
    return token == JsonToken.VALUE_NULL ? null : readTypeObject(parser, token, description);
  }

  private static String readTypeObject(JsonParser parser, JsonToken token, String description)
      throws IOException {
    require(token, JsonToken.START_OBJECT, description);
    String type = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, description + " field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("type".equals(name)) {
        type = readString(parser, value, name);
      } else {
        parser.skipChildren();
      }
    }
    return type;
  }

  private static List<String> readMerges(JsonParser parser, JsonToken token) throws IOException {
    require(token, JsonToken.START_ARRAY, "merges");
    List<String> values = new ArrayList<>();
    Set<String> unique = new HashSet<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      String value;
      if (parser.currentToken() == JsonToken.VALUE_STRING) {
        value = parser.getText();
      } else if (parser.currentToken() == JsonToken.START_ARRAY) {
        require(parser.nextToken(), JsonToken.VALUE_STRING, "merge left token");
        String left = parser.getText();
        require(parser.nextToken(), JsonToken.VALUE_STRING, "merge right token");
        String right = parser.getText();
        require(parser.nextToken(), JsonToken.END_ARRAY, "merge pair");
        value = left + " " + right;
      } else {
        throw malformed("merge entry must be a string or two-token array");
      }
      if (!unique.add(value)) {
        throw malformed("merges contains a duplicate entry: " + value);
      }
      values.add(value);
    }
    return List.copyOf(values);
  }

  private static String readTokenText(JsonParser parser, JsonToken token, String name)
      throws IOException {
    if (token == JsonToken.VALUE_NULL) {
      return null;
    }
    if (token == JsonToken.VALUE_STRING) {
      return parser.getText();
    }
    require(token, JsonToken.START_OBJECT, name);
    String content = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, name + " field");
      String field = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("content".equals(field)) {
        content = readString(parser, value, field);
      } else {
        parser.skipChildren();
      }
    }
    return content;
  }

  private static String readString(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.VALUE_STRING, name);
    return parser.getText();
  }

  private static String readNullableString(JsonParser parser, JsonToken token, String name)
      throws IOException {
    return token == JsonToken.VALUE_NULL ? null : readString(parser, token, name);
  }

  private static int readInt(JsonParser parser, JsonToken token, String name) throws IOException {
    if (!token.isNumeric()) {
      throw malformed(name + " must be an integer");
    }
    return parser.getIntValue();
  }

  private static boolean readNullableBoolean(JsonParser parser, JsonToken token, String name) {
    if (token == JsonToken.VALUE_NULL) {
      return false;
    }
    if (token != JsonToken.VALUE_TRUE && token != JsonToken.VALUE_FALSE) {
      throw malformed(name + " must be boolean or null");
    }
    return token == JsonToken.VALUE_TRUE;
  }

  private static int resolveToken(
      String text, int fallback, Definition definition, String description) {
    if (text == null) {
      if (fallback >= 0 && fallback < definition.vocabulary().length) {
        return fallback;
      }
      throw malformed(description + " token is not declared");
    }
    Integer token = definition.tokenIds().get(text);
    if (token == null) {
      throw malformed(description + " token is absent from vocabulary: " + text);
    }
    return token;
  }

  private static void require(JsonToken actual, JsonToken expected, String description) {
    if (actual != expected) {
      throw malformed(description + " must be " + expected + "; got " + actual);
    }
  }

  private static IllegalArgumentException malformed(String message) {
    return new IllegalArgumentException(message);
  }

  private static IllegalArgumentException malformed(String message, Throwable cause) {
    return new IllegalArgumentException(message, cause);
  }

  private record Definition(
      String[] vocabulary,
      Map<String, Integer> tokenIds,
      List<String> merges,
      Set<Integer> controlTokenIds,
      String unknownToken,
      boolean normalizeNfc,
      String preTokenizerName) {

    private Definition {
      vocabulary = vocabulary.clone();
      tokenIds = Map.copyOf(tokenIds);
      merges = List.copyOf(merges);
      controlTokenIds = Set.copyOf(controlTokenIds);
    }

    @Override
    public String[] vocabulary() {
      return vocabulary.clone();
    }
  }

  private record Settings(
      String bosToken,
      String eosToken,
      String unknownToken,
      boolean addBosToken,
      boolean addEosToken) {}

  private record PreTokenizerPart(
      String type,
      String regex,
      String behavior,
      boolean invert,
      boolean addPrefixSpace,
      boolean trimOffsets,
      boolean useRegex) {

    private boolean isSplit(Family family) {
      return "Split".equals(type)
          && family.regex().equals(regex)
          && "Isolated".equals(behavior)
          && !invert;
    }

    private boolean isByteLevel(Family family) {
      return "ByteLevel".equals(type)
          && !addPrefixSpace
          && trimOffsets == family.trimOffsets()
          && !useRegex;
    }
  }

  private enum Family {
    QWEN2(QWEN2_REGEX, false, "NFC", true, "qwen2"),
    GPT_OSS(GPT_OSS_REGEX, true, null, false, "gpt-oss"),
    MOBILE_MOE(LLAMA3_REGEX, true, null, false, "llama3");

    private final String regex;
    private final boolean trimOffsets;
    private final String normalizerType;
    private final boolean normalizeNfc;
    private final String preTokenizerName;

    Family(
        String regex,
        boolean trimOffsets,
        String normalizerType,
        boolean normalizeNfc,
        String preTokenizerName) {
      this.regex = regex;
      this.trimOffsets = trimOffsets;
      this.normalizerType = normalizerType;
      this.normalizeNfc = normalizeNfc;
      this.preTokenizerName = preTokenizerName;
    }

    private String regex() {
      return regex;
    }

    private boolean trimOffsets() {
      return trimOffsets;
    }
  }

  private static final class DefinitionFields {
    private final int modelVocabularySize;
    private final Family family;
    private final Map<Integer, String> tokens = new HashMap<>();
    private final Set<Integer> controlTokenIds = new HashSet<>();
    private boolean byteFallback;
    private String continuingPrefix;
    private String decoderType;
    private String endSuffix;
    private boolean hasDropout;
    private List<String> merges;
    private String modelType;
    private String normalizerType;
    private boolean supportedPreTokenizer;
    private String unknownToken;

    private DefinitionFields(int modelVocabularySize, Family family) {
      if (modelVocabularySize <= 0) {
        throw malformed("model vocabulary size must be positive");
      }
      this.modelVocabularySize = modelVocabularySize;
      this.family = Objects.requireNonNull(family, "family");
    }

    private void addToken(int id, String text) {
      if (id < 0 || id >= modelVocabularySize) {
        throw malformed(
            "token id " + id + " is outside model vocabulary size " + modelVocabularySize);
      }
      String previous = tokens.putIfAbsent(id, text);
      if (previous != null && !previous.equals(text)) {
        throw malformed("token id " + id + " is assigned to both " + previous + " and " + text);
      }
    }

    private Definition build() {
      if (!"BPE".equals(modelType)
          || byteFallback
          || hasDropout
          || (continuingPrefix != null && !continuingPrefix.isEmpty())
          || (endSuffix != null && !endSuffix.isEmpty())) {
        throw malformed("unsupported Hugging Face byte-level BPE model options");
      }
      if (!Objects.equals(family.normalizerType, normalizerType)
          || !"ByteLevel".equals(decoderType)
          || !supportedPreTokenizer) {
        throw malformed("unsupported byte-level normalization or pre-tokenizer pipeline");
      }
      if (merges == null || tokens.isEmpty()) {
        throw malformed("byte-level tokenizer requires vocabulary and merges");
      }
      int highestDeclared =
          tokens.keySet().stream().mapToInt(Integer::intValue).max().orElseThrow();
      String[] vocabulary = new String[modelVocabularySize];
      Map<String, Integer> tokenIds = new HashMap<>(tokens.size() * 2);
      for (int id = 0; id <= highestDeclared; id++) {
        String text = tokens.get(id);
        if (text == null) {
          throw malformed("tokenizer vocabulary has a gap at id " + id);
        }
        vocabulary[id] = text;
        Integer previous = tokenIds.put(text, id);
        if (previous != null) {
          throw malformed("token text is assigned to multiple ids: " + text);
        }
      }
      java.util.Arrays.fill(vocabulary, highestDeclared + 1, vocabulary.length, "");
      return new Definition(
          vocabulary,
          tokenIds,
          merges,
          controlTokenIds,
          unknownToken,
          family.normalizeNfc,
          family.preTokenizerName);
    }

    private void acceptPreTokenizer(String type, List<PreTokenizerPart> parts) {
      if (!"Sequence".equals(type)
          || parts.size() != 2
          || !parts.get(0).isSplit(family)
          || !parts.get(1).isByteLevel(family)) {
        throw malformed("unsupported " + family + " pre_tokenizer pipeline");
      }
      supportedPreTokenizer = true;
    }
  }
}
