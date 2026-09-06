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
package com.integrallis.models.backend.purejava.deberta;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.StreamReadFeature;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Validated Hugging Face DeBERTa-v2 sequence-classification configuration. */
public record DebertaV2Config(
    int hiddenSize,
    int intermediateSize,
    int numLayers,
    int numHeads,
    int vocabSize,
    int maxPositions,
    int positionBuckets,
    int maxRelativePositions,
    int poolerHiddenSize,
    int padTokenId,
    float layerNormEpsilon,
    boolean relativeAttention,
    boolean shareAttentionKey,
    boolean positionBiasedInput) {

  private static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();

  public DebertaV2Config {
    requirePositive(hiddenSize, "hiddenSize");
    requirePositive(intermediateSize, "intermediateSize");
    requirePositive(numLayers, "numLayers");
    requirePositive(numHeads, "numHeads");
    requirePositive(vocabSize, "vocabSize");
    requirePositive(maxPositions, "maxPositions");
    requirePositive(positionBuckets, "positionBuckets");
    requirePositive(poolerHiddenSize, "poolerHiddenSize");
    if (hiddenSize % numHeads != 0) {
      throw new IllegalArgumentException("hiddenSize must be divisible by numHeads");
    }
    if (!(layerNormEpsilon > 0.0f) || !Float.isFinite(layerNormEpsilon)) {
      throw new IllegalArgumentException("layerNormEpsilon must be finite and positive");
    }
    if (maxRelativePositions < 1) {
      maxRelativePositions = maxPositions;
    }
  }

  /** Width of one attention head. */
  public int headSize() {
    return hiddenSize / numHeads;
  }

  /** Returns whether the JSON declares the supported DeBERTa-v2 family. */
  public static boolean matches(Path configPath) {
    if (!Files.isRegularFile(configPath)) {
      return false;
    }
    try (JsonParser parser = JSON.createParser(configPath.toFile())) {
      if (parser.nextToken() != JsonToken.START_OBJECT) {
        return false;
      }
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        if ("model_type".equals(name)) {
          return value == JsonToken.VALUE_STRING && "deberta-v2".equals(parser.getText());
        }
        parser.skipChildren();
      }
      return false;
    } catch (IOException | RuntimeException malformed) {
      return false;
    }
  }

  /** Parses and rejects graph variants not implemented by the pure-Java path. */
  public static DebertaV2Config parse(Path configPath) throws IOException {
    Objects.requireNonNull(configPath, "configPath");
    Fields fields = new Fields();
    try (JsonParser parser = JSON.createParser(configPath.toFile())) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "configuration root");
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "configuration field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "architectures" -> fields.architectures = readStrings(parser, value, name);
          case "hidden_act" -> fields.hiddenActivation = readString(parser, value, name);
          case "hidden_size" -> fields.hiddenSize = readInt(parser, value, name);
          case "intermediate_size" -> fields.intermediateSize = readInt(parser, value, name);
          case "layer_norm_eps" -> fields.layerNormEpsilon = readFloat(parser, value, name);
          case "max_position_embeddings" -> fields.maxPositions = readInt(parser, value, name);
          case "max_relative_positions" ->
              fields.maxRelativePositions = readInt(parser, value, name);
          case "model_type" -> fields.modelType = readString(parser, value, name);
          case "norm_rel_ebd" -> fields.relativeEmbeddingNorm = readString(parser, value, name);
          case "num_attention_heads" -> fields.numHeads = readInt(parser, value, name);
          case "num_hidden_layers" -> fields.numLayers = readInt(parser, value, name);
          case "pad_token_id" -> fields.padTokenId = readInt(parser, value, name);
          case "pooler_hidden_act" -> fields.poolerActivation = readString(parser, value, name);
          case "pooler_hidden_size" -> fields.poolerHiddenSize = readInt(parser, value, name);
          case "pos_att_type" -> fields.positionAttentionTypes = readStrings(parser, value, name);
          case "position_biased_input" ->
              fields.positionBiasedInput = readBoolean(parser, value, name);
          case "position_buckets" -> fields.positionBuckets = readInt(parser, value, name);
          case "relative_attention" -> fields.relativeAttention = readBoolean(parser, value, name);
          case "share_att_key" -> fields.shareAttentionKey = readBoolean(parser, value, name);
          case "type_vocab_size" -> fields.typeVocabularySize = readInt(parser, value, name);
          case "vocab_size" -> fields.vocabSize = readInt(parser, value, name);
          default -> parser.skipChildren();
        }
      }
      if (parser.nextToken() != null) {
        throw malformed("configuration has content after its root object");
      }
    }
    return fields.build();
  }

  private static List<String> readStrings(JsonParser parser, JsonToken token, String description)
      throws IOException {
    require(token, JsonToken.START_ARRAY, description);
    List<String> values = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      values.add(readString(parser, parser.currentToken(), description + " item"));
    }
    return List.copyOf(values);
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

  private static float readFloat(JsonParser parser, JsonToken token, String description)
      throws IOException {
    if (!token.isNumeric()) {
      throw malformed(description + " must be numeric");
    }
    return parser.getFloatValue();
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

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw malformed(name + " must be positive: " + value);
    }
  }

  private static final class Fields {
    private List<String> architectures = List.of();
    private String hiddenActivation;
    private int hiddenSize;
    private int intermediateSize;
    private float layerNormEpsilon;
    private int maxPositions;
    private int maxRelativePositions;
    private String modelType;
    private int numHeads;
    private int numLayers;
    private int padTokenId;
    private String poolerActivation;
    private int poolerHiddenSize;
    private List<String> positionAttentionTypes = List.of();
    private boolean positionBiasedInput;
    private int positionBuckets;
    private boolean relativeAttention;
    private String relativeEmbeddingNorm;
    private boolean shareAttentionKey;
    private int typeVocabularySize;
    private int vocabSize;

    private DebertaV2Config build() {
      if (!architectures.equals(List.of("DebertaV2ForSequenceClassification"))
          || !"deberta-v2".equals(modelType)
          || !"gelu".equals(hiddenActivation)
          || !"gelu".equals(poolerActivation)
          || !"layer_norm".equals(relativeEmbeddingNorm)
          || !positionAttentionTypes.equals(List.of("p2c", "c2p"))
          || !relativeAttention
          || !shareAttentionKey
          || positionBiasedInput
          || typeVocabularySize != 0
          || poolerHiddenSize != hiddenSize) {
        throw malformed("unsupported DeBERTa-v2 sequence-classification graph variant");
      }
      return new DebertaV2Config(
          hiddenSize,
          intermediateSize,
          numLayers,
          numHeads,
          vocabSize,
          maxPositions,
          positionBuckets,
          maxRelativePositions,
          poolerHiddenSize,
          padTokenId,
          layerNormEpsilon,
          relativeAttention,
          shareAttentionKey,
          positionBiasedInput);
    }
  }
}
