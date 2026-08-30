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
package com.integrallis.models.backend.purejava.gptoss;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.StreamReadFeature;
import com.fasterxml.jackson.core.exc.StreamReadException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Validated GPT-OSS execution contract read from a Hugging Face {@code config.json}. */
public record GptOssHuggingFaceConfig(
    List<String> architectures,
    int hiddenSize,
    int numLayers,
    int numHeads,
    int numKvHeads,
    int headDim,
    int vocabSize,
    int maxPosition,
    int initialContextLength,
    int intermediateSize,
    int numExperts,
    int expertsPerToken,
    int slidingWindow,
    float rmsNormEps,
    float ropeTheta,
    float ropeYarnFactor,
    float ropeBetaFast,
    float ropeBetaSlow,
    int ropeOriginalContext,
    float swigluLimit,
    float hiddenActAlpha,
    String hiddenActivation,
    String quantizationMethod,
    boolean attentionBias,
    boolean tieWordEmbeddings,
    int eosTokenId,
    int padTokenId,
    List<String> layerTypes) {

  private static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();

  public GptOssHuggingFaceConfig {
    architectures = List.copyOf(Objects.requireNonNull(architectures, "architectures"));
    hiddenActivation = Objects.requireNonNull(hiddenActivation, "hiddenActivation");
    quantizationMethod = Objects.requireNonNull(quantizationMethod, "quantizationMethod");
    layerTypes = List.copyOf(Objects.requireNonNull(layerTypes, "layerTypes"));
    if (!architectures.contains("GptOssForCausalLM")) {
      throw malformed("architectures must include GptOssForCausalLM");
    }
    directPositive(hiddenSize, "hiddenSize");
    directPositive(numLayers, "numLayers");
    directPositive(numHeads, "numHeads");
    directPositive(numKvHeads, "numKvHeads");
    directPositive(headDim, "headDim");
    directPositive(vocabSize, "vocabSize");
    directPositive(maxPosition, "maxPosition");
    directPositive(initialContextLength, "initialContextLength");
    directPositive(intermediateSize, "intermediateSize");
    directPositive(numExperts, "numExperts");
    directPositive(expertsPerToken, "expertsPerToken");
    directPositive(ropeOriginalContext, "ropeOriginalContext");
    directFinitePositive(rmsNormEps, "rmsNormEps");
    directFinitePositive(ropeTheta, "ropeTheta");
    directFinitePositive(ropeYarnFactor, "ropeYarnFactor");
    directFinitePositive(ropeBetaFast, "ropeBetaFast");
    directFinitePositive(ropeBetaSlow, "ropeBetaSlow");
    directFinitePositive(swigluLimit, "swigluLimit");
    directFinitePositive(hiddenActAlpha, "hiddenActAlpha");
    if (hiddenSize % 32 != 0 || intermediateSize % 32 != 0) {
      throw malformed("MXFP4 hiddenSize and intermediateSize must be multiples of 32");
    }
    if (numHeads % numKvHeads != 0) {
      throw malformed("numHeads must be divisible by numKvHeads");
    }
    directProduct(numHeads, headDim, "queryDimension");
    directProduct(numKvHeads, headDim, "keyValueDimension");
    if (expertsPerToken > numExperts) {
      throw malformed("expertsPerToken must not exceed numExperts");
    }
    if (maxPosition < initialContextLength) {
      throw malformed("maxPosition must cover initialContextLength");
    }
    if (ropeOriginalContext != initialContextLength) {
      throw malformed("ropeOriginalContext must equal initialContextLength");
    }
    if (!"silu".equals(hiddenActivation)) {
      throw malformed("GPT-OSS hiddenActivation must be silu; got " + hiddenActivation);
    }
    if (!"mxfp4".equals(quantizationMethod)) {
      throw malformed("GPT-OSS Java path currently requires mxfp4 quantization");
    }
    if (layerTypes.size() != numLayers
        || layerTypes.stream()
            .anyMatch(
                type -> !"full_attention".equals(type) && !"sliding_attention".equals(type))) {
      throw malformed("layerTypes must contain one supported attention type per layer");
    }
    if (layerTypes.contains("sliding_attention")) {
      directPositive(slidingWindow, "slidingWindow");
    } else if (slidingWindow != 0) {
      throw malformed("slidingWindow must be zero when no layer uses sliding attention");
    }
    directTokenId(eosTokenId, vocabSize, "eosTokenId");
    directTokenId(padTokenId, vocabSize, "padTokenId");
  }

  public static GptOssHuggingFaceConfig parse(Path path) throws IOException {
    Objects.requireNonNull(path, "path");
    Fields fields = new Fields();
    try (JsonParser parser = JSON.createParser(path.toFile())) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "config root");
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "config field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "architectures" -> fields.architectures = readStringArray(parser, value, name);
          case "attention_bias" -> fields.attentionBias = readBoolean(parser, value, name);
          case "eos_token_id" -> fields.eosTokenId = readInt(parser, value, name);
          case "experts_per_token" -> fields.legacyExpertsPerToken = readInt(parser, value, name);
          case "head_dim" -> fields.headDim = readInt(parser, value, name);
          case "hidden_act" -> fields.hiddenActivation = readString(parser, value, name);
          case "hidden_act_alpha" -> fields.hiddenActAlpha = readFloat(parser, value, name);
          case "hidden_size" -> fields.hiddenSize = readInt(parser, value, name);
          case "initial_context_length" ->
              fields.initialContextLength = readInt(parser, value, name);
          case "intermediate_size" -> fields.intermediateSize = readInt(parser, value, name);
          case "layer_types" -> fields.layerTypes = readStringArray(parser, value, name);
          case "max_position_embeddings" -> fields.maxPosition = readInt(parser, value, name);
          case "model_type" -> fields.modelType = readString(parser, value, name);
          case "num_attention_heads" -> fields.numHeads = readInt(parser, value, name);
          case "num_experts_per_tok" -> fields.expertsPerToken = readInt(parser, value, name);
          case "num_hidden_layers" -> fields.numLayers = readInt(parser, value, name);
          case "num_key_value_heads" -> fields.numKvHeads = readInt(parser, value, name);
          case "num_local_experts" -> fields.numExperts = readInt(parser, value, name);
          case "pad_token_id" -> fields.padTokenId = readInt(parser, value, name);
          case "quantization_config" -> readQuantization(parser, value, fields);
          case "rms_norm_eps" -> fields.rmsNormEps = readFloat(parser, value, name);
          case "rope_scaling" -> readRopeScaling(parser, value, fields);
          case "rope_theta" -> fields.ropeTheta = readFloat(parser, value, name);
          case "sliding_window" -> fields.slidingWindow = readInt(parser, value, name);
          case "swiglu_limit" -> fields.swigluLimit = readFloat(parser, value, name);
          case "tie_word_embeddings" -> fields.tieWordEmbeddings = readBoolean(parser, value, name);
          case "vocab_size" -> fields.vocabSize = readInt(parser, value, name);
          default -> parser.skipChildren();
        }
      }
      if (parser.nextToken() != null) {
        throw malformed("config has content after its root object");
      }
    } catch (StreamReadException malformed) {
      throw malformed("invalid or duplicate config JSON: " + malformed.getMessage(), malformed);
    }
    return fields.toConfig();
  }

  public int queryDimension() {
    return Math.multiplyExact(numHeads, headDim);
  }

  public int keyValueDimension() {
    return Math.multiplyExact(numKvHeads, headDim);
  }

  public boolean usesSlidingAttention(int layer) {
    requireLayer(layer);
    return "sliding_attention".equals(layerTypes.get(layer));
  }

  public int attentionStartPosition(int layer, int position) {
    requireLayer(layer);
    if (position < 0 || position >= maxPosition) {
      throw new IllegalArgumentException("position out of range: " + position);
    }
    return usesSlidingAttention(layer) ? Math.max(0, position - slidingWindow + 1) : 0;
  }

  private void requireLayer(int layer) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
  }

  private static void readQuantization(JsonParser parser, JsonToken token, Fields fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "quantization_config");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "quantization_config field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("quant_method".equals(name)) {
        fields.quantizationMethod = readString(parser, value, "quantization_config.quant_method");
      } else {
        parser.skipChildren();
      }
    }
  }

  private static void readRopeScaling(JsonParser parser, JsonToken token, Fields fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "rope_scaling");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "rope_scaling field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "beta_fast" ->
            fields.ropeBetaFast = readFloat(parser, value, "rope_scaling.beta_fast");
        case "beta_slow" ->
            fields.ropeBetaSlow = readFloat(parser, value, "rope_scaling.beta_slow");
        case "factor" -> fields.ropeYarnFactor = readFloat(parser, value, "rope_scaling.factor");
        case "original_max_position_embeddings" ->
            fields.ropeOriginalContext =
                readInt(parser, value, "rope_scaling.original_max_position_embeddings");
        case "rope_type" -> fields.ropeType = readString(parser, value, "rope_scaling.rope_type");
        case "truncate" ->
            fields.ropeTruncate = readBoolean(parser, value, "rope_scaling.truncate");
        default -> parser.skipChildren();
      }
    }
  }

  private static List<String> readStringArray(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.START_ARRAY, name);
    List<String> values = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      require(parser.currentToken(), JsonToken.VALUE_STRING, name + " entry");
      values.add(parser.getText());
    }
    return List.copyOf(values);
  }

  private static int readInt(JsonParser parser, JsonToken token, String name) throws IOException {
    if (!token.isNumeric()) {
      throw malformed(name + " must be an integer");
    }
    return parser.getIntValue();
  }

  private static float readFloat(JsonParser parser, JsonToken token, String name)
      throws IOException {
    if (!token.isNumeric()) {
      throw malformed(name + " must be numeric");
    }
    return parser.getFloatValue();
  }

  private static String readString(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.VALUE_STRING, name);
    return parser.getText();
  }

  private static boolean readBoolean(JsonParser parser, JsonToken token, String name) {
    if (token != JsonToken.VALUE_TRUE && token != JsonToken.VALUE_FALSE) {
      throw malformed(name + " must be boolean");
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

  private static IllegalArgumentException malformed(String message, Throwable cause) {
    return new IllegalArgumentException(message, cause);
  }

  private static int positive(Integer value, String name) {
    if (value == null || value <= 0) {
      throw malformed(name + " must be present and positive");
    }
    return value;
  }

  private static float positive(Float value, String name) {
    if (value == null || !(value > 0.0f) || !Float.isFinite(value)) {
      throw malformed(name + " must be present, finite, and positive");
    }
    return value;
  }

  private static void directPositive(int value, String name) {
    if (value <= 0) {
      throw malformed(name + " must be positive: " + value);
    }
  }

  private static void directFinitePositive(float value, String name) {
    if (!(value > 0.0f) || !Float.isFinite(value)) {
      throw malformed(name + " must be finite and positive: " + value);
    }
  }

  private static void directTokenId(int tokenId, int vocabSize, String name) {
    if (tokenId < -1 || tokenId >= vocabSize) {
      throw malformed(name + " must be -1 or inside the vocabulary: " + tokenId);
    }
  }

  private static void directProduct(int left, int right, String name) {
    long product = (long) left * right;
    if (product > Integer.MAX_VALUE) {
      throw malformed(name + " exceeds the Java array limit: " + product);
    }
  }

  private static final class Fields {
    private List<String> architectures;
    private Boolean attentionBias;
    private Integer eosTokenId;
    private Integer expertsPerToken;
    private Integer headDim;
    private Float hiddenActAlpha;
    private String hiddenActivation;
    private Integer hiddenSize;
    private Integer initialContextLength;
    private Integer intermediateSize;
    private List<String> layerTypes;
    private Integer legacyExpertsPerToken;
    private Integer maxPosition;
    private String modelType;
    private Integer numExperts;
    private Integer numHeads;
    private Integer numKvHeads;
    private Integer numLayers;
    private Integer padTokenId;
    private String quantizationMethod;
    private Float rmsNormEps;
    private Float ropeBetaFast;
    private Float ropeBetaSlow;
    private Integer ropeOriginalContext;
    private Float ropeTheta;
    private boolean ropeTruncate;
    private String ropeType;
    private Float ropeYarnFactor;
    private Integer slidingWindow;
    private Float swigluLimit;
    private boolean tieWordEmbeddings;
    private Integer vocabSize;

    private GptOssHuggingFaceConfig toConfig() {
      if (!"gpt_oss".equals(modelType)) {
        throw malformed("GPT-OSS config requires model_type gpt_oss; got " + modelType);
      }
      List<String> resolvedArchitectures =
          architectures == null ? List.of("GptOssForCausalLM") : architectures;
      if (!resolvedArchitectures.contains("GptOssForCausalLM")) {
        throw malformed("architectures must include GptOssForCausalLM");
      }
      if (!"silu".equals(hiddenActivation)) {
        throw malformed("GPT-OSS hidden_act must be silu; got " + hiddenActivation);
      }
      if (!"mxfp4".equals(quantizationMethod)) {
        throw malformed("GPT-OSS Java path currently requires mxfp4 quantization");
      }
      if (!"yarn".equals(ropeType) || ropeTruncate) {
        throw malformed("GPT-OSS Java path currently requires non-truncating yarn rope_scaling");
      }

      int resolvedHidden = positive(hiddenSize, "hidden_size");
      int resolvedLayers = positive(numLayers, "num_hidden_layers");
      int resolvedHeads = positive(numHeads, "num_attention_heads");
      int resolvedKvHeads =
          numKvHeads == null ? resolvedHeads : positive(numKvHeads, "num_key_value_heads");
      int resolvedHeadDim =
          headDim == null ? resolvedHidden / resolvedHeads : positive(headDim, "head_dim");
      int resolvedIntermediate = positive(intermediateSize, "intermediate_size");
      if (resolvedHidden % 32 != 0 || resolvedIntermediate % 32 != 0) {
        throw malformed("MXFP4 hidden_size and intermediate_size must be multiples of 32");
      }
      int resolvedExperts = positive(numExperts, "num_local_experts");
      int resolvedTopK =
          expertsPerToken == null
              ? positive(legacyExpertsPerToken, "experts_per_token")
              : positive(expertsPerToken, "num_experts_per_tok");
      if (legacyExpertsPerToken != null && legacyExpertsPerToken != resolvedTopK) {
        throw malformed("experts_per_token and num_experts_per_tok must agree");
      }
      if (resolvedHeads % resolvedKvHeads != 0) {
        throw malformed("num_attention_heads must be divisible by num_key_value_heads");
      }
      if (resolvedTopK > resolvedExperts) {
        throw malformed("num_experts_per_tok must not exceed num_local_experts");
      }
      int resolvedInitialContext = positive(initialContextLength, "initial_context_length");
      int resolvedMaximum = positive(maxPosition, "max_position_embeddings");
      if (resolvedMaximum < resolvedInitialContext) {
        throw malformed("max_position_embeddings must cover initial_context_length");
      }
      int resolvedOriginalContext =
          positive(ropeOriginalContext, "rope_scaling.original_max_position_embeddings");
      if (resolvedOriginalContext != resolvedInitialContext) {
        throw malformed("rope original context must equal initial_context_length");
      }

      List<String> schedule = layerTypes;
      if (schedule == null) {
        schedule = Collections.nCopies(resolvedLayers, "full_attention");
      }
      if (schedule.size() != resolvedLayers
          || schedule.stream()
              .anyMatch(
                  type -> !"full_attention".equals(type) && !"sliding_attention".equals(type))) {
        throw malformed("layer_types must contain one supported attention type per layer");
      }
      boolean hasSliding = schedule.contains("sliding_attention");
      int resolvedSliding = hasSliding ? positive(slidingWindow, "sliding_window") : 0;

      return new GptOssHuggingFaceConfig(
          resolvedArchitectures,
          resolvedHidden,
          resolvedLayers,
          resolvedHeads,
          resolvedKvHeads,
          resolvedHeadDim,
          positive(vocabSize, "vocab_size"),
          resolvedMaximum,
          resolvedInitialContext,
          resolvedIntermediate,
          resolvedExperts,
          resolvedTopK,
          resolvedSliding,
          positive(rmsNormEps, "rms_norm_eps"),
          positive(ropeTheta, "rope_theta"),
          positive(ropeYarnFactor, "rope_scaling.factor"),
          positive(ropeBetaFast, "rope_scaling.beta_fast"),
          positive(ropeBetaSlow, "rope_scaling.beta_slow"),
          resolvedOriginalContext,
          positive(swigluLimit, "swiglu_limit"),
          hiddenActAlpha == null ? 1.702f : positive(hiddenActAlpha, "hidden_act_alpha"),
          hiddenActivation,
          quantizationMethod,
          attentionBias == null || attentionBias,
          tieWordEmbeddings,
          eosTokenId == null ? -1 : eosTokenId,
          padTokenId == null ? -1 : padTokenId,
          schedule);
    }
  }
}
