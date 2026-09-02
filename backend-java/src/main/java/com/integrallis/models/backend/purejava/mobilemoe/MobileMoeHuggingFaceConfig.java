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
package com.integrallis.models.backend.purejava.mobilemoe;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.StreamReadFeature;
import com.fasterxml.jackson.core.exc.StreamReadException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.IntStream;

/** Validated MobileMoE execution contract read from a Hugging Face {@code config.json}. */
public record MobileMoeHuggingFaceConfig(
    List<String> architectures,
    int hiddenSize,
    int intermediateSize,
    int sharedIntermediateSize,
    int numLayers,
    int numHeads,
    int numKvHeads,
    int headDimension,
    int vocabSize,
    int contextLength,
    float rmsNormEpsilon,
    float ropeTheta,
    RopeScaling ropeScaling,
    int numExperts,
    int expertsPerToken,
    int moeLayerStep,
    List<Integer> ropeLayerFlags,
    List<String> layerTypes,
    boolean qkNorm,
    boolean tiedEmbeddings,
    boolean attentionBias,
    float routeScale,
    boolean normalizeTopK,
    Optional<Quantization> quantization) {

  private static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();

  public record RopeScaling(
      String type, float factor, int originalContext, float lowFrequency, float highFrequency) {}

  public record Quantization(
      String format,
      int groupSize,
      int embeddingGroupSize,
      int minimum,
      int maximum,
      boolean symmetric,
      boolean packed,
      String scaleType,
      String linearGroupAxis,
      String expertGroupAxis) {}

  public MobileMoeHuggingFaceConfig {
    architectures = List.copyOf(Objects.requireNonNull(architectures, "architectures"));
    ropeScaling = Objects.requireNonNull(ropeScaling, "ropeScaling");
    ropeLayerFlags = List.copyOf(Objects.requireNonNull(ropeLayerFlags, "ropeLayerFlags"));
    layerTypes = List.copyOf(Objects.requireNonNull(layerTypes, "layerTypes"));
    quantization = Objects.requireNonNull(quantization, "quantization");
    if (!architectures.contains("MobileMoEForCausalLM")) {
      throw malformed("architectures must include MobileMoEForCausalLM");
    }
    positive(hiddenSize, "hiddenSize");
    positive(intermediateSize, "intermediateSize");
    positive(sharedIntermediateSize, "sharedIntermediateSize");
    positive(numLayers, "numLayers");
    positive(numHeads, "numHeads");
    positive(numKvHeads, "numKvHeads");
    positive(headDimension, "headDimension");
    positive(vocabSize, "vocabSize");
    positive(contextLength, "contextLength");
    positive(numExperts, "numExperts");
    positive(expertsPerToken, "expertsPerToken");
    positive(moeLayerStep, "moeLayerStep");
    finitePositive(rmsNormEpsilon, "rmsNormEpsilon");
    finitePositive(ropeTheta, "ropeTheta");
    finitePositive(routeScale, "routeScale");
    if (Math.multiplyExact(numHeads, headDimension) != hiddenSize) {
      throw malformed("attention heads must cover hiddenSize");
    }
    if (numHeads % numKvHeads != 0) {
      throw malformed("numHeads must be divisible by numKvHeads");
    }
    if (expertsPerToken > numExperts) {
      throw malformed("expertsPerToken must not exceed numExperts");
    }
    if (ropeLayerFlags.size() != numLayers
        || ropeLayerFlags.stream().anyMatch(flag -> flag != 0 && flag != 1)) {
      throw malformed("no_rope_layers must contain one 0/1 flag per layer");
    }
    if (layerTypes.size() != numLayers
        || layerTypes.stream().anyMatch(type -> !"full_attention".equals(type))) {
      throw malformed("layer_types must contain one supported full_attention entry per layer");
    }
    if (!qkNorm || attentionBias || !tiedEmbeddings || !normalizeTopK) {
      throw malformed(
          "current MobileMoE path requires QK norm, tied embeddings, unbiased attention, and normalized top-k");
    }
    if (!"llama3".equals(ropeScaling.type())
        || ropeScaling.originalContext() != contextLength
        || !(ropeScaling.factor() >= 1.0f)
        || !(ropeScaling.lowFrequency() > 0.0f)
        || !(ropeScaling.highFrequency() > 0.0f)) {
      throw malformed("unsupported MobileMoE rope_scaling contract");
    }
    quantization.ifPresent(MobileMoeHuggingFaceConfig::validateQuantization);
  }

  public static MobileMoeHuggingFaceConfig parse(Path path) throws IOException {
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
          case "attention_bias" -> fields.attentionBias = readBoolean(value, name);
          case "head_dim" -> fields.headDimension = readInt(parser, value, name);
          case "hidden_act" -> fields.hiddenActivation = readString(parser, value, name);
          case "hidden_size" -> fields.hiddenSize = readInt(parser, value, name);
          case "intermediate_size" -> fields.intermediateSize = readInt(parser, value, name);
          case "intermediate_size_mlp" ->
              fields.sharedIntermediateSize = readInt(parser, value, name);
          case "interleave_moe_layer_step" -> fields.moeLayerStep = readInt(parser, value, name);
          case "layer_types" -> fields.layerTypes = readStringArray(parser, value, name);
          case "max_position_embeddings" -> fields.contextLength = readInt(parser, value, name);
          case "model_type" -> fields.modelType = readString(parser, value, name);
          case "no_rope_layers" -> fields.ropeLayerFlags = readIntArray(parser, value, name);
          case "norm_topk_prob" -> fields.normalizeTopK = readBoolean(value, name);
          case "num_attention_heads" -> fields.numHeads = readInt(parser, value, name);
          case "num_experts_per_tok" -> fields.expertsPerToken = readInt(parser, value, name);
          case "num_hidden_layers" -> fields.numLayers = readInt(parser, value, name);
          case "num_key_value_heads" -> fields.numKvHeads = readInt(parser, value, name);
          case "num_local_experts" -> fields.numExperts = readInt(parser, value, name);
          case "quantization" -> fields.quantization = readQuantization(parser, value);
          case "rms_norm_eps" -> fields.rmsNormEpsilon = readFloat(parser, value, name);
          case "rope_scaling" -> fields.ropeScaling = readRopeScaling(parser, value);
          case "rope_theta" -> fields.ropeTheta = readFloat(parser, value, name);
          case "routed_scaling_factor" -> fields.routeScale = readFloat(parser, value, name);
          case "tie_word_embeddings" -> fields.tiedEmbeddings = readBoolean(value, name);
          case "use_qk_norm" -> fields.qkNorm = readBoolean(value, name);
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

  public static boolean matches(Path path) throws IOException {
    Objects.requireNonNull(path, "path");
    try (JsonParser parser = JSON.createParser(path.toFile())) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "config root");
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "config field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        if ("model_type".equals(name)) {
          return "mobilemoe".equals(readString(parser, value, name));
        }
        parser.skipChildren();
      }
      return false;
    }
  }

  public int queryDimension() {
    return Math.multiplyExact(numHeads, headDimension);
  }

  public int keyValueDimension() {
    return Math.multiplyExact(numKvHeads, headDimension);
  }

  public List<Integer> moeLayers() {
    return IntStream.iterate(
            moeLayerStep - 1, layer -> layer < numLayers, layer -> layer + moeLayerStep)
        .boxed()
        .toList();
  }

  public boolean usesRope(int layer) {
    if (layer < 0 || layer >= numLayers) {
      throw new IndexOutOfBoundsException("layer out of range: " + layer);
    }
    return ropeLayerFlags.get(layer) == 1;
  }

  private static void validateQuantization(Quantization quantization) {
    if (!"mobilemoe-int4-g32".equals(quantization.format())) {
      throw malformed("quantization format must be mobilemoe-int4-g32");
    }
    if (quantization.groupSize() != 32
        || quantization.embeddingGroupSize() != 32
        || quantization.minimum() != -8
        || quantization.maximum() != 7
        || !quantization.symmetric()
        || !quantization.packed()
        || !"float16".equals(quantization.scaleType())) {
      throw malformed("unsupported MobileMoE packed INT4 quantization geometry");
    }
    if (!"in (last dim of [out, in])".equals(quantization.linearGroupAxis())
        || !"out (last dim of [E, in, out])".equals(quantization.expertGroupAxis())) {
      throw malformed("unsupported MobileMoE quantization axes");
    }
  }

  private static RopeScaling readRopeScaling(JsonParser parser, JsonToken token)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "rope_scaling");
    String type = null;
    Float factor = null;
    Integer originalContext = null;
    Float low = null;
    Float high = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "rope_scaling field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "rope_type" -> type = readString(parser, value, "rope_scaling.rope_type");
        case "factor" -> factor = readFloat(parser, value, "rope_scaling.factor");
        case "original_max_position_embeddings" ->
            originalContext =
                readInt(parser, value, "rope_scaling.original_max_position_embeddings");
        case "low_freq_factor" -> low = readFloat(parser, value, "rope_scaling.low_freq_factor");
        case "high_freq_factor" -> high = readFloat(parser, value, "rope_scaling.high_freq_factor");
        default -> parser.skipChildren();
      }
    }
    return new RopeScaling(
        required(type, "rope_scaling.rope_type"),
        required(factor, "rope_scaling.factor"),
        required(originalContext, "rope_scaling.original_max_position_embeddings"),
        required(low, "rope_scaling.low_freq_factor"),
        required(high, "rope_scaling.high_freq_factor"));
  }

  private static Quantization readQuantization(JsonParser parser, JsonToken token)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "quantization");
    String format = null;
    Integer groupSize = null;
    Integer embeddingGroupSize = null;
    Integer minimum = null;
    Integer maximum = null;
    Boolean symmetric = null;
    Boolean packed = null;
    String scaleType = null;
    String linearAxis = null;
    String expertAxis = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "quantization field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "format" -> format = readString(parser, value, "quantization.format");
        case "group_size" -> groupSize = readInt(parser, value, "quantization.group_size");
        case "embedding_group_size" ->
            embeddingGroupSize = readInt(parser, value, "quantization.embedding_group_size");
        case "qmin" -> minimum = readInt(parser, value, "quantization.qmin");
        case "qmax" -> maximum = readInt(parser, value, "quantization.qmax");
        case "symmetric" -> symmetric = readBoolean(value, "quantization.symmetric");
        case "packed" -> packed = readBoolean(value, "quantization.packed");
        case "scale_dtype" -> scaleType = readString(parser, value, "quantization.scale_dtype");
        case "linear_group_axis" ->
            linearAxis = readString(parser, value, "quantization.linear_group_axis");
        case "expert_group_axis" ->
            expertAxis = readString(parser, value, "quantization.expert_group_axis");
        default -> parser.skipChildren();
      }
    }
    return new Quantization(
        required(format, "quantization.format"),
        required(groupSize, "quantization.group_size"),
        required(embeddingGroupSize, "quantization.embedding_group_size"),
        required(minimum, "quantization.qmin"),
        required(maximum, "quantization.qmax"),
        required(symmetric, "quantization.symmetric"),
        required(packed, "quantization.packed"),
        required(scaleType, "quantization.scale_dtype"),
        required(linearAxis, "quantization.linear_group_axis"),
        required(expertAxis, "quantization.expert_group_axis"));
  }

  private static List<String> readStringArray(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.START_ARRAY, name);
    List<String> values = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      values.add(readString(parser, parser.currentToken(), name + " entry"));
    }
    return List.copyOf(values);
  }

  private static List<Integer> readIntArray(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.START_ARRAY, name);
    List<Integer> values = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      values.add(readInt(parser, parser.currentToken(), name + " entry"));
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

  private static boolean readBoolean(JsonToken token, String name) {
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

  private static <T> T required(T value, String name) {
    if (value == null) {
      throw malformed(name + " must be present");
    }
    return value;
  }

  private static void positive(int value, String name) {
    if (value <= 0) {
      throw malformed(name + " must be positive: " + value);
    }
  }

  private static void finitePositive(float value, String name) {
    if (!(value > 0.0f) || !Float.isFinite(value)) {
      throw malformed(name + " must be finite and positive: " + value);
    }
  }

  private static IllegalArgumentException malformed(String message) {
    return new IllegalArgumentException(message);
  }

  private static IllegalArgumentException malformed(String message, Throwable cause) {
    return new IllegalArgumentException(message, cause);
  }

  private static final class Fields {
    private List<String> architectures;
    private Boolean attentionBias;
    private Integer headDimension;
    private String hiddenActivation;
    private Integer hiddenSize;
    private Integer intermediateSize;
    private Integer sharedIntermediateSize;
    private Integer moeLayerStep;
    private List<String> layerTypes;
    private Integer contextLength;
    private String modelType;
    private List<Integer> ropeLayerFlags;
    private Boolean normalizeTopK;
    private Integer numHeads;
    private Integer expertsPerToken;
    private Integer numLayers;
    private Integer numKvHeads;
    private Integer numExperts;
    private Quantization quantization;
    private Float rmsNormEpsilon;
    private RopeScaling ropeScaling;
    private Float ropeTheta;
    private Float routeScale;
    private Boolean tiedEmbeddings;
    private Boolean qkNorm;
    private Integer vocabSize;

    private MobileMoeHuggingFaceConfig toConfig() {
      if (!"mobilemoe".equals(required(modelType, "model_type"))) {
        throw malformed("model_type must be mobilemoe; got " + modelType);
      }
      if (!"silu".equals(required(hiddenActivation, "hidden_act"))) {
        throw malformed("hidden_act must be silu; got " + hiddenActivation);
      }
      return new MobileMoeHuggingFaceConfig(
          required(architectures, "architectures"),
          required(hiddenSize, "hidden_size"),
          required(intermediateSize, "intermediate_size"),
          required(sharedIntermediateSize, "intermediate_size_mlp"),
          required(numLayers, "num_hidden_layers"),
          required(numHeads, "num_attention_heads"),
          required(numKvHeads, "num_key_value_heads"),
          required(headDimension, "head_dim"),
          required(vocabSize, "vocab_size"),
          required(contextLength, "max_position_embeddings"),
          required(rmsNormEpsilon, "rms_norm_eps"),
          required(ropeTheta, "rope_theta"),
          required(ropeScaling, "rope_scaling"),
          required(numExperts, "num_local_experts"),
          required(expertsPerToken, "num_experts_per_tok"),
          required(moeLayerStep, "interleave_moe_layer_step"),
          required(ropeLayerFlags, "no_rope_layers"),
          required(layerTypes, "layer_types"),
          required(qkNorm, "use_qk_norm"),
          required(tiedEmbeddings, "tie_word_embeddings"),
          required(attentionBias, "attention_bias"),
          required(routeScale, "routed_scaling_factor"),
          required(normalizeTopK, "norm_topk_prob"),
          Optional.ofNullable(quantization));
    }
  }
}
