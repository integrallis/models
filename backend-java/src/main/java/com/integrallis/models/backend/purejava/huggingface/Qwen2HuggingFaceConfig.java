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
package com.integrallis.models.backend.purejava.huggingface;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.StreamReadFeature;
import com.fasterxml.jackson.core.exc.StreamReadException;
import com.integrallis.models.backend.purejava.llama.DecoderArchitecture;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** The Qwen 2 decoder contract read from a Hugging Face {@code config.json}. */
public record Qwen2HuggingFaceConfig(
    LlamaConfig model,
    List<String> architectures,
    String hiddenActivation,
    String torchDtype,
    boolean tieWordEmbeddings,
    int bosTokenId,
    int eosTokenId) {

  private static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();

  public Qwen2HuggingFaceConfig {
    Objects.requireNonNull(model, "model");
    architectures = List.copyOf(Objects.requireNonNull(architectures, "architectures"));
    hiddenActivation = Objects.requireNonNull(hiddenActivation, "hiddenActivation");
    torchDtype = Objects.requireNonNull(torchDtype, "torchDtype");
  }

  /**
   * Parses Qwen 2 configuration using the same defaults as FreeToken's Qwen 2 adapter: KV heads
   * fall back to attention heads, head size falls back to hidden size divided by attention heads,
   * RoPE base falls back through {@code rope_scaling.rope_theta} to 10,000, and tied embeddings
   * default to false.
   */
  public static Qwen2HuggingFaceConfig parse(Path path) throws IOException {
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
          case "bos_token_id" -> fields.bosTokenId = readInt(parser, value, name);
          case "eos_token_id" -> fields.eosTokenId = readInt(parser, value, name);
          case "head_dim" -> fields.headDim = readInt(parser, value, name);
          case "hidden_act" -> fields.hiddenActivation = readString(parser, value, name);
          case "hidden_size" -> fields.hiddenSize = readInt(parser, value, name);
          case "intermediate_size" -> fields.intermediateSize = readInt(parser, value, name);
          case "max_position_embeddings" -> fields.maxPosition = readInt(parser, value, name);
          case "model_type" -> fields.modelType = readString(parser, value, name);
          case "num_attention_heads" -> fields.numAttentionHeads = readInt(parser, value, name);
          case "num_hidden_layers" -> fields.numLayers = readInt(parser, value, name);
          case "num_key_value_heads" -> fields.numKvHeads = readInt(parser, value, name);
          case "rms_norm_eps" -> fields.rmsNormEps = readFloat(parser, value, name);
          case "rope_scaling" -> fields.scaledRopeTheta = readRopeTheta(parser, value);
          case "rope_theta" -> fields.ropeTheta = readFloat(parser, value, name);
          case "sliding_window" -> fields.slidingWindow = readInt(parser, value, name);
          case "tie_word_embeddings" -> fields.tieWordEmbeddings = readBoolean(parser, value, name);
          case "torch_dtype" -> fields.torchDtype = readString(parser, value, name);
          case "use_sliding_window" -> fields.useSlidingWindow = readBoolean(parser, value, name);
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

  private static Float readRopeTheta(JsonParser parser, JsonToken token) throws IOException {
    if (token == JsonToken.VALUE_NULL) {
      return null;
    }
    require(token, JsonToken.START_OBJECT, "rope_scaling");
    Float theta = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "rope_scaling field");
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if ("rope_theta".equals(name)) {
        theta = readFloat(parser, value, "rope_scaling.rope_theta");
      } else {
        parser.skipChildren();
      }
    }
    return theta;
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

  private static int requiredPositive(Integer value, String name) {
    if (value == null || value <= 0) {
      throw malformed(name + " must be present and positive");
    }
    return value;
  }

  private static float requiredPositive(Float value, String name) {
    if (value == null || !(value > 0.0f) || !Float.isFinite(value)) {
      throw malformed(name + " must be present, finite, and positive");
    }
    return value;
  }

  private static final class Fields {
    private List<String> architectures;
    private Integer bosTokenId;
    private Integer eosTokenId;
    private Integer headDim;
    private String hiddenActivation;
    private Integer hiddenSize;
    private Integer intermediateSize;
    private Integer maxPosition;
    private String modelType;
    private Integer numAttentionHeads;
    private Integer numKvHeads;
    private Integer numLayers;
    private Float rmsNormEps;
    private Float ropeTheta;
    private Float scaledRopeTheta;
    private Integer slidingWindow;
    private boolean tieWordEmbeddings;
    private String torchDtype;
    private boolean useSlidingWindow;
    private Integer vocabSize;

    private Qwen2HuggingFaceConfig toConfig() {
      String resolvedModelType = modelType == null ? "qwen2" : modelType;
      if (!"qwen2".equals(resolvedModelType)) {
        throw malformed("Qwen 2 config requires model_type qwen2; got " + resolvedModelType);
      }
      if (hiddenActivation == null) {
        throw malformed("hidden_act must be present in Qwen 2 config");
      }
      String activation = hiddenActivation;
      if (!"silu".equals(activation)) {
        throw malformed("Qwen 2 runtime requires hidden_act silu; got " + activation);
      }
      int resolvedHiddenSize = requiredPositive(hiddenSize, "hidden_size");
      int resolvedHeads = requiredPositive(numAttentionHeads, "num_attention_heads");
      int resolvedKvHeads = numKvHeads == null ? resolvedHeads : numKvHeads;
      int resolvedHeadDim =
          headDim == null
              ? resolvedHiddenSize / resolvedHeads
              : requiredPositive(headDim, "head_dim");
      float resolvedRopeTheta =
          ropeTheta != null
              ? requiredPositive(ropeTheta, "rope_theta")
              : scaledRopeTheta != null
                  ? requiredPositive(scaledRopeTheta, "rope_scaling.rope_theta")
                  : 10_000.0f;
      int resolvedSlidingWindow =
          useSlidingWindow ? requiredPositive(slidingWindow, "sliding_window") : 0;
      List<String> resolvedArchitectures =
          architectures == null ? List.of("Qwen2ForCausalLM") : architectures;
      if (!resolvedArchitectures.contains("Qwen2ForCausalLM")) {
        throw malformed("Qwen 2 config architectures must include Qwen2ForCausalLM");
      }

      LlamaConfig model =
          new LlamaConfig(
              DecoderArchitecture.QWEN2,
              resolvedHiddenSize,
              requiredPositive(numLayers, "num_hidden_layers"),
              resolvedHeads,
              requiredPositive(resolvedKvHeads, "num_key_value_heads"),
              resolvedHeadDim,
              resolvedHeadDim,
              requiredPositive(vocabSize, "vocab_size"),
              requiredPositive(maxPosition, "max_position_embeddings"),
              requiredPositive(intermediateSize, "intermediate_size"),
              resolvedRopeTheta,
              1.0f,
              10_000.0f,
              requiredPositive(rmsNormEps, "rms_norm_eps"),
              resolvedSlidingWindow,
              6,
              0.0f);
      return new Qwen2HuggingFaceConfig(
          model,
          resolvedArchitectures,
          activation,
          torchDtype == null ? "" : torchDtype,
          tieWordEmbeddings,
          bosTokenId == null ? -1 : bosTokenId,
          eosTokenId == null ? -1 : eosTokenId);
    }
  }
}
