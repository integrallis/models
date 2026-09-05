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
package com.integrallis.models.backend.purejava.soprano;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.io.IOException;

/** Shape and synthesis constants for a Soprano 1.1 checkpoint. */
public record SopranoConfig(
    int hiddenSize,
    int intermediateSize,
    int layers,
    int attentionHeads,
    int kvHeads,
    int headDim,
    int vocabSize,
    int contextLength,
    float rmsNormEpsilon,
    float ropeTheta,
    int bosToken,
    int eosToken,
    int decoderDim,
    int decoderIntermediateSize,
    int decoderLayers,
    int decoderKernelSize,
    int fftSize,
    int hopLength,
    int upscale,
    int sampleRate) {

  private static final JsonFactory JSON = new JsonFactory();

  public SopranoConfig {
    requirePositive(hiddenSize, "hiddenSize");
    requirePositive(intermediateSize, "intermediateSize");
    requirePositive(layers, "layers");
    requirePositive(attentionHeads, "attentionHeads");
    requirePositive(kvHeads, "kvHeads");
    requirePositive(headDim, "headDim");
    requirePositive(vocabSize, "vocabSize");
    requirePositive(contextLength, "contextLength");
    if (!(rmsNormEpsilon > 0.0f) || !Float.isFinite(rmsNormEpsilon)) {
      throw new IllegalArgumentException("rmsNormEpsilon must be finite and positive");
    }
    if (!(ropeTheta > 0.0f) || !Float.isFinite(ropeTheta)) {
      throw new IllegalArgumentException("ropeTheta must be finite and positive");
    }
    requirePositive(decoderDim, "decoderDim");
    requirePositive(decoderIntermediateSize, "decoderIntermediateSize");
    requirePositive(decoderLayers, "decoderLayers");
    requirePositive(decoderKernelSize, "decoderKernelSize");
    requirePositive(fftSize, "fftSize");
    requirePositive(hopLength, "hopLength");
    requirePositive(upscale, "upscale");
    requirePositive(sampleRate, "sampleRate");
  }

  /** Parses the Hugging Face config embedded in the standalone GGUF. */
  public static SopranoConfig fromJson(String json) {
    String modelType = null;
    Integer hiddenSize = null;
    Integer intermediateSize = null;
    Integer layers = null;
    Integer attentionHeads = null;
    Integer kvHeads = null;
    Integer headDim = null;
    Integer vocabSize = null;
    Integer contextLength = null;
    Float rmsNormEpsilon = null;
    Float ropeTheta = null;
    Integer bosToken = null;
    Integer eosToken = null;
    try (JsonParser parser = JSON.createParser(json)) {
      if (parser.nextToken() != JsonToken.START_OBJECT) {
        throw new IllegalArgumentException("Soprano config must be a JSON object");
      }
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "model_type" -> modelType = parser.getValueAsString();
          case "hidden_size" -> hiddenSize = parser.getIntValue();
          case "intermediate_size" -> intermediateSize = parser.getIntValue();
          case "num_hidden_layers" -> layers = parser.getIntValue();
          case "num_attention_heads" -> attentionHeads = parser.getIntValue();
          case "num_key_value_heads" -> kvHeads = parser.getIntValue();
          case "head_dim" -> headDim = parser.getIntValue();
          case "vocab_size" -> vocabSize = parser.getIntValue();
          case "max_position_embeddings" -> contextLength = parser.getIntValue();
          case "rms_norm_eps" -> rmsNormEpsilon = parser.getFloatValue();
          case "rope_theta" -> ropeTheta = parser.getFloatValue();
          case "bos_token_id" -> bosToken = parser.getIntValue();
          case "eos_token_id" -> eosToken = parser.getIntValue();
          default -> parser.skipChildren();
        }
      }
    } catch (IOException failure) {
      throw new IllegalArgumentException("Cannot parse Soprano config", failure);
    }
    if (!"qwen3".equals(modelType)) {
      throw new IllegalArgumentException("Soprano config must declare model_type qwen3");
    }
    return new SopranoConfig(
        required(hiddenSize, "hidden_size"),
        required(intermediateSize, "intermediate_size"),
        required(layers, "num_hidden_layers"),
        required(attentionHeads, "num_attention_heads"),
        required(kvHeads, "num_key_value_heads"),
        required(headDim, "head_dim"),
        required(vocabSize, "vocab_size"),
        required(contextLength, "max_position_embeddings"),
        required(rmsNormEpsilon, "rms_norm_eps"),
        required(ropeTheta, "rope_theta"),
        required(bosToken, "bos_token_id"),
        required(eosToken, "eos_token_id"),
        768,
        2304,
        8,
        3,
        2048,
        512,
        4,
        32_000);
  }

  private static int required(Integer value, String name) {
    if (value == null) {
      throw new IllegalArgumentException("Soprano config is missing " + name);
    }
    return value;
  }

  private static float required(Float value, String name) {
    if (value == null) {
      throw new IllegalArgumentException("Soprano config is missing " + name);
    }
    return value;
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive: " + value);
    }
  }
}
