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
package com.integrallis.models.backend.purejava.deepseek2;

import java.util.ArrayList;
import java.util.List;

/**
 * An independent scalar DeepSeek-V2, written from the reference rather than from the graph.
 *
 * <p>Plain loops, fresh arrays, keys and values in lists. It shares no code with {@link
 * Deepseek2ForwardPass}, including the rotary table and the expert routing, so agreement is
 * evidence about the algorithm and not about a helper they both call.
 */
final class Deepseek2ScalarReference {

  private final Deepseek2ToyModel model;
  private final Deepseek2Config config;
  private final List<List<float[]>> keys = new ArrayList<>();
  private final List<List<float[]>> values = new ArrayList<>();
  private final float[] ropeFrequencies;
  private int position;

  Deepseek2ScalarReference(Deepseek2ToyModel model, Deepseek2Config config) {
    this.model = model;
    this.config = config;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      keys.add(new ArrayList<>());
      values.add(new ArrayList<>());
    }
    this.ropeFrequencies = yarnFrequencies();
  }

  /**
   * YaRN's frequency blend, recomputed here rather than taken from RotaryTable.
   *
   * <p>{@code low} and {@code high} bound a ramp over the pair index; below it the frequency is
   * untouched, above it divided by the factor, between it interpolated.
   */
  private float[] yarnFrequencies() {
    int rotaryDim = config.ropeDimension();
    double theta = config.ropeTheta();
    double factor = config.ropeScalingFactor();
    double denominator = 2.0 * Math.log(theta);
    double low =
        rotaryDim
            * Math.log(config.ropeOriginalContext() / (Deepseek2Config.BETA_FAST * 2.0 * Math.PI))
            / denominator;
    double high =
        rotaryDim
            * Math.log(config.ropeOriginalContext() / (Deepseek2Config.BETA_SLOW * 2.0 * Math.PI))
            / denominator;
    low = Math.max(low, 0.0);
    high = Math.min(high, rotaryDim - 1.0);
    if (low == high) {
      high += 0.001;
    }
    float[] frequencies = new float[rotaryDim / 2];
    for (int pair = 0; pair < frequencies.length; pair++) {
      double inverse = 1.0 / Math.pow(theta, (double) (pair * 2) / rotaryDim);
      double ramp = Math.max(0.0, Math.min(1.0, (pair - low) / (high - low)));
      frequencies[pair] = (float) ((inverse / factor) * ramp + inverse * (1.0 - ramp));
    }
    return frequencies;
  }

  float[] forward(int token) {
    int dim = config.embeddingDim();
    float[] state = row(model.tensor("token_embd.weight"), token, dim);

    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "blk." + layer + ".";
      float[] normed = rms(state, model.tensor(prefix + "attn_norm.weight"));
      float[] attention = attention(layer, prefix, normed);
      for (int index = 0; index < dim; index++) {
        state[index] += attention[index];
      }

      float[] ffnInput = rms(state, model.tensor(prefix + "ffn_norm.weight"));
      float[] ffn =
          config.usesMixtureOfExperts(layer)
              ? routedFeedForward(prefix, ffnInput)
              : denseFeedForward(prefix, ffnInput);
      for (int index = 0; index < dim; index++) {
        state[index] += ffn[index];
      }
    }

    state = rms(state, model.tensor("output_norm.weight"));
    position++;
    return matmul(model.tensor("token_embd.weight"), config.vocabSize(), dim, state);
  }

  private float[] attention(int layer, String prefix, float[] normed) {
    int dim = config.embeddingDim();
    int heads = config.numHeads();
    int keyLength = config.keyLength();
    int noRope = config.noRopeDimension();
    int ropeDim = config.ropeDimension();
    int valueLength = config.valueLength();
    int rank = config.kvLoraRank();

    float[] query = matmul(model.tensor(prefix + "attn_q.weight"), config.queryDim(), dim, normed);
    float[] compressed =
        matmul(
            model.tensor(prefix + "attn_kv_a_mqa.weight"),
            config.compressedKeyValueDim(),
            dim,
            normed);
    float[] latent = slice(compressed, 0, rank);
    float[] ropeKey = slice(compressed, rank, ropeDim);

    // The query's rotary half sits after its no-rope half within each head.
    for (int head = 0; head < heads; head++) {
      float[] rotated = rope(slice(query, head * keyLength + noRope, ropeDim));
      System.arraycopy(rotated, 0, query, head * keyLength + noRope, ropeDim);
    }
    ropeKey = rope(ropeKey);

    latent = rms(latent, model.tensor(prefix + "attn_kv_a_norm.weight"));
    float[] decompressed =
        matmul(
            model.tensor(prefix + "attn_kv_b.weight"),
            config.decompressedKeyValueDim(),
            rank,
            latent);

    float[] key = new float[heads * keyLength];
    float[] value = new float[heads * valueLength];
    int stride = noRope + valueLength;
    for (int head = 0; head < heads; head++) {
      System.arraycopy(decompressed, head * stride, key, head * keyLength, noRope);
      // The SAME rotary part on every head.
      System.arraycopy(ropeKey, 0, key, head * keyLength + noRope, ropeDim);
      System.arraycopy(
          decompressed, head * stride + noRope, value, head * valueLength, valueLength);
    }
    keys.get(layer).add(key);
    values.get(layer).add(value);

    float scale = config.attentionScale();
    float[] attended = new float[heads * valueLength];
    for (int head = 0; head < heads; head++) {
      List<Float> scores = new ArrayList<>();
      for (int at = 0; at <= position; at++) {
        float total = 0.0f;
        for (int index = 0; index < keyLength; index++) {
          total +=
              query[head * keyLength + index] * keys.get(layer).get(at)[head * keyLength + index];
        }
        scores.add(total * scale);
      }
      softmax(scores);
      for (int at = 0; at <= position; at++) {
        float probability = scores.get(at - 0);
        for (int index = 0; index < valueLength; index++) {
          attended[head * valueLength + index] +=
              probability * values.get(layer).get(at)[head * valueLength + index];
        }
      }
    }
    return matmul(
        model.tensor(prefix + "attn_output.weight"), dim, config.cachedValueDim(), attended);
  }

  private float[] denseFeedForward(String prefix, float[] input) {
    int dim = config.embeddingDim();
    float[] gate = matmul(model.tensor(prefix + "ffn_gate.weight"), config.hiddenDim(), dim, input);
    float[] up = matmul(model.tensor(prefix + "ffn_up.weight"), config.hiddenDim(), dim, input);
    for (int index = 0; index < gate.length; index++) {
      gate[index] = silu(gate[index]) * up[index];
    }
    return matmul(model.tensor(prefix + "ffn_down.weight"), dim, config.hiddenDim(), gate);
  }

  private float[] routedFeedForward(String prefix, float[] input) {
    int dim = config.embeddingDim();
    int experts = config.numExperts();
    int hidden = config.expertHiddenDim();
    float[] logits = matmul(model.tensor(prefix + "ffn_gate_inp.weight"), experts, dim, input);

    // Softmax over ALL experts; the chosen weights are used as they are, not renormalised.
    List<Float> probabilities = new ArrayList<>();
    for (float logit : logits) {
      probabilities.add(logit);
    }
    softmax(probabilities);
    Integer[] order = new Integer[experts];
    for (int index = 0; index < experts; index++) {
      order[index] = index;
    }
    java.util.Arrays.sort(
        order,
        (a, b) -> {
          int compared = Float.compare(probabilities.get(b), probabilities.get(a));
          return compared != 0 ? compared : Integer.compare(a, b);
        });

    float[] result = new float[dim];
    for (int slot = 0; slot < config.numExpertsUsed(); slot++) {
      int expert = order[slot];
      float weight = probabilities.get(expert);
      float[] gate = expertSlice(prefix + "ffn_gate_exps.weight", expert, hidden, dim, input);
      float[] up = expertSlice(prefix + "ffn_up_exps.weight", expert, hidden, dim, input);
      for (int index = 0; index < hidden; index++) {
        gate[index] = silu(gate[index]) * up[index];
      }
      float[] down = expertSlice(prefix + "ffn_down_exps.weight", expert, dim, hidden, gate);
      for (int index = 0; index < dim; index++) {
        result[index] += weight * down[index];
      }
    }

    int shared = config.sharedExpertHiddenDim();
    float[] sharedGate = matmul(model.tensor(prefix + "ffn_gate_shexp.weight"), shared, dim, input);
    float[] sharedUp = matmul(model.tensor(prefix + "ffn_up_shexp.weight"), shared, dim, input);
    for (int index = 0; index < shared; index++) {
      sharedGate[index] = silu(sharedGate[index]) * sharedUp[index];
    }
    float[] sharedDown =
        matmul(model.tensor(prefix + "ffn_down_shexp.weight"), dim, shared, sharedGate);
    for (int index = 0; index < dim; index++) {
      result[index] += sharedDown[index];
    }
    return result;
  }

  private float[] expertSlice(String name, int expert, int rows, int columns, float[] input) {
    float[] all = model.tensor(name);
    float[] result = new float[rows];
    int base = expert * rows * columns;
    for (int row = 0; row < rows; row++) {
      float total = 0.0f;
      for (int column = 0; column < columns; column++) {
        total += all[base + row * columns + column] * input[column];
      }
      result[row] = total;
    }
    return result;
  }

  private float[] rope(float[] values) {
    int half = values.length / 2;
    float magnitude = config.ropeAttentionFactor();
    float[] result = values.clone();
    for (int pair = 0; pair < half; pair++) {
      double angle = position * (double) ropeFrequencies[pair];
      float cos = (float) (magnitude * Math.cos(angle));
      float sin = (float) (magnitude * Math.sin(angle));
      float low = values[pair];
      float high = values[pair + half];
      result[pair] = low * cos - high * sin;
      result[pair + half] = low * sin + high * cos;
    }
    return result;
  }

  private float[] rms(float[] values, float[] weight) {
    double total = 0.0;
    for (float value : values) {
      total += (double) value * value;
    }
    float inverse = (float) (1.0 / Math.sqrt(total / values.length + config.rmsNormEpsilon()));
    float[] result = new float[values.length];
    for (int index = 0; index < values.length; index++) {
      result[index] = values[index] * inverse * weight[index];
    }
    return result;
  }

  private static float silu(float value) {
    return (float) (value / (1.0 + Math.exp(-value)));
  }

  private static void softmax(List<Float> values) {
    float largest = Float.NEGATIVE_INFINITY;
    for (float value : values) {
      largest = Math.max(largest, value);
    }
    double total = 0.0;
    for (int index = 0; index < values.size(); index++) {
      double exponent = Math.exp(values.get(index) - largest);
      values.set(index, (float) exponent);
      total += exponent;
    }
    for (int index = 0; index < values.size(); index++) {
      values.set(index, (float) (values.get(index) / total));
    }
  }

  private static float[] slice(float[] values, int offset, int length) {
    float[] result = new float[length];
    System.arraycopy(values, offset, result, 0, length);
    return result;
  }

  private static float[] row(float[] table, int index, int width) {
    return slice(table, index * width, width);
  }

  private static float[] matmul(float[] weight, int rows, int columns, float[] input) {
    float[] result = new float[rows];
    for (int row = 0; row < rows; row++) {
      float total = 0.0f;
      for (int column = 0; column < columns; column++) {
        total += weight[row * columns + column] * input[column];
      }
      result[row] = total;
    }
    return result;
  }
}
