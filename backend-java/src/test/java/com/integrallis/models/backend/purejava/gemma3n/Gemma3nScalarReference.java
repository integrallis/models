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
package com.integrallis.models.backend.purejava.gemma3n;

import java.util.ArrayList;
import java.util.List;

/**
 * An independent scalar Gemma 3n, written from the specification rather than from the graph.
 *
 * <p>Deliberately naive: plain loops, fresh arrays, no buffer reuse, no vectorisation, no cache
 * abstraction -- keys and values are kept as lists of arrays per layer. The point is that it shares
 * no code with {@link Gemma3nForwardPass}, so the two agreeing is evidence about the algorithm and
 * not about a helper both of them call.
 *
 * <p>It was written from {@code GEMMA3N-SPEC.md}, which was in turn transcribed from {@code
 * llama.cpp/src/models/gemma3n.cpp}. Where the two implementations disagreed during development,
 * the reference was checked against the C++ again before either was changed.
 */
final class Gemma3nScalarReference {

  private final Gemma3nToyModel model;
  private final Gemma3nConfig config;
  private final List<List<float[]>> keys = new ArrayList<>();
  private final List<List<float[]>> values = new ArrayList<>();
  private int position;

  Gemma3nScalarReference(Gemma3nToyModel model, Gemma3nConfig config) {
    this.model = model;
    this.config = config;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      keys.add(new ArrayList<>());
      values.add(new ArrayList<>());
    }
  }

  float[] forward(int token) {
    int dim = config.embeddingDim();
    int altup = config.altupInputs();
    int act = config.altupActiveIndex();
    int perLayer = config.perLayerEmbeddingDim();

    // --- the parallel streams, built from the scaled embedding
    float[][] streams = new float[altup][];
    float[] embedding = row(model.tensor("token_embd.weight"), token, dim);
    scale(embedding, (float) Math.sqrt(dim));
    streams[act] = embedding.clone();
    float targetMagnitude = magnitude(embedding);
    int slice = 0;
    for (int stream = 0; stream < altup; stream++) {
      if (stream == act) {
        continue;
      }
      float[] projection =
          matmulSlice(model.tensor("altup_proj.weight"), slice++, dim, dim, embedding);
      scale(projection, targetMagnitude / magnitude(projection));
      streams[stream] = projection;
    }

    // --- the second embedding table, one slice per layer
    float[] table =
        row(model.tensor("per_layer_token_embd.weight"), token, perLayer * config.numLayers());
    float[] projected =
        matmul(
            model.tensor("per_layer_model_proj.weight"),
            perLayer * config.numLayers(),
            dim,
            embedding);
    scale(projected, (float) (1.0 / Math.sqrt(dim)));
    float[][] perLayerInputs = new float[config.numLayers()][];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      float[] sliceValues = new float[perLayer];
      System.arraycopy(projected, layer * perLayer, sliceValues, 0, perLayer);
      sliceValues = rms(sliceValues, model.tensor("per_layer_proj_norm.weight"));
      float[] combined = new float[perLayer];
      for (int index = 0; index < perLayer; index++) {
        combined[index] =
            (float) ((sliceValues[index] + table[layer * perLayer + index]) / Math.sqrt(2.0));
      }
      perLayerInputs[layer] = combined;
    }

    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "blk." + layer + ".";

      // --- predict
      float[] modalities = modalities(streams[act], prefix);
      float[] predictCoefficients =
          matmul(
              model.tensor(prefix + "altup_predict_coef.weight"), altup * altup, altup, modalities);
      float[][] predictions = new float[altup][];
      for (int stream = 0; stream < altup; stream++) {
        float[] prediction = streams[stream].clone();
        for (int source = 0; source < altup; source++) {
          float weight = predictCoefficients[stream * altup + source];
          for (int index = 0; index < dim; index++) {
            prediction[index] += weight * streams[source][index];
          }
        }
        predictions[stream] = prediction;
      }
      float[] activePrediction = predictions[act];

      // --- attention and LAuReL over the normed active prediction
      float[] normed = rms(activePrediction, model.tensor(prefix + "attn_norm.weight"));
      float[] laurel = laurel(normed, prefix);
      float[] attention = attention(layer, prefix, normed);
      attention = rms(attention, model.tensor(prefix + "post_attention_norm.weight"));
      float[] attnLaurel = new float[dim];
      for (int index = 0; index < dim; index++) {
        attnLaurel[index] =
            (float) ((attention[index] + activePrediction[index] + laurel[index]) / Math.sqrt(2.0));
      }

      // --- feed-forward
      float[] ffnInput = rms(attnLaurel, model.tensor(prefix + "ffn_norm.weight"));
      float[] gate =
          matmul(model.tensor(prefix + "ffn_gate.weight"), config.hiddenDim(), dim, ffnInput);
      float[] up =
          matmul(model.tensor(prefix + "ffn_up.weight"), config.hiddenDim(), dim, ffnInput);
      if (config.usesActivationSparsity(layer)) {
        gate = gaussianTopk(gate, config.activationSparsityScale(layer));
      }
      for (int index = 0; index < gate.length; index++) {
        gate[index] = gelu(gate[index]) * up[index];
      }
      float[] ffnOut =
          matmul(model.tensor(prefix + "ffn_down.weight"), dim, config.hiddenDim(), gate);
      ffnOut = rms(ffnOut, model.tensor(prefix + "post_ffw_norm.weight"));
      float[] gated = new float[dim];
      for (int index = 0; index < dim; index++) {
        gated[index] = ffnOut[index] + attnLaurel[index];
      }

      // --- correct
      float[] correctModalities = modalities(gated, prefix);
      float[] correctCoefficients =
          matmul(
              model.tensor(prefix + "altup_correct_coef.weight"), altup, altup, correctModalities);
      float[][] corrected = new float[altup][];
      for (int stream = 0; stream < altup; stream++) {
        float weight = correctCoefficients[stream] + 1.0f;
        float[] result = new float[dim];
        for (int index = 0; index < dim; index++) {
          float innovation = gated[index] - activePrediction[index];
          result[index] = predictions[stream][index] + innovation * weight;
        }
        corrected[stream] = result;
      }

      // --- per-layer input, into the inactive streams only
      float[] first = new float[dim];
      float[] correctScale = model.tensor(prefix + "altup_correct_scale.weight");
      for (int index = 0; index < dim; index++) {
        first[index] = corrected[act][index] * correctScale[index];
      }
      float[] gatedPerLayer =
          matmul(model.tensor(prefix + "inp_gate.weight"), perLayer, dim, first);
      for (int index = 0; index < perLayer; index++) {
        gatedPerLayer[index] = gelu(gatedPerLayer[index]) * perLayerInputs[layer][index];
      }
      float[] contribution =
          matmul(model.tensor(prefix + "proj.weight"), dim, perLayer, gatedPerLayer);
      contribution = rms(contribution, model.tensor(prefix + "post_norm.weight"));
      for (int stream = 0; stream < altup; stream++) {
        if (stream == act) {
          continue;
        }
        for (int index = 0; index < dim; index++) {
          corrected[stream][index] += contribution[index];
        }
      }
      streams = corrected;
    }

    // --- merge the streams back to one
    float target = magnitude(streams[act]);
    float[] merged = streams[act].clone();
    int unembedSlice = 0;
    for (int stream = 0; stream < altup; stream++) {
      if (stream == act) {
        continue;
      }
      float[] projection =
          matmulSlice(
              model.tensor("altup_unembd_proj.weight"), unembedSlice++, dim, dim, streams[stream]);
      scale(projection, target / magnitude(projection));
      for (int index = 0; index < dim; index++) {
        merged[index] += projection[index];
      }
    }
    scale(merged, 1.0f / altup);
    merged = rms(merged, model.tensor("output_norm.weight"));
    position++;
    float[] logits = matmul(model.tensor("token_embd.weight"), config.vocabSize(), dim, merged);
    // The bounded final-logit transform, written out here rather than calling the production helper
    // so
    // this stays an independent implementation. The reference graph's last three nodes are a scale,
    // a
    // tanh and a scale around the output projection.
    float cap = config.finalLogitSoftcap();
    for (int index = 0; index < logits.length; index++) {
      logits[index] = cap * (float) Math.tanh(logits[index] / cap);
    }
    return logits;
  }

  private float[] modalities(float[] source, String prefix) {
    int dim = config.embeddingDim();
    float[] normed = rms(source, model.tensor(prefix + "altup_router_norm.weight"));
    scale(normed, 1.0f / dim);
    float[] routed =
        matmul(model.tensor(prefix + "altup_router.weight"), config.altupInputs(), dim, normed);
    for (int index = 0; index < routed.length; index++) {
      routed[index] = (float) Math.tanh(routed[index]);
    }
    return routed;
  }

  private float[] laurel(float[] normed, String prefix) {
    int dim = config.embeddingDim();
    int rank = Gemma3nToyModel.LAUREL_RANK;
    float[] low = matmul(model.tensor(prefix + "laurel_l.weight"), rank, dim, normed);
    float[] high = matmul(model.tensor(prefix + "laurel_r.weight"), dim, rank, low);
    high = rms(high, model.tensor(prefix + "laurel_post_norm.weight"));
    float[] result = new float[dim];
    for (int index = 0; index < dim; index++) {
      result[index] = high[index] + normed[index];
    }
    return result;
  }

  private float[] attention(int layer, String prefix, float[] normed) {
    int dim = config.embeddingDim();
    int headDim = config.headDim();
    int heads = config.numHeads();
    int kvHeads = config.numKvHeads();
    int kvDim = kvHeads * headDim;

    float[] query = matmul(model.tensor(prefix + "attn_q.weight"), heads * headDim, dim, normed);
    for (int head = 0; head < heads; head++) {
      float[] slice = slice(query, head * headDim, headDim);
      slice = rms(slice, model.tensor(prefix + "attn_q_norm.weight"));
      slice = rope(slice, position);
      System.arraycopy(slice, 0, query, head * headDim, headDim);
    }

    int kvLayer = config.kvSourceLayer(layer);
    if (config.ownsKvCache(layer)) {
      String own = "blk." + layer + ".";
      float[] key = matmul(model.tensor(own + "attn_k.weight"), kvDim, dim, normed);
      float[] value = matmul(model.tensor(own + "attn_v.weight"), kvDim, dim, normed);
      for (int head = 0; head < kvHeads; head++) {
        float[] keySlice =
            rms(slice(key, head * headDim, headDim), model.tensor(own + "attn_k_norm.weight"));
        keySlice = rope(keySlice, position);
        System.arraycopy(keySlice, 0, key, head * headDim, headDim);
        // No weight vector on the value normalisation.
        float[] valueSlice = rmsWithoutWeight(slice(value, head * headDim, headDim));
        System.arraycopy(valueSlice, 0, value, head * headDim, headDim);
      }
      keys.get(kvLayer).add(key);
      values.get(kvLayer).add(value);
    }

    int first =
        config.usesSlidingWindow(kvLayer) ? Math.max(0, position - config.slidingWindow() + 1) : 0;
    List<float[]> cachedKeys = keys.get(kvLayer);
    List<float[]> cachedValues = values.get(kvLayer);
    float[] attended = new float[heads * headDim];
    int groupSize = heads / kvHeads;
    for (int head = 0; head < heads; head++) {
      int kvHead = head / groupSize;
      List<Float> scores = new ArrayList<>();
      for (int at = first; at <= position; at++) {
        float total = 0.0f;
        for (int index = 0; index < headDim; index++) {
          total += query[head * headDim + index] * cachedKeys.get(at)[kvHead * headDim + index];
        }
        scores.add(total * Gemma3nConfig.ATTENTION_SCALE);
      }
      softmax(scores);
      for (int at = first; at <= position; at++) {
        float probability = scores.get(at - first);
        for (int index = 0; index < headDim; index++) {
          attended[head * headDim + index] +=
              probability * cachedValues.get(at)[kvHead * headDim + index];
        }
      }
    }
    return matmul(model.tensor(prefix + "attn_output.weight"), dim, heads * headDim, attended);
  }

  // --- plain arithmetic, written out rather than delegated

  private float[] rope(float[] values, int at) {
    int half = values.length / 2;
    float[] result = values.clone();
    for (int pair = 0; pair < half; pair++) {
      double frequency = 1.0 / Math.pow(Gemma3nToyModel.ROPE_THETA, (2.0 * pair) / values.length);
      double angle = at * frequency;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);
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

  private float[] rmsWithoutWeight(float[] values) {
    double total = 0.0;
    for (float value : values) {
      total += (double) value * value;
    }
    float inverse = (float) (1.0 / Math.sqrt(total / values.length + config.rmsNormEpsilon()));
    float[] result = new float[values.length];
    for (int index = 0; index < values.length; index++) {
      result[index] = values[index] * inverse;
    }
    return result;
  }

  private static float[] gaussianTopk(float[] values, float multiplier) {
    double total = 0.0;
    for (float value : values) {
      total += value;
    }
    float mean = (float) (total / values.length);
    double variance = 0.0;
    for (float value : values) {
      variance += (value - mean) * (double) (value - mean);
    }
    float deviation = (float) Math.sqrt(variance / (values.length - 1));
    float cutoff = mean + deviation * multiplier;
    float[] result = new float[values.length];
    for (int index = 0; index < values.length; index++) {
      result[index] = Math.max(0.0f, values[index] - cutoff);
    }
    return result;
  }

  /** llama.cpp's tanh GELU approximation, which is what TensorOps.gelu implements. */
  private static float gelu(float value) {
    double inner = 0.797884560802865 * (value + 0.044715 * value * value * value);
    return (float) (0.5 * value * (1.0 + Math.tanh(inner)));
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

  private static float magnitude(float[] values) {
    double total = 0.0;
    for (float value : values) {
      total += (double) value * value;
    }
    return (float) Math.sqrt(total);
  }

  private static void scale(float[] values, float factor) {
    for (int index = 0; index < values.length; index++) {
      values[index] *= factor;
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

  /** Row-major {@code [rows, columns]} times a vector, as GGUF stores its matrices. */
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

  /** One {@code [rows, columns]} slice of a stacked three-dimensional tensor. */
  private static float[] matmulSlice(
      float[] weight, int slice, int rows, int columns, float[] input) {
    float[] result = new float[rows];
    int base = slice * rows * columns;
    for (int row = 0; row < rows; row++) {
      float total = 0.0f;
      for (int column = 0; column < columns; column++) {
        total += weight[base + row * columns + column] * input[column];
      }
      result[row] = total;
    }
    return result;
  }
}
