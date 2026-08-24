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
package com.integrallis.models.backend.purejava.cact;

import java.util.Arrays;
import java.util.Objects;

/** Incremental scalar reference forward pass for the Needle 2 `.cact` architecture. */
public final class Needle2ForwardPass {

  private static final int ENGRAM_SEED = 0x9E3779B9;
  private static final int ENGRAM_PRIME = 0x01000193;
  private static final float RMS_EPSILON = 1.0e-6f;

  private record EngramState(float[] key, float[] value) {}

  private final Needle2Weights weights;
  private final CactHeader config;
  private final int capacity;
  private final int modelWidth;
  private final int attentionWidth;
  private final int kvWidth;
  private final int[] tokenHistory;
  private final float[][][] keyCache;
  private final float[][][] valueCache;
  private final float[][][] engramValueHistory;
  private int checkpoint;

  public Needle2ForwardPass(Needle2Weights weights, int capacity) {
    this.weights = Objects.requireNonNull(weights, "weights");
    this.config = weights.header;
    if (capacity <= 0 || capacity > config.maximumSequenceLength()) {
      throw new IllegalArgumentException(
          "capacity must be in [1, " + config.maximumSequenceLength() + "]");
    }
    this.capacity = capacity;
    this.modelWidth = config.modelWidth();
    this.attentionWidth = Math.multiplyExact(config.queryHeadCount(), config.headWidth());
    this.kvWidth = Math.multiplyExact(config.kvHeadCount(), config.headWidth());
    this.tokenHistory = new int[capacity];
    this.keyCache = new float[config.layerCount()][capacity][kvWidth];
    this.valueCache = new float[config.layerCount()][capacity][kvWidth];
    this.engramValueHistory = new float[config.engramSites().size()][capacity][modelWidth];
  }

  public float[] forward(int token, int position) {
    if (token < 0 || token >= config.vocabularySize()) {
      throw new IllegalArgumentException("token outside Needle vocabulary: " + token);
    }
    if (position != checkpoint) {
      throw new IllegalArgumentException(
          "Needle forward position must equal checkpoint " + checkpoint + "; got " + position);
    }
    if (position >= capacity) {
      throw new IllegalArgumentException("Needle context capacity exceeded: " + capacity);
    }
    tokenHistory[position] = token;
    EngramState[] engrams = prepareEngrams(position);

    float[][] lanes = new float[config.mhcLanes()][modelWidth];
    float[] embedding = new float[modelWidth];
    weights.embedding.decodeRow(token, embedding);
    float embeddingScale = (float) Math.sqrt(modelWidth);
    for (int lane = 0; lane < lanes.length; lane++) {
      for (int column = 0; column < modelWidth; column++) {
        lanes[lane][column] = embedding[column] * embeddingScale;
      }
    }

    for (int layer = 0; layer < config.layerCount(); layer++) {
      lanes = executeLayer(lanes, layer, position, engrams);
    }

    float[] hidden = new float[modelWidth];
    for (float[] lane : lanes) {
      for (int column = 0; column < modelWidth; column++) {
        hidden[column] += lane[column] / lanes.length;
      }
    }
    float[] normalized = new float[modelWidth];
    Needle2Math.zeroCenteredRmsNorm(hidden, weights.finalNorm, normalized);
    float[] logits = multiply(weights.embedding, normalized);
    checkpoint++;
    return logits;
  }

  public int checkpoint() {
    return checkpoint;
  }

  public void rewind(int targetCheckpoint) {
    if (targetCheckpoint < 0 || targetCheckpoint > checkpoint) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + checkpoint + ": " + targetCheckpoint);
    }
    checkpoint = targetCheckpoint;
  }

  public void reset() {
    rewind(0);
  }

  private float[][] executeLayer(
      float[][] lanes, int layerIndex, int position, EngramState[] engrams) {
    int laneCount = config.mhcLanes();
    Needle2Weights.Layer layer = weights.layers[layerIndex];
    float[] normalizedLanes = rmsUnit(flatten(lanes));

    float[] preProjection = multiply(layer.mhcPhiPre(), normalizedLanes);
    float[] preWeights = new float[laneCount];
    int activeLane = layerIndex % laneCount;
    for (int lane = 0; lane < laneCount; lane++) {
      float offset = lane == activeLane ? 4.0f : -4.0f;
      preWeights[lane] =
          Needle2Math.sigmoid(
              weights.mhcAPre[layerIndex] * preProjection[lane]
                  + weights.mhcBPre[layerIndex * laneCount + lane]
                  + offset);
    }

    float[] combined = weightedLanes(lanes, preWeights);
    float[] blockInput = combined;
    int engramSite = config.engramSites().indexOf(layerIndex);
    if (engramSite >= 0) {
      EngramState engram = engrams[engramSite];
      float alpha =
          Needle2Math.sigmoid(
              dot(rmsUnit(combined), rmsUnit(engram.key())) / (float) Math.sqrt(modelWidth));
      blockInput = combined.clone();
      addScaled(blockInput, engram.value(), alpha);
    }

    float[] blockOutput = executeBlock(layer, layerIndex, blockInput, position);
    float[] residual = subtract(blockOutput, combined);

    float[] postProjection = multiply(layer.mhcPhiPost(), normalizedLanes);
    float[] postWeights = new float[laneCount];
    for (int lane = 0; lane < laneCount; lane++) {
      float offset = lane == activeLane ? 0.0f : -4.0f;
      postWeights[lane] =
          2.0f
              * Needle2Math.sigmoid(
                  weights.mhcAPost[layerIndex] * postProjection[lane]
                      + weights.mhcBPost[layerIndex * laneCount + lane]
                      + offset);
    }

    float[] mixing = multiply(layer.mhcPhiResidual(), normalizedLanes);
    int matrixOffset = layerIndex * laneCount * laneCount;
    for (int index = 0; index < mixing.length; index++) {
      mixing[index] =
          weights.mhcAResidual[layerIndex] * mixing[index]
              + weights.mhcBResidual[matrixOffset + index];
    }
    Needle2Math.sinkhorn(mixing, laneCount, 20);

    float[][] next = new float[laneCount][modelWidth];
    for (int outputLane = 0; outputLane < laneCount; outputLane++) {
      for (int inputLane = 0; inputLane < laneCount; inputLane++) {
        addScaled(next[outputLane], lanes[inputLane], mixing[outputLane * laneCount + inputLane]);
      }
      addScaled(next[outputLane], residual, postWeights[outputLane]);
    }
    return next;
  }

  private float[] executeBlock(
      Needle2Weights.Layer layer, int layerIndex, float[] input, int position) {
    float[] attentionInput = new float[modelWidth];
    Needle2Math.zeroCenteredRmsNorm(input, layer.normIn(), attentionInput);

    float[] query = multiply(layer.query(), attentionInput);
    float[] key = multiply(layer.key(), attentionInput);
    float[] value = multiply(layer.value(), attentionInput);
    normalizeHeads(query, config.queryHeadCount(), layer.queryNorm());
    normalizeHeads(key, config.kvHeadCount(), layer.keyNorm());
    applyRope(query, config.queryHeadCount(), position);
    applyRope(key, config.kvHeadCount(), position);
    System.arraycopy(key, 0, keyCache[layerIndex][position], 0, kvWidth);
    System.arraycopy(value, 0, valueCache[layerIndex][position], 0, kvWidth);

    float[] attended = attention(layerIndex, query, position);
    float[] gate = multiply(layer.attentionGateProjection(), attentionInput);
    for (int index = 0; index < attended.length; index++) {
      attended[index] *= Needle2Math.sigmoid(gate[index]);
    }
    float[] projected = multiply(layer.output(), attended);
    float[] normalizedAttention = new float[modelWidth];
    Needle2Math.zeroCenteredRmsNorm(projected, layer.postAttentionNorm(), normalizedAttention);

    float[] output = input.clone();
    addScaled(output, normalizedAttention, Needle2Math.sigmoid(layer.attentionGate()));
    float[] hadamardInput = new float[modelWidth];
    Needle2Math.zeroCenteredRmsNorm(output, layer.preHadamardNorm(), hadamardInput);
    float[] hadamardOutput = new float[modelWidth];
    Needle2Math.hadamardMlp(
        hadamardInput,
        layer.hadamardD1(),
        layer.hadamardD2(),
        layer.hadamardD3(),
        modelWidth,
        hadamardOutput);
    addScaled(output, hadamardOutput, 1.0f);
    return output;
  }

  private float[] attention(int layer, float[] query, int position) {
    int queryHeads = config.queryHeadCount();
    int kvHeads = config.kvHeadCount();
    int repeats = queryHeads / kvHeads;
    int headWidth = config.headWidth();
    int first = config.kvWindow() == 0 ? 0 : Math.max(0, position - config.kvWindow() + 1);
    int attendedPositions = position - first + 1;
    float[] result = new float[attentionWidth];
    float[] scores = new float[attendedPositions];
    float scale = (float) Math.sqrt(headWidth);
    for (int queryHead = 0; queryHead < queryHeads; queryHead++) {
      int kvHead = queryHead / repeats;
      int queryOffset = queryHead * headWidth;
      int kvOffset = kvHead * headWidth;
      float maximum = Float.NEGATIVE_INFINITY;
      for (int index = 0; index < attendedPositions; index++) {
        int cachedPosition = first + index;
        float score =
            dot(query, queryOffset, keyCache[layer][cachedPosition], kvOffset, headWidth) / scale;
        scores[index] = score;
        maximum = Math.max(maximum, score);
      }
      float denominator = 0.0f;
      for (int index = 0; index < attendedPositions; index++) {
        scores[index] = (float) Math.exp(scores[index] - maximum);
        denominator += scores[index];
      }
      for (int index = 0; index < attendedPositions; index++) {
        float probability = scores[index] / denominator;
        int cachedPosition = first + index;
        for (int column = 0; column < headWidth; column++) {
          result[queryOffset + column] +=
              probability * valueCache[layer][cachedPosition][kvOffset + column];
        }
      }
    }
    return result;
  }

  private EngramState[] prepareEngrams(int position) {
    EngramState[] states = new EngramState[weights.engrams.length];
    for (int site = 0; site < states.length; site++) {
      Needle2Weights.Engram weightsAtSite = weights.engrams[site];
      int tables = config.engramTableCount();
      int subDimension = config.engramSubDimension();
      int headsPerOrder = tables / config.engramOrders().size();
      float[] concatenated = new float[Math.multiplyExact(tables, subDimension)];
      float[] tableRow = new float[subDimension];
      int table = 0;
      for (int orderIndex = 0; orderIndex < config.engramOrders().size(); orderIndex++) {
        int order = config.engramOrders().get(orderIndex);
        for (int head = 0; head < headsPerOrder; head++, table++) {
          if (position + 1 < order) {
            continue;
          }
          int slot = engramIndex(position, order, table, config.engramSlots());
          weightsAtSite.tables().decodeRow(table * config.engramSlots() + slot, tableRow);
          System.arraycopy(tableRow, 0, concatenated, table * subDimension, subDimension);
        }
      }
      float[] key = multiply(weightsAtSite.key(), concatenated);
      float[] rawValue = multiply(weightsAtSite.value(), concatenated);
      System.arraycopy(rawValue, 0, engramValueHistory[site][position], 0, modelWidth);
      float[] convolvedValue = new float[modelWidth];
      for (int tap = 0; tap < config.engramConvolutionTaps(); tap++) {
        int sourcePosition = position - tap * config.engramConvolutionDilation();
        if (sourcePosition < 0) {
          continue;
        }
        int tapOffset = tap * modelWidth;
        for (int column = 0; column < modelWidth; column++) {
          convolvedValue[column] +=
              weightsAtSite.taps()[tapOffset + column]
                  * engramValueHistory[site][sourcePosition][column];
        }
      }
      states[site] = new EngramState(key, convolvedValue);
    }
    return states;
  }

  private int engramIndex(int position, int order, int table, int slots) {
    int accumulator = ENGRAM_SEED * (table + 1);
    for (int offset = 0; offset < order; offset++) {
      accumulator = (accumulator ^ tokenHistory[position - offset]) * ENGRAM_PRIME;
    }
    accumulator ^= accumulator >>> 15;
    return Integer.remainderUnsigned(accumulator, slots);
  }

  private void normalizeHeads(float[] vector, int heads, float[] scale) {
    int headWidth = config.headWidth();
    float[] input = new float[headWidth];
    float[] output = new float[headWidth];
    for (int head = 0; head < heads; head++) {
      int offset = head * headWidth;
      System.arraycopy(vector, offset, input, 0, headWidth);
      Needle2Math.zeroCenteredRmsNorm(input, scale, output);
      System.arraycopy(output, 0, vector, offset, headWidth);
    }
  }

  private void applyRope(float[] vector, int heads, int position) {
    for (int head = 0; head < heads; head++) {
      Needle2Math.applyRope(
          vector, head * config.headWidth(), config.headWidth(), position, config.ropeTheta());
    }
  }

  private static float[] multiply(CactCqMatrix matrix, float[] input) {
    float[] output = new float[matrix.rows()];
    matrix.multiply(matrix.prepare(input), output);
    return output;
  }

  private static float[] flatten(float[][] values) {
    int columns = values[0].length;
    float[] result = new float[Math.multiplyExact(values.length, columns)];
    for (int row = 0; row < values.length; row++) {
      System.arraycopy(values[row], 0, result, row * columns, columns);
    }
    return result;
  }

  private static float[] weightedLanes(float[][] lanes, float[] weights) {
    float[] result = new float[lanes[0].length];
    for (int lane = 0; lane < lanes.length; lane++) {
      addScaled(result, lanes[lane], weights[lane]);
    }
    return result;
  }

  private static float[] rmsUnit(float[] input) {
    float squared = 0.0f;
    for (float value : input) {
      squared += value * value;
    }
    float inverse = (float) (1.0 / Math.sqrt(squared / input.length + RMS_EPSILON));
    float[] output = new float[input.length];
    for (int index = 0; index < input.length; index++) {
      output[index] = input[index] * inverse;
    }
    return output;
  }

  private static float dot(float[] left, float[] right) {
    return dot(left, 0, right, 0, left.length);
  }

  private static float dot(
      float[] left, int leftOffset, float[] right, int rightOffset, int length) {
    float result = 0.0f;
    for (int index = 0; index < length; index++) {
      result += left[leftOffset + index] * right[rightOffset + index];
    }
    return result;
  }

  private static void addScaled(float[] target, float[] value, float scale) {
    for (int index = 0; index < target.length; index++) {
      target[index] += value[index] * scale;
    }
  }

  private static float[] subtract(float[] left, float[] right) {
    float[] result = Arrays.copyOf(left, left.length);
    for (int index = 0; index < result.length; index++) {
      result[index] -= right[index];
    }
    return result;
  }
}
