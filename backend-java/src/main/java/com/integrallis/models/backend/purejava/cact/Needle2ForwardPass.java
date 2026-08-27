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

import com.integrallis.vectors.core.RotatedCodebookMatrix;
import java.util.Arrays;
import java.util.Objects;

/** Incremental pure-Java forward pass for the Needle 2 `.cact` architecture. */
public final class Needle2ForwardPass {

  private static final int ENGRAM_SEED = 0x9E3779B9;
  private static final int ENGRAM_PRIME = 0x01000193;
  private static final float RMS_EPSILON = 1.0e-6f;

  private record EngramState(float[] key, float[] value) {}

  @FunctionalInterface
  private interface HiddenCellConsumer {
    void accept(float[] hiddenCell);
  }

  private final Needle2Weights weights;
  private final CactHeader config;
  private final int capacity;
  private final int modelWidth;
  private final int attentionWidth;
  private final int kvWidth;
  private final int[] tokenHistory;
  private final Needle2AttentionWindow attentionWindow;
  private final float[][][] keyCache;
  private final float[][][] valueCache;
  private final float[][][] engramValueHistory;
  private int checkpoint;

  public Needle2ForwardPass(Needle2Weights weights, int capacity) {
    this(weights, capacity, 9);
  }

  /** Creates a sequence using the artifact tokenizer's {@code </tools>} token as a sink marker. */
  public Needle2ForwardPass(Needle2Weights weights, int capacity, int toolDocumentEndToken) {
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
    this.attentionWindow = new Needle2AttentionWindow(config.kvWindow(), toolDocumentEndToken);
    this.keyCache = new float[config.layerCount()][capacity][kvWidth];
    this.valueCache = new float[config.layerCount()][capacity][kvWidth];
    this.engramValueHistory = new float[config.engramSites().size()][capacity][modelWidth];
  }

  public float[] forward(int token, int position) {
    return forward(token, position, null, true);
  }

  /** Encodes a sequence with Needle's serialized contrastive retrieval head. */
  public float[] encodeContrastive(int[] tokens) {
    Needle2ProbeHead head =
        weights
            .contrastiveHead()
            .orElseThrow(
                () -> new IllegalStateException("Needle artifact has no contrastive head"));
    return evaluateHead(tokens, head);
  }

  /** Scores a prompt-and-call sequence with Needle's serialized confidence head. */
  public float scoreConfidence(int[] tokens) {
    Needle2ProbeHead head =
        weights
            .confidenceHead()
            .orElseThrow(() -> new IllegalStateException("Needle artifact has no confidence head"));
    float logit = evaluateHead(tokens, head)[0];
    return Needle2Math.sigmoid(logit);
  }

  private float[] evaluateHead(int[] tokens, Needle2ProbeHead head) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("Needle auxiliary head requires at least one token");
    }
    if (tokens.length > config.maximumSequenceLength()) {
      throw new IllegalArgumentException(
          "Needle auxiliary-head sequence exceeds " + config.maximumSequenceLength());
    }
    Needle2ForwardPass sequence = new Needle2ForwardPass(weights, tokens.length);
    Needle2ProbeHead.Accumulator accumulator = head.newAccumulator();
    for (int position = 0; position < tokens.length; position++) {
      sequence.forward(tokens[position], position, accumulator::accept, false);
    }
    return accumulator.finish();
  }

  private float[] forward(
      int token, int position, HiddenCellConsumer hiddenCells, boolean projectLogits) {
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
    attentionWindow.accept(token, position);
    int[] visiblePositions = attentionWindow.visiblePositions(position);
    EngramState[] engrams = prepareEngrams(position);

    float[][] lanes = new float[config.mhcLanes()][modelWidth];
    float[] embedding = new float[modelWidth];
    weights.embedding.decodeRow(token, embedding);
    float embeddingScale = (float) Math.sqrt(modelWidth);
    for (int column = 0; column < modelWidth; column++) {
      embedding[column] *= embeddingScale;
    }
    if (hiddenCells != null) {
      hiddenCells.accept(embedding);
    }
    for (int lane = 0; lane < lanes.length; lane++) {
      for (int column = 0; column < modelWidth; column++) {
        lanes[lane][column] = embedding[column];
      }
    }

    for (int layer = 0; layer < config.layerCount(); layer++) {
      lanes = executeLayer(lanes, layer, position, visiblePositions, engrams);
      if (hiddenCells != null) {
        hiddenCells.accept(meanLanes(lanes));
      }
    }

    checkpoint++;
    if (!projectLogits) {
      return null;
    }

    float[] hidden = meanLanes(lanes);
    float[] normalized = new float[modelWidth];
    Needle2Math.zeroCenteredRmsNorm(hidden, weights.finalNorm, normalized);
    return multiply(weights.embedding, normalized);
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
    attentionWindow.rewind(targetCheckpoint);
  }

  public void reset() {
    rewind(0);
  }

  private float[][] executeLayer(
      float[][] lanes,
      int layerIndex,
      int position,
      int[] visiblePositions,
      EngramState[] engrams) {
    int laneCount = config.mhcLanes();
    Needle2Weights.Layer layer = weights.layers[layerIndex];
    float[] normalizedLanes = rmsUnit(flatten(lanes));

    float[][] routingProjections =
        multiplyTogether(
            normalizedLanes, layer.mhcPhiPre(), layer.mhcPhiPost(), layer.mhcPhiResidual());
    float[] preProjection = routingProjections[0];
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

    float[] blockOutput = executeBlock(layer, layerIndex, blockInput, position, visiblePositions);
    float[] residual = subtract(blockOutput, combined);

    float[] postProjection = routingProjections[1];
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

    float[] mixing = routingProjections[2];
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
      Needle2Weights.Layer layer,
      int layerIndex,
      float[] input,
      int position,
      int[] visiblePositions) {
    float[] attentionInput = new float[modelWidth];
    Needle2Math.zeroCenteredRmsNorm(input, layer.normIn(), attentionInput);

    float[][] attentionProjections =
        multiplyTogether(
            attentionInput,
            layer.query(),
            layer.key(),
            layer.value(),
            layer.attentionGateProjection());
    float[] query = attentionProjections[0];
    float[] key = attentionProjections[1];
    float[] value = attentionProjections[2];
    normalizeHeads(query, config.queryHeadCount(), layer.queryNorm());
    normalizeHeads(key, config.kvHeadCount(), layer.keyNorm());
    applyRope(query, config.queryHeadCount(), position);
    applyRope(key, config.kvHeadCount(), position);
    if (config.kvBits() != Byte.SIZE) {
      throw new IllegalStateException(
          "Needle Java inference does not yet implement " + config.kvBits() + "-bit KV cache");
    }
    System.arraycopy(key, 0, keyCache[layerIndex][position], 0, kvWidth);
    System.arraycopy(value, 0, valueCache[layerIndex][position], 0, kvWidth);

    float[] attended = attention(layerIndex, query, key, value, position, visiblePositions);
    float[] gate = attentionProjections[3];
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

  private float[] attention(
      int layer,
      float[] query,
      float[] currentKey,
      float[] currentValue,
      int position,
      int[] visiblePositions) {
    int queryHeads = config.queryHeadCount();
    int kvHeads = config.kvHeadCount();
    int repeats = queryHeads / kvHeads;
    int headWidth = config.headWidth();
    int attendedPositions = visiblePositions.length;
    float[] result = new float[attentionWidth];
    float[] scores = new float[attendedPositions];
    float scale = (float) Math.sqrt(headWidth);
    for (int queryHead = 0; queryHead < queryHeads; queryHead++) {
      int kvHead = queryHead / repeats;
      int queryOffset = queryHead * headWidth;
      int kvOffset = kvHead * headWidth;
      float maximum = Float.NEGATIVE_INFINITY;
      for (int index = 0; index < attendedPositions; index++) {
        int cachedPosition = visiblePositions[index];
        float[] cachedKey =
            cachedPosition == position ? currentKey : keyCache[layer][cachedPosition];
        float score = dot(query, queryOffset, cachedKey, kvOffset, headWidth) / scale;
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
        int cachedPosition = visiblePositions[index];
        float[] cachedValue =
            cachedPosition == position ? currentValue : valueCache[layer][cachedPosition];
        for (int column = 0; column < headWidth; column++) {
          result[queryOffset + column] += probability * cachedValue[kvOffset + column];
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
      float[][] projections =
          multiplyTogether(concatenated, weightsAtSite.key(), weightsAtSite.value());
      float[] key = projections[0];
      float[] rawValue = projections[1];
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

  private static float[][] multiplyTogether(float[] input, CactCqMatrix... matrices) {
    float[][] outputs = new float[matrices.length][];
    RotatedCodebookMatrix.PreparedActivation activation = null;
    for (int index = 0; index < matrices.length; index++) {
      CactCqMatrix matrix = matrices[index];
      if (!matrix.accepts(activation)) {
        activation = matrix.prepare(input);
      }
      float[] output = new float[matrix.rows()];
      matrix.multiply(activation, output);
      outputs[index] = output;
    }
    return outputs;
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

  private static float[] meanLanes(float[][] lanes) {
    float[] result = new float[lanes[0].length];
    float scale = 1.0f / lanes.length;
    for (float[] lane : lanes) {
      addScaled(result, lane, scale);
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
