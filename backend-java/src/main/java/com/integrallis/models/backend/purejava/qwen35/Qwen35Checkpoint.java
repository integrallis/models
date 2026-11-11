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
package com.integrallis.models.backend.purejava.qwen35;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

/** Complete immutable hybrid state. Arrays are owned here and only read by attention/fork code. */
final class Qwen35Checkpoint {
  private static final int MAGIC = 0x51435031; // QCP1: raw FP32, complete captured boundary.
  final int[] tokens;
  private final int[] segmentEnds;
  private final float[][][] keys;
  private final float[][][] values;
  final float[][] convolution;
  final float[][] recurrent;

  private Qwen35Checkpoint(
      int[] tokens,
      int[] segmentEnds,
      float[][][] keys,
      float[][][] values,
      float[][] convolution,
      float[][] recurrent) {
    this.tokens = tokens;
    this.segmentEnds = segmentEnds;
    this.keys = keys;
    this.values = values;
    this.convolution = convolution;
    this.recurrent = recurrent;
  }

  static Qwen35Checkpoint capture(
      Qwen35Config config,
      int[] history,
      int count,
      Qwen35Checkpoint prefix,
      float[][] suffixKeys,
      float[][] suffixValues,
      float[][] convolution,
      float[][] recurrent) {
    int retained = prefix == null ? 0 : prefix.tokens.length;
    if (count == retained) return prefix;
    int previousSegments = prefix == null ? 0 : prefix.segmentEnds.length;
    int[] ends =
        prefix == null ? new int[1] : Arrays.copyOf(prefix.segmentEnds, previousSegments + 1);
    ends[previousSegments] = count;
    float[][][] allKeys =
        prefix == null ? new float[1][][] : Arrays.copyOf(prefix.keys, previousSegments + 1);
    float[][][] allValues =
        prefix == null ? new float[1][][] : Arrays.copyOf(prefix.values, previousSegments + 1);
    int dimension = config.attentionKeyDim();
    float[][] keys = new float[config.numLayers()][];
    float[][] values = new float[config.numLayers()][];
    float[][] conv = new float[config.numLayers()][];
    float[][] state = new float[config.numLayers()][];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesFullAttention(layer)) {
        keys[layer] =
            Arrays.copyOf(suffixKeys[layer], Math.multiplyExact(count - retained, dimension));
        values[layer] = Arrays.copyOf(suffixValues[layer], keys[layer].length);
      } else {
        conv[layer] = convolution[layer].clone();
        state[layer] = recurrent[layer].clone();
      }
    }
    allKeys[previousSegments] = keys;
    allValues[previousSegments] = values;
    return new Qwen35Checkpoint(
        Arrays.copyOf(history, count), ends, allKeys, allValues, conv, state);
  }

  int segmentAt(int position) {
    int found = Arrays.binarySearch(segmentEnds, position + 1);
    return found >= 0 ? found : -found - 1;
  }

  int segmentStart(int segment) {
    return segment == 0 ? 0 : segmentEnds[segment - 1];
  }

  float[] keys(int segment, int layer) {
    return keys[segment][layer];
  }

  float[] values(int segment, int layer) {
    return values[segment][layer];
  }

  long attentionBytes() {
    long bytes = 0;
    for (float[][] segment : keys)
      for (float[] layer : segment) if (layer != null) bytes += 8L * layer.length;
    return bytes;
  }

  boolean sharesAttention(Qwen35Checkpoint other) {
    return other != null && keys[0] == other.keys[0];
  }

  void copyAttention(int layer, float[] destinationKeys, float[] destinationValues) {
    int offset = 0;
    for (int segment = 0; segment < keys.length; segment++) {
      System.arraycopy(
          keys[segment][layer], 0, destinationKeys, offset, keys[segment][layer].length);
      System.arraycopy(
          values[segment][layer], 0, destinationValues, offset, values[segment][layer].length);
      offset += keys[segment][layer].length;
    }
  }

  static int convolutionElements(Qwen35Config config) {
    return Math.multiplyExact(config.gdnConvDim(), config.gdnConvKernel() - 1);
  }

  static int recurrentElements(Qwen35Config config) {
    return Math.multiplyExact(
        Math.multiplyExact(config.gdnValueHeads(), config.gdnHeadDim()), config.gdnHeadDim());
  }

  static long serializedBytes(Qwen35Config config, int count) {
    if (count <= 0 || count > config.contextLength())
      throw new IllegalArgumentException("invalid checkpoint length");
    int attentionElements = Math.multiplyExact(count, config.attentionKeyDim());
    long elements = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      elements =
          Math.addExact(
              elements,
              config.usesFullAttention(layer)
                  ? 2L * attentionElements
                  : (long) convolutionElements(config) + recurrentElements(config));
    }
    return Math.addExact(24L + config.numLayers() + 4L * count, Math.multiplyExact(4L, elements));
  }

  void write(DataOutput output, Qwen35Config config, int count) throws IOException {
    requireBoundary(count);
    output.writeInt(MAGIC);
    output.writeInt(count);
    output.writeInt(config.numLayers());
    output.writeInt(config.attentionKeyDim());
    output.writeInt(convolutionElements(config));
    output.writeInt(recurrentElements(config));
    for (int layer = 0; layer < config.numLayers(); layer++)
      output.writeBoolean(config.usesFullAttention(layer));
    for (int token : tokens) output.writeInt(token);
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesFullAttention(layer)) {
        for (float[][] segment : keys) writeFloats(output, segment[layer]);
        for (float[][] segment : values) writeFloats(output, segment[layer]);
      } else {
        writeFloats(output, convolution[layer]);
        writeFloats(output, recurrent[layer]);
      }
    }
  }

  static Qwen35Checkpoint read(DataInput input, Qwen35Config config, int count, long maxBytes)
      throws IOException {
    try {
      if (serializedBytes(config, count) > maxBytes)
        throw new IOException("checkpoint exceeds import budget");
    } catch (IllegalArgumentException | ArithmeticException invalid) {
      throw new IOException("invalid checkpoint dimensions", invalid);
    }
    if (input.readInt() != MAGIC
        || input.readInt() != count
        || input.readInt() != config.numLayers()
        || input.readInt() != config.attentionKeyDim()
        || input.readInt() != convolutionElements(config)
        || input.readInt() != recurrentElements(config))
      throw new IOException("incompatible hybrid checkpoint header");
    for (int layer = 0; layer < config.numLayers(); layer++) {
      int kind = input.readUnsignedByte();
      if (kind != (config.usesFullAttention(layer) ? 1 : 0))
        throw new IOException("incompatible hybrid layer layout");
    }
    int[] tokens = new int[count];
    for (int index = 0; index < count; index++) {
      tokens[index] = input.readInt();
      if (tokens[index] < 0 || tokens[index] >= config.vocabSize())
        throw new IOException("invalid checkpoint token");
    }
    float[][] keys = new float[config.numLayers()][];
    float[][] values = new float[config.numLayers()][];
    float[][] conv = new float[config.numLayers()][];
    float[][] state = new float[config.numLayers()][];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesFullAttention(layer)) {
        keys[layer] = readFloats(input, Math.multiplyExact(count, config.attentionKeyDim()));
        values[layer] = readFloats(input, keys[layer].length);
      } else {
        conv[layer] = readFloats(input, convolutionElements(config));
        state[layer] = readFloats(input, recurrentElements(config));
      }
    }
    return new Qwen35Checkpoint(
        tokens, new int[] {count}, new float[][][] {keys}, new float[][][] {values}, conv, state);
  }

  void requireBoundary(int count) {
    if (count != tokens.length)
      throw new IllegalArgumentException(
          "hybrid checkpoint retains only boundary "
              + tokens.length
              + "; capture the requested boundary or recompute");
  }

  private static void writeFloats(DataOutput output, float[] values) throws IOException {
    for (float value : values) output.writeInt(Float.floatToRawIntBits(value));
  }

  private static float[] readFloats(DataInput input, int count) throws IOException {
    float[] values = new float[count];
    for (int index = 0; index < count; index++)
      values[index] = Float.intBitsToFloat(input.readInt());
    return values;
  }
}
