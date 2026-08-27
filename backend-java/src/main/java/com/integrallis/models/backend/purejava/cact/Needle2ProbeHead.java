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

/** Probe-attention pooling and dense projection used by Needle 2 auxiliary heads. */
final class Needle2ProbeHead {

  private final int modelWidth;
  private final int probeCount;
  private final int outputWidth;
  private final float[] probes;
  private final float[] projection;
  private final float[] bias;
  private final boolean normalizeOutput;

  Needle2ProbeHead(
      int modelWidth,
      int probeCount,
      int outputWidth,
      float[] probes,
      float[] projection,
      float[] bias,
      boolean normalizeOutput) {
    if (modelWidth <= 0 || probeCount <= 0 || outputWidth <= 0) {
      throw new IllegalArgumentException("probe-head dimensions must be positive");
    }
    this.modelWidth = modelWidth;
    this.probeCount = probeCount;
    this.outputWidth = outputWidth;
    this.probes = requireLength(probes, Math.multiplyExact(probeCount, modelWidth), "probes");
    this.projection =
        requireLength(
            projection,
            Math.multiplyExact(outputWidth, Math.multiplyExact(probeCount, modelWidth)),
            "projection");
    this.bias = requireLength(bias, outputWidth, "bias");
    this.normalizeOutput = normalizeOutput;
  }

  int outputWidth() {
    return outputWidth;
  }

  Accumulator newAccumulator() {
    return new Accumulator();
  }

  final class Accumulator {
    private final float[] maximum = new float[probeCount];
    private final double[] denominator = new double[probeCount];
    private final double[][] weighted = new double[probeCount][modelWidth];
    private int cells;

    private Accumulator() {
      Arrays.fill(maximum, Float.NEGATIVE_INFINITY);
    }

    void accept(float[] cell) {
      Objects.requireNonNull(cell, "cell");
      if (cell.length != modelWidth) {
        throw new IllegalArgumentException(
            "hidden cell width must be " + modelWidth + "; got " + cell.length);
      }
      float scale = (float) (1.0 / Math.sqrt(modelWidth));
      for (int probe = 0; probe < probeCount; probe++) {
        int probeOffset = probe * modelWidth;
        float score = 0.0f;
        for (int column = 0; column < modelWidth; column++) {
          score += cell[column] * probes[probeOffset + column];
        }
        score *= scale;
        if (score <= maximum[probe]) {
          double weight = Math.exp(score - maximum[probe]);
          denominator[probe] += weight;
          for (int column = 0; column < modelWidth; column++) {
            weighted[probe][column] += weight * cell[column];
          }
        } else {
          double rescale = Math.exp(maximum[probe] - score);
          denominator[probe] = denominator[probe] * rescale + 1.0d;
          for (int column = 0; column < modelWidth; column++) {
            weighted[probe][column] = weighted[probe][column] * rescale + cell[column];
          }
          maximum[probe] = score;
        }
      }
      cells++;
    }

    float[] finish() {
      if (cells == 0) {
        throw new IllegalStateException("probe head requires at least one hidden cell");
      }
      float[] pooled = new float[Math.multiplyExact(probeCount, modelWidth)];
      for (int probe = 0; probe < probeCount; probe++) {
        int offset = probe * modelWidth;
        for (int column = 0; column < modelWidth; column++) {
          pooled[offset + column] = (float) (weighted[probe][column] / denominator[probe]);
        }
      }
      float[] result = new float[outputWidth];
      for (int output = 0; output < outputWidth; output++) {
        float sum = bias[output];
        int projectionOffset = output * pooled.length;
        for (int input = 0; input < pooled.length; input++) {
          sum += projection[projectionOffset + input] * pooled[input];
        }
        result[output] = sum;
      }
      if (normalizeOutput) {
        normalize(result);
      }
      return result;
    }
  }

  private static float[] requireLength(float[] values, int expected, String name) {
    Objects.requireNonNull(values, name);
    if (values.length != expected) {
      throw new IllegalArgumentException(
          name + " length must be " + expected + "; got " + values.length);
    }
    float[] copy = values.clone();
    for (float value : copy) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException(name + " must contain only finite values");
      }
    }
    return copy;
  }

  private static void normalize(float[] values) {
    double squared = 0.0d;
    for (float value : values) {
      squared += (double) value * value;
    }
    double denominator = Math.sqrt(squared + 1.0e-12d);
    for (int index = 0; index < values.length; index++) {
      values[index] /= (float) denominator;
    }
  }
}
