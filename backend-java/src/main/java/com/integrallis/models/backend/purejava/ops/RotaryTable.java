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
package com.integrallis.models.backend.purejava.ops;

import java.util.Arrays;

/** Per-position rotary factors shared by attention heads and transformer layers. */
public final class RotaryTable {

  private final int rotaryDim;
  private final float frequencyScale;
  private final float attentionFactor;
  private final float[] frequencyFactors;
  private final float[] frequencies;
  private final float[] cosine;
  private final float[] sine;
  private float[] batchCosine = new float[0];
  private float[] batchSine = new float[0];
  private int[] preparedPositions = new int[0];
  private int preparedPosition = -1;
  private int preparedBatchStart = -1;
  private int preparedBatchSize;
  private int preparationCount;

  /** Creates a table using ordinary RoPE frequencies. */
  public RotaryTable(int rotaryDim, float theta, float frequencyScale) {
    this(rotaryDim, theta, frequencyScale, null);
  }

  /**
   * Creates a table with optional per-pair GGUF frequency divisors.
   *
   * <p>A factor divides that pair's angle. Large factors can therefore encode proportional RoPE
   * layouts whose remaining dimensions are effectively unrotated.
   */
  public RotaryTable(int rotaryDim, float theta, float frequencyScale, float[] frequencyFactors) {
    this(rotaryDim, theta, frequencyScale, frequencyFactors, 1.0f, null);
  }

  private RotaryTable(
      int rotaryDim,
      float theta,
      float frequencyScale,
      float[] frequencyFactors,
      float attentionFactor,
      float[] preparedFrequencies) {
    if (rotaryDim <= 0 || (rotaryDim & 1) != 0) {
      throw new IllegalArgumentException("rotaryDim must be positive and even: " + rotaryDim);
    }
    finitePositive("theta", theta);
    finitePositive("frequencyScale", frequencyScale);
    finitePositive("attentionFactor", attentionFactor);
    int pairCount = rotaryDim / 2;
    if (frequencyFactors != null) {
      if (frequencyFactors.length != pairCount) {
        throw new IllegalArgumentException(
            "frequencyFactors must contain " + pairCount + " entries: " + frequencyFactors.length);
      }
      for (int pair = 0; pair < pairCount; pair++) {
        finitePositive("frequencyFactors[" + pair + "]", frequencyFactors[pair]);
      }
      this.frequencyFactors = frequencyFactors.clone();
    } else {
      this.frequencyFactors = null;
    }
    this.rotaryDim = rotaryDim;
    this.frequencyScale = frequencyScale;
    this.attentionFactor = attentionFactor;
    if (preparedFrequencies == null) {
      this.frequencies = new float[pairCount];
      for (int pair = 0; pair < pairCount; pair++) {
        frequencies[pair] = (float) (1.0 / Math.pow(theta, (double) (pair * 2) / rotaryDim));
      }
    } else {
      if (preparedFrequencies.length != pairCount) {
        throw new IllegalArgumentException(
            "preparedFrequencies must contain " + pairCount + " entries");
      }
      this.frequencies = preparedFrequencies.clone();
    }
    this.cosine = new float[pairCount];
    this.sine = new float[pairCount];
  }

  /**
   * Creates the non-trivial YaRN frequency blend and attention magnitude used by GPT-OSS and other
   * long-context models.
   */
  public static RotaryTable yarn(
      int rotaryDim,
      float theta,
      float factor,
      float betaFast,
      float betaSlow,
      int originalContext,
      boolean truncateCorrectionRange) {
    if (rotaryDim <= 0 || (rotaryDim & 1) != 0) {
      throw new IllegalArgumentException("rotaryDim must be positive and even: " + rotaryDim);
    }
    finitePositive("theta", theta);
    finitePositive("factor", factor);
    finitePositive("betaFast", betaFast);
    finitePositive("betaSlow", betaSlow);
    if (originalContext <= 0) {
      throw new IllegalArgumentException("originalContext must be positive: " + originalContext);
    }

    double denominator = 2.0 * Math.log(theta);
    double low = rotaryDim * Math.log(originalContext / (betaFast * 2.0 * Math.PI)) / denominator;
    double high = rotaryDim * Math.log(originalContext / (betaSlow * 2.0 * Math.PI)) / denominator;
    if (truncateCorrectionRange) {
      low = Math.floor(low);
      high = Math.ceil(high);
    }
    low = Math.max(low, 0.0);
    high = Math.min(high, rotaryDim - 1.0);
    if (low == high) {
      high += 0.001;
    }

    float[] frequencies = new float[rotaryDim / 2];
    for (int pair = 0; pair < frequencies.length; pair++) {
      double inverseFrequency = 1.0 / Math.pow(theta, (double) (pair * 2) / rotaryDim);
      double ramp = Math.max(0.0, Math.min(1.0, (pair - low) / (high - low)));
      frequencies[pair] =
          (float) ((inverseFrequency / factor) * ramp + inverseFrequency * (1.0 - ramp));
    }
    float attentionFactor = factor <= 1.0f ? 1.0f : 0.1f * (float) Math.log(factor) + 1.0f;
    return new RotaryTable(rotaryDim, theta, 1.0f, null, attentionFactor, frequencies);
  }

  /** Prepares factors for one absolute sequence position. */
  public void prepare(int position) {
    if (position < 0) {
      throw new IllegalArgumentException("position must be >= 0: " + position);
    }
    if (position == preparedPosition) {
      return;
    }
    fillFactors(position, cosine, sine, 0);
    preparedPosition = position;
    preparationCount++;
  }

  /** Applies the prepared factors to one head at the supplied vector offset. */
  public void apply(float[] vector, int offset, boolean neox) {
    if (preparedPosition < 0) {
      throw new IllegalStateException("rotary factors have not been prepared");
    }
    if (neox) {
      TensorOps.ropeNeox(vector, offset, cosine, sine);
    } else {
      TensorOps.rope(vector, offset, cosine, sine);
    }
  }

  /** Prepares factors for consecutive positions in a prefill batch. */
  public void prepareBatch(int startPosition, int batchSize) {
    if (startPosition < 0) {
      throw new IllegalArgumentException("startPosition must be >= 0");
    }
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1");
    }
    if (startPosition == preparedBatchStart && batchSize == preparedBatchSize) {
      return;
    }

    ensureBatchCapacity(batchSize);
    for (int batch = 0; batch < batchSize; batch++) {
      fillFactors(startPosition + batch, batchCosine, batchSine, batch * cosine.length);
    }
    preparedBatchStart = startPosition;
    preparedBatchSize = batchSize;
    preparedPositions = new int[0];
    preparationCount += batchSize;
  }

  /** Prepares factors for caller-supplied absolute positions. */
  public void preparePositions(int[] positions, int count) {
    if (positions == null) {
      throw new NullPointerException("positions");
    }
    if (count < 1 || count > positions.length) {
      throw new IllegalArgumentException("count must be between 1 and positions.length: " + count);
    }
    for (int index = 0; index < count; index++) {
      if (positions[index] < 0) {
        throw new IllegalArgumentException("position must be >= 0: " + positions[index]);
      }
    }
    if (preparedPositions.length == count
        && Arrays.equals(preparedPositions, 0, count, positions, 0, count)) {
      return;
    }

    ensureBatchCapacity(count);
    for (int batch = 0; batch < count; batch++) {
      fillFactors(positions[batch], batchCosine, batchSine, batch * cosine.length);
    }
    preparedPositions = Arrays.copyOf(positions, count);
    preparedBatchStart = -1;
    preparedBatchSize = count;
    preparationCount += count;
  }

  /** Applies one prepared batch row to a head at the supplied vector offset. */
  public void applyBatch(float[] vector, int offset, int batchIndex, boolean neox) {
    if (batchIndex < 0 || batchIndex >= preparedBatchSize) {
      throw new IllegalArgumentException("batchIndex out of range: " + batchIndex);
    }
    int factorOffset = batchIndex * cosine.length;
    if (neox) {
      TensorOps.ropeNeox(vector, offset, batchCosine, batchSine, factorOffset, cosine.length);
    } else {
      TensorOps.rope(vector, offset, batchCosine, batchSine, factorOffset, cosine.length);
    }
  }

  int preparationCount() {
    return preparationCount;
  }

  private void ensureBatchCapacity(int batchSize) {
    int requiredFactors = Math.multiplyExact(batchSize, cosine.length);
    if (batchCosine.length < requiredFactors) {
      batchCosine = new float[requiredFactors];
      batchSine = new float[requiredFactors];
    }
  }

  private void fillFactors(
      int position, float[] cosineDestination, float[] sineDestination, int destinationOffset) {
    float scaledPosition = position * frequencyScale;
    for (int pair = 0; pair < cosine.length; pair++) {
      float divisor = frequencyFactors == null ? 1.0f : frequencyFactors[pair];
      float angle = scaledPosition * frequencies[pair] / divisor;
      cosineDestination[destinationOffset + pair] = attentionFactor * (float) Math.cos(angle);
      sineDestination[destinationOffset + pair] = attentionFactor * (float) Math.sin(angle);
    }
  }

  private static void finitePositive(String name, float value) {
    if (!(value > 0.0f) || !Float.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite and > 0: " + value);
    }
  }
}
