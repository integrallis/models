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
package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.backend.purejava.ops.TensorOps;

/** Per-position rotary factors shared by every attention head and transformer layer. */
final class RopeTable {

  private final int headDim;
  private final float theta;
  private final float frequencyScale;
  private final float[] cosine;
  private final float[] sine;
  private float[] batchCosine = new float[0];
  private float[] batchSine = new float[0];
  private int preparedPosition = -1;
  private int preparedBatchStart = -1;
  private int preparedBatchSize;
  private int preparationCount;

  RopeTable(int headDim, float theta, float frequencyScale) {
    if (headDim <= 0 || (headDim & 1) != 0) {
      throw new IllegalArgumentException("headDim must be positive and even: " + headDim);
    }
    this.headDim = headDim;
    this.theta = theta;
    this.frequencyScale = frequencyScale;
    this.cosine = new float[headDim / 2];
    this.sine = new float[headDim / 2];
  }

  void prepare(int position) {
    if (position == preparedPosition) {
      return;
    }
    float scaledPosition = position * frequencyScale;
    for (int pair = 0; pair < cosine.length; pair++) {
      float frequency = (float) (1.0 / Math.pow(theta, (double) (pair * 2) / headDim));
      float angle = scaledPosition * frequency;
      cosine[pair] = (float) Math.cos(angle);
      sine[pair] = (float) Math.sin(angle);
    }
    preparedPosition = position;
    preparationCount++;
  }

  void apply(float[] vector, int offset, boolean neox) {
    if (preparedPosition < 0) {
      throw new IllegalStateException("rotary factors have not been prepared");
    }
    if (neox) {
      TensorOps.ropeNeox(vector, offset, cosine, sine);
    } else {
      TensorOps.rope(vector, offset, cosine, sine);
    }
  }

  void prepareBatch(int startPosition, int batchSize) {
    if (startPosition < 0) {
      throw new IllegalArgumentException("startPosition must be >= 0");
    }
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1");
    }
    if (startPosition == preparedBatchStart && batchSize == preparedBatchSize) {
      return;
    }

    int pairCount = headDim / 2;
    int requiredFactors = Math.multiplyExact(batchSize, pairCount);
    if (batchCosine.length < requiredFactors) {
      batchCosine = new float[requiredFactors];
      batchSine = new float[requiredFactors];
    }
    for (int batch = 0; batch < batchSize; batch++) {
      float scaledPosition = (startPosition + batch) * frequencyScale;
      int factorOffset = batch * pairCount;
      for (int pair = 0; pair < pairCount; pair++) {
        float frequency = (float) (1.0 / Math.pow(theta, (double) (pair * 2) / headDim));
        float angle = scaledPosition * frequency;
        batchCosine[factorOffset + pair] = (float) Math.cos(angle);
        batchSine[factorOffset + pair] = (float) Math.sin(angle);
      }
    }
    preparedBatchStart = startPosition;
    preparedBatchSize = batchSize;
    preparationCount += batchSize;
  }

  void applyBatch(float[] vector, int offset, int batchIndex, boolean neox) {
    if (batchIndex < 0 || batchIndex >= preparedBatchSize) {
      throw new IllegalArgumentException("batchIndex out of range: " + batchIndex);
    }
    int pairCount = headDim / 2;
    int factorOffset = batchIndex * pairCount;
    if (neox) {
      TensorOps.ropeNeox(vector, offset, batchCosine, batchSine, factorOffset, pairCount);
    } else {
      TensorOps.rope(vector, offset, batchCosine, batchSine, factorOffset, pairCount);
    }
  }

  int preparationCount() {
    return preparationCount;
  }
}
