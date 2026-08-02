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
package com.integrallis.models.backend.purejava.gemma4;

import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Objects;

/** Allocation-free Gemma 4 routing and output transformations. */
final class Gemma4Math {

  private Gemma4Math() {}

  /** Applies RMS normalization without a learned weight. */
  static void normalizeWithoutWeight(float[] output, float[] input, float epsilon) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    if (input.length == 0 || output.length != input.length) {
      throw new IllegalArgumentException("input and output lengths must match and be non-empty");
    }
    normalizeWithoutWeight(output, 0, input, 0, input.length, epsilon);
  }

  /** Applies unweighted RMS normalization over matching source and destination slices. */
  static void normalizeWithoutWeight(
      float[] output, int outputOffset, float[] input, int inputOffset, int length, float epsilon) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    if (length <= 0) {
      throw new IllegalArgumentException("length must be > 0: " + length);
    }
    Objects.checkFromIndexSize(outputOffset, length, output.length);
    Objects.checkFromIndexSize(inputOffset, length, input.length);
    if (!(epsilon > 0.0f) || !Float.isFinite(epsilon)) {
      throw new IllegalArgumentException("epsilon must be finite and > 0: " + epsilon);
    }
    float sumSquares = VectorUtil.dotProduct(input, inputOffset, input, inputOffset, length);
    float inverse = (float) (1.0 / Math.sqrt(sumSquares / length + epsilon));
    for (int index = 0; index < length; index++) {
      output[outputOffset + index] = input[inputOffset + index] * inverse;
    }
  }

  /** Applies the router's unweighted RMS normalization and learned dimension scale. */
  static void normalizeRouterInput(
      float[] output, float[] hidden, float[] routerScale, float epsilon) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(hidden, "hidden");
    Objects.requireNonNull(routerScale, "routerScale");
    if (hidden.length == 0
        || output.length != hidden.length
        || routerScale.length != hidden.length) {
      throw new IllegalArgumentException("router input, output, and scale lengths must match");
    }
    normalizeRouterInput(output, 0, hidden, 0, routerScale, hidden.length, epsilon);
  }

  /** Applies router normalization over matching source and destination slices. */
  static void normalizeRouterInput(
      float[] output,
      int outputOffset,
      float[] hidden,
      int hiddenOffset,
      float[] routerScale,
      int length,
      float epsilon) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(hidden, "hidden");
    Objects.requireNonNull(routerScale, "routerScale");
    if (length <= 0 || routerScale.length != length) {
      throw new IllegalArgumentException("router scale length must match the normalized slice");
    }
    Objects.checkFromIndexSize(outputOffset, length, output.length);
    Objects.checkFromIndexSize(hiddenOffset, length, hidden.length);
    if (!(epsilon > 0.0f) || !Float.isFinite(epsilon)) {
      throw new IllegalArgumentException("epsilon must be finite and > 0: " + epsilon);
    }

    float sumSquares = VectorUtil.dotProduct(hidden, hiddenOffset, hidden, hiddenOffset, length);
    float rmsInverse = (float) (1.0 / Math.sqrt(sumSquares / length + epsilon));
    float dimensionScale = (float) (1.0 / Math.sqrt(length));
    float normalization = rmsInverse * dimensionScale;
    for (int index = 0; index < length; index++) {
      float scale = routerScale[index];
      if (!Float.isFinite(scale)) {
        throw new IllegalArgumentException("routerScale[" + index + "] must be finite: " + scale);
      }
      output[outputOffset + index] = hidden[hiddenOffset + index] * normalization * scale;
    }
  }

  /** Selects a stable top-k, softmaxes selected logits, then applies per-expert output scales. */
  static void selectExperts(
      float[] routerLogits, float[] expertScales, int[] selectedExperts, float[] routingWeights) {
    Objects.requireNonNull(routerLogits, "routerLogits");
    Objects.requireNonNull(expertScales, "expertScales");
    Objects.requireNonNull(selectedExperts, "selectedExperts");
    Objects.requireNonNull(routingWeights, "routingWeights");
    if (routerLogits.length == 0 || expertScales.length != routerLogits.length) {
      throw new IllegalArgumentException("router logits and expert scale lengths must match");
    }
    if (selectedExperts.length == 0 || selectedExperts.length > routerLogits.length) {
      throw new IllegalArgumentException("top-k length must be between 1 and the expert count");
    }
    if (routingWeights.length != selectedExperts.length) {
      throw new IllegalArgumentException("router output buffers must have the same top-k length");
    }
    if (routingWeights == routerLogits || routingWeights == expertScales) {
      throw new IllegalArgumentException("routingWeights must not alias router inputs");
    }

    Arrays.fill(selectedExperts, -1);
    Arrays.fill(routingWeights, Float.NEGATIVE_INFINITY);
    for (int expert = 0; expert < routerLogits.length; expert++) {
      float logit = routerLogits[expert];
      float expertScale = expertScales[expert];
      if (!Float.isFinite(logit)) {
        throw new IllegalArgumentException("routerLogits[" + expert + "] must be finite: " + logit);
      }
      if (!Float.isFinite(expertScale)) {
        throw new IllegalArgumentException(
            "expertScales[" + expert + "] must be finite: " + expertScale);
      }
      insertCandidate(logit, expert, selectedExperts, routingWeights);
    }

    float maximum = routingWeights[0];
    float sum = 0.0f;
    for (int rank = 0; rank < routingWeights.length; rank++) {
      float exponent = (float) Math.exp(routingWeights[rank] - maximum);
      routingWeights[rank] = exponent;
      sum += exponent;
    }
    for (int rank = 0; rank < routingWeights.length; rank++) {
      routingWeights[rank] = routingWeights[rank] / sum * expertScales[selectedExperts[rank]];
    }
  }

  /** Applies Gemma 4's bounded final-logit transform in place. */
  static void softcap(float[] logits, float softcap) {
    Objects.requireNonNull(logits, "logits");
    if (!(softcap > 0.0f) || !Float.isFinite(softcap)) {
      throw new IllegalArgumentException("softcap must be finite and > 0: " + softcap);
    }
    for (int index = 0; index < logits.length; index++) {
      logits[index] = softcap * (float) Math.tanh(logits[index] / softcap);
    }
  }

  private static void insertCandidate(
      float logit, int expert, int[] selectedExperts, float[] selectedLogits) {
    for (int rank = 0; rank < selectedLogits.length; rank++) {
      int selectedExpert = selectedExperts[rank];
      if (logit > selectedLogits[rank]
          || (logit == selectedLogits[rank] && expert < selectedExpert)) {
        for (int shifted = selectedLogits.length - 1; shifted > rank; shifted--) {
          selectedLogits[shifted] = selectedLogits[shifted - 1];
          selectedExperts[shifted] = selectedExperts[shifted - 1];
        }
        selectedLogits[rank] = logit;
        selectedExperts[rank] = expert;
        return;
      }
    }
  }
}
