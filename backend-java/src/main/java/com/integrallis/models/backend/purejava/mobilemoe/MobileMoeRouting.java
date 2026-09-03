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
package com.integrallis.models.backend.purejava.mobilemoe;

import java.util.Arrays;
import java.util.Objects;

/** MobileMoE sigmoid routing with selection-only expert bias and normalized route weights. */
final class MobileMoeRouting {

  private static final float NORMALIZATION_EPSILON = 1.0e-20f;

  private MobileMoeRouting() {}

  static void select(
      float[] logits,
      float[] expertBias,
      int topK,
      float routeScale,
      int[] selectedExperts,
      float[] routingWeights) {
    Objects.requireNonNull(logits, "logits");
    Objects.requireNonNull(expertBias, "expertBias");
    Objects.requireNonNull(selectedExperts, "selectedExperts");
    Objects.requireNonNull(routingWeights, "routingWeights");
    if (logits.length == 0 || expertBias.length != logits.length) {
      throw new IllegalArgumentException("expert bias must match the non-empty router logits");
    }
    if (topK <= 0 || topK > logits.length) {
      throw new IllegalArgumentException("topK must be in [1, " + logits.length + "]: " + topK);
    }
    if (selectedExperts.length != topK || routingWeights.length != topK) {
      throw new IllegalArgumentException("selected expert and routing buffers must match topK");
    }
    if (!(routeScale >= 0.0f) || !Float.isFinite(routeScale)) {
      throw new IllegalArgumentException("routeScale must be finite and non-negative");
    }

    float[] scores = new float[logits.length];
    float[] selectedScores = new float[topK];
    Arrays.fill(selectedExperts, -1);
    Arrays.fill(selectedScores, Float.NEGATIVE_INFINITY);
    for (int expert = 0; expert < logits.length; expert++) {
      float score = sigmoid(logits[expert]);
      scores[expert] = score;
      insertSelection(score + expertBias[expert], expert, selectedScores, selectedExperts);
    }

    float sum = 0.0f;
    for (int route = 0; route < topK; route++) {
      float score = scores[selectedExperts[route]];
      routingWeights[route] = score;
      sum += score;
    }
    float multiplier = routeScale / (sum + NORMALIZATION_EPSILON);
    for (int route = 0; route < topK; route++) {
      routingWeights[route] *= multiplier;
    }
  }

  static float sigmoid(float value) {
    if (value >= 0.0f) {
      double exponential = Math.exp(-value);
      return (float) (1.0 / (1.0 + exponential));
    }
    double exponential = Math.exp(value);
    return (float) (exponential / (1.0 + exponential));
  }

  private static void insertSelection(
      float candidateScore, int candidateExpert, float[] selectedScores, int[] selectedExperts) {
    for (int rank = 0; rank < selectedScores.length; rank++) {
      if (candidateScore > selectedScores[rank]
          || (candidateScore == selectedScores[rank]
              && (selectedExperts[rank] < 0 || candidateExpert < selectedExperts[rank]))) {
        for (int shifted = selectedScores.length - 1; shifted > rank; shifted--) {
          selectedScores[shifted] = selectedScores[shifted - 1];
          selectedExperts[shifted] = selectedExperts[shifted - 1];
        }
        selectedScores[rank] = candidateScore;
        selectedExperts[rank] = candidateExpert;
        return;
      }
    }
  }
}
