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
package com.integrallis.models.backend.purejava.gptoss;

import java.util.Arrays;
import java.util.Objects;

/** Allocation-free GPT-OSS routing and expert activation operations. */
final class GptOssMath {

  private GptOssMath() {}

  /** Selects stable top-k logits and softmax-normalizes only the selected experts. */
  static void selectExperts(float[] routerLogits, int[] selectedExperts, float[] routingWeights) {
    Objects.requireNonNull(routerLogits, "routerLogits");
    Objects.requireNonNull(selectedExperts, "selectedExperts");
    Objects.requireNonNull(routingWeights, "routingWeights");
    if (routerLogits.length == 0) {
      throw new IllegalArgumentException("router logits must not be empty");
    }
    if (selectedExperts.length == 0 || selectedExperts.length > routerLogits.length) {
      throw new IllegalArgumentException("top-k length must be between 1 and the expert count");
    }
    if (routingWeights.length != selectedExperts.length) {
      throw new IllegalArgumentException("router output buffers must have the same top-k length");
    }
    if (routingWeights == routerLogits) {
      throw new IllegalArgumentException("routingWeights must not alias routerLogits");
    }

    Arrays.fill(selectedExperts, -1);
    Arrays.fill(routingWeights, Float.NEGATIVE_INFINITY);
    for (int expert = 0; expert < routerLogits.length; expert++) {
      float logit = routerLogits[expert];
      if (!Float.isFinite(logit)) {
        throw new IllegalArgumentException("routerLogits[" + expert + "] must be finite: " + logit);
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
      routingWeights[rank] /= sum;
    }
  }

  /** Applies GPT-OSS's interleaved, clamped SwiGLU-OAI activation. */
  static void swigluOai(float[] gateUp, float[] output, float alpha, float limit) {
    Objects.requireNonNull(gateUp, "gateUp");
    Objects.requireNonNull(output, "output");
    if (gateUp.length != Math.multiplyExact(output.length, 2)) {
      throw new IllegalArgumentException("gateUp length must be twice the output length");
    }
    if (output.length == 0) {
      throw new IllegalArgumentException("activation output must not be empty");
    }
    if (!(alpha > 0.0f) || !Float.isFinite(alpha)) {
      throw new IllegalArgumentException("alpha must be finite and > 0: " + alpha);
    }
    if (!(limit > 0.0f) || !Float.isFinite(limit)) {
      throw new IllegalArgumentException("limit must be finite and > 0: " + limit);
    }

    for (int index = 0; index < output.length; index++) {
      float rawGate = gateUp[2 * index];
      float rawUp = gateUp[2 * index + 1];
      if (!Float.isFinite(rawGate) || !Float.isFinite(rawUp)) {
        throw new IllegalArgumentException(
            "gateUp pair " + index + " must be finite: " + rawGate + ", " + rawUp);
      }
      float gate = Math.min(rawGate, limit);
      float up = Math.max(-limit, Math.min(rawUp, limit));
      float sigmoid = (float) (1.0 / (1.0 + Math.exp(-alpha * gate)));
      output[index] = gate * sigmoid * (up + 1.0f);
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
