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

import com.integrallis.vectors.core.GgufQ8_0Batch;
import java.util.Arrays;
import java.util.Objects;

/** Session-owned workspace for a single-token GPT-OSS MXFP4 routed expert layer. */
final class GptOssMxfp4Moe {

  private final GptOssMxfp4ExpertWeights weights;
  private final int hiddenSize;
  private final int intermediateSize;
  private final float alpha;
  private final float limit;
  private final boolean fusedShape;
  private final float[] gateUp;
  private final float[] splitGate;
  private final float[] splitUp;
  private final float[] activation;
  private final float[] expertOutput;
  private final GgufQ8_0Batch hiddenQ8;
  private final GgufQ8_0Batch activationQ8;

  GptOssMxfp4Moe(GptOssMxfp4ExpertWeights weights, float alpha, float limit) {
    this.weights = Objects.requireNonNull(weights, "weights");
    if (!(alpha > 0.0f) || !Float.isFinite(alpha)) {
      throw new IllegalArgumentException("alpha must be finite and > 0: " + alpha);
    }
    if (!(limit > 0.0f) || !Float.isFinite(limit)) {
      throw new IllegalArgumentException("limit must be finite and > 0: " + limit);
    }
    this.alpha = alpha;
    this.limit = limit;

    GptOssMxfp4ExpertWeights.Expert first = weights.expert(0);
    hiddenSize = first.hiddenSize();
    if (first.isFused() && first.gateUp().rows() % 2 != 0) {
      throw new IllegalArgumentException("gate/up projection must contain an even number of rows");
    }
    intermediateSize = first.intermediateSize();
    // Before any validation: validateGeometry compares every expert against this, and reading it
    // from
    // the not-yet-assigned scratch was a null dereference that took out the whole fused path.
    fusedShape = first.isFused();
    validateGeometry(first, 0);
    for (int expert = 1; expert < weights.expertCount(); expert++) {
      validateGeometry(weights.expert(expert), expert);
    }

    gateUp = new float[Math.multiplyExact(2, intermediateSize)];
    // Only the split shape needs these, and only one shape is ever present, so the unused pair is
    // empty rather than a second full-width allocation per layer.
    splitGate = fusedShape ? new float[0] : new float[intermediateSize];
    splitUp = fusedShape ? new float[0] : new float[intermediateSize];
    activation = new float[intermediateSize];
    expertOutput = new float[hiddenSize];
    hiddenQ8 = GgufQ8_0Batch.allocate(1, hiddenSize);
    activationQ8 = GgufQ8_0Batch.allocate(1, intermediateSize);
  }

  void forwardExact(float[] hidden, int[] selectedExperts, float[] routingWeights, float[] output) {
    forward(hidden, selectedExperts, routingWeights, output, false);
  }

  void forwardQ8(float[] hidden, int[] selectedExperts, float[] routingWeights, float[] output) {
    forward(hidden, selectedExperts, routingWeights, output, true);
  }

  private void forward(
      float[] hidden,
      int[] selectedExperts,
      float[] routingWeights,
      float[] output,
      boolean q8Activations) {
    validateForwardArguments(hidden, selectedExperts, routingWeights, output);
    Arrays.fill(output, 0.0f);
    if (q8Activations) {
      hiddenQ8.quantize(hidden, 1);
    }

    for (int route = 0; route < selectedExperts.length; route++) {
      GptOssMxfp4ExpertWeights.Expert expert = weights.expert(selectedExperts[route]);
      if (expert.isFused()) {
        if (q8Activations) {
          expert.gateUp().multiplyQ8(hiddenQ8, gateUp);
        } else {
          expert.gateUp().multiply(hidden, gateUp);
        }
        addBias(gateUp, expert.gateUpBias());
        GptOssMath.swigluOai(gateUp, activation, alpha, limit);
      } else {
        // The split shape has no Q8 activation fast path: ggufMatmul quantizes the activation
        // itself
        // for the quantized types, so asking for it here would quantize twice.
        expert.gate().multiply(hidden, splitGate);
        expert.up().multiply(hidden, splitUp);
        addBias(splitGate, expert.gateBias());
        addBias(splitUp, expert.upBias());
        GptOssMath.swigluOaiSplit(splitGate, splitUp, activation, alpha, limit);
      }

      if (!expert.isFused()) {
        expert.splitDown().multiply(activation, expertOutput);
      } else if (q8Activations) {
        activationQ8.quantize(activation, 1);
        expert.down().multiplyQ8(activationQ8, expertOutput);
      } else {
        expert.down().multiply(activation, expertOutput);
      }
      addBias(expertOutput, expert.downBias());

      float routingWeight = routingWeights[route];
      for (int index = 0; index < output.length; index++) {
        output[index] += routingWeight * expertOutput[index];
      }
    }
  }

  private void validateGeometry(GptOssMxfp4ExpertWeights.Expert expert, int index) {
    // Every expert must have the SAME shape as the first: the scratch buffers and the branch above
    // are
    // chosen once from expert 0, so a mixed set would be read with the wrong indexing.
    if (expert.isFused() != fusedShape) {
      throw new IllegalArgumentException(
          "expert " + index + " has a different storage shape from expert 0");
    }
    boolean consistent =
        expert.isFused()
            ? expert.gateUp().columns() == hiddenSize
                && expert.gateUp().rows() == Math.multiplyExact(2, intermediateSize)
                && expert.gateUpBias().length == Math.multiplyExact(2, intermediateSize)
                && expert.down().columns() == intermediateSize
                && expert.down().rows() == hiddenSize
            : expert.gate().columns() == hiddenSize
                && expert.gate().rows() == intermediateSize
                && expert.up().columns() == hiddenSize
                && expert.up().rows() == intermediateSize
                && expert.gateBias().length == intermediateSize
                && expert.upBias().length == intermediateSize
                && expert.splitDown().columns() == intermediateSize
                && expert.splitDown().rows() == hiddenSize;
    if (!consistent || expert.downBias().length != hiddenSize) {
      throw new IllegalArgumentException(
          "expert " + index + " has inconsistent projection geometry");
    }
  }

  private void validateForwardArguments(
      float[] hidden, int[] selectedExperts, float[] routingWeights, float[] output) {
    Objects.requireNonNull(hidden, "hidden");
    Objects.requireNonNull(selectedExperts, "selectedExperts");
    Objects.requireNonNull(routingWeights, "routingWeights");
    Objects.requireNonNull(output, "output");
    if (hidden.length != hiddenSize) {
      throw new IllegalArgumentException(
          "hidden length must equal " + hiddenSize + "; got " + hidden.length);
    }
    if (output.length != hiddenSize) {
      throw new IllegalArgumentException(
          "output length must equal " + hiddenSize + "; got " + output.length);
    }
    if (hidden == output) {
      throw new IllegalArgumentException("hidden and output must not alias");
    }
    if (selectedExperts.length == 0 || selectedExperts.length > weights.expertCount()) {
      throw new IllegalArgumentException("selected experts must contain a valid non-empty top-k");
    }
    if (routingWeights.length != selectedExperts.length) {
      throw new IllegalArgumentException("routing weights must match the selected expert count");
    }
    boolean[] seen = new boolean[weights.expertCount()];
    for (int route = 0; route < selectedExperts.length; route++) {
      int expert = selectedExperts[route];
      if (expert < 0 || expert >= weights.expertCount()) {
        throw new IllegalArgumentException(
            "selected expert is outside the expert range: " + expert);
      }
      if (seen[expert]) {
        throw new IllegalArgumentException("duplicate selected expert: " + expert);
      }
      seen[expert] = true;
      float routingWeight = routingWeights[route];
      if (!(routingWeight >= 0.0f) || !Float.isFinite(routingWeight)) {
        throw new IllegalArgumentException(
            "routing weight " + route + " must be finite and non-negative: " + routingWeight);
      }
    }
  }

  private static void addBias(float[] values, float[] bias) {
    for (int index = 0; index < values.length; index++) {
      values[index] += bias[index];
    }
  }
}
