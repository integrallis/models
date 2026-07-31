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
package com.integrallis.models.api;

import java.util.Objects;

/**
 * Metadata describing a loaded model's architecture.
 *
 * <p>Names are nonblank, dimensions are positive, and {@code numHeads} is divisible by {@code
 * numKvHeads} for grouped-query attention.
 */
public record ModelMetadata(
    String modelFamily,
    String modelName,
    int contextLength,
    int vocabSize,
    int embeddingDim,
    int numLayers,
    int numHeads,
    int numKvHeads) {

  public ModelMetadata {
    modelFamily = requireText(modelFamily, "modelFamily");
    modelName = requireText(modelName, "modelName");
    requirePositive(contextLength, "contextLength");
    requirePositive(vocabSize, "vocabSize");
    requirePositive(embeddingDim, "embeddingDim");
    requirePositive(numLayers, "numLayers");
    requirePositive(numHeads, "numHeads");
    requirePositive(numKvHeads, "numKvHeads");
    if (numHeads % numKvHeads != 0) {
      throw new IllegalArgumentException(
          "numHeads must be divisible by numKvHeads: " + numHeads + " and " + numKvHeads);
    }
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value.trim();
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive: " + value);
    }
  }
}
