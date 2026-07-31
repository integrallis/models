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
package com.integrallis.models.rag;

/** Size limits for retrieved evidence and deterministic extractive fallbacks. */
public record GroundingLimits(
    int maximumRetrievedDocuments,
    int maximumRetrievedCharacters,
    int maximumExtractiveDocuments,
    int maximumExtractiveCharacters) {
  public static final int DEFAULT_MAXIMUM_RETRIEVED_DOCUMENTS = 32;
  public static final int DEFAULT_MAXIMUM_RETRIEVED_CHARACTERS = 131_072;
  public static final int DEFAULT_MAXIMUM_EXTRACTIVE_DOCUMENTS = 3;
  public static final int DEFAULT_MAXIMUM_EXTRACTIVE_CHARACTERS = 4_000;

  public GroundingLimits {
    requirePositive(maximumRetrievedDocuments, "maximumRetrievedDocuments");
    requirePositive(maximumRetrievedCharacters, "maximumRetrievedCharacters");
    requirePositive(maximumExtractiveDocuments, "maximumExtractiveDocuments");
    requirePositive(maximumExtractiveCharacters, "maximumExtractiveCharacters");
  }

  public static GroundingLimits productionDefault() {
    return new GroundingLimits(
        DEFAULT_MAXIMUM_RETRIEVED_DOCUMENTS,
        DEFAULT_MAXIMUM_RETRIEVED_CHARACTERS,
        DEFAULT_MAXIMUM_EXTRACTIVE_DOCUMENTS,
        DEFAULT_MAXIMUM_EXTRACTIVE_CHARACTERS);
  }

  private static void requirePositive(int value, String name) {
    if (value < 1) {
      throw new IllegalArgumentException(name + " must be positive");
    }
  }
}
