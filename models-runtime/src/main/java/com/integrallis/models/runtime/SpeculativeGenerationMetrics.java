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
package com.integrallis.models.runtime;

import java.util.List;

/** Per-request measurements for speculative generation. */
public record SpeculativeGenerationMetrics(
    boolean active,
    int draftAttempts,
    int proposedTokens,
    int acceptedTokens,
    int rollbacks,
    int suppressedSteps,
    int ordinaryForwardCalls,
    List<Integer> proposedByPosition,
    List<Integer> acceptedByPosition,
    List<Integer> verificationBatchHistogram,
    long draftSearchNanos,
    long verificationNanos,
    long ordinaryForwardNanos) {

  public SpeculativeGenerationMetrics {
    proposedByPosition = List.copyOf(proposedByPosition);
    acceptedByPosition = List.copyOf(acceptedByPosition);
    verificationBatchHistogram = List.copyOf(verificationBatchHistogram);
  }

  public static SpeculativeGenerationMetrics inactive() {
    return new SpeculativeGenerationMetrics(
        false, 0, 0, 0, 0, 0, 0, List.of(), List.of(), List.of(), 0, 0, 0);
  }

  public double acceptanceRate() {
    return proposedTokens == 0 ? 0.0 : (double) acceptedTokens / proposedTokens;
  }

  public double meanAcceptedTokens() {
    return draftAttempts == 0 ? 0.0 : (double) acceptedTokens / draftAttempts;
  }
}
