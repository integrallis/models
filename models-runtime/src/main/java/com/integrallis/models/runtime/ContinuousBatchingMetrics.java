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

/**
 * Lifetime scheduler measurements for one loaded physical model.
 *
 * @param completedRequests requests that completed normally
 * @param failedRequests requests that failed or were rejected
 * @param rejectedRequests requests rejected because the scheduler was closed or its queue full
 * @param batchInvocations backend batch decode calls
 * @param sessionSteps session decode steps across all batch calls
 * @param largestBatch largest batch decoded
 * @param activeRequests requests currently admitted
 * @param queuedRequests requests currently waiting
 * @param repetitionLoopStops completed requests the repetition-loop detector stopped
 */
public record ContinuousBatchingMetrics(
    long completedRequests,
    long failedRequests,
    long rejectedRequests,
    long batchInvocations,
    long sessionSteps,
    int largestBatch,
    int activeRequests,
    int queuedRequests,
    long repetitionLoopStops) {

  public ContinuousBatchingMetrics {
    if (completedRequests < 0
        || failedRequests < 0
        || rejectedRequests < 0
        || batchInvocations < 0
        || sessionSteps < 0
        || largestBatch < 0
        || activeRequests < 0
        || queuedRequests < 0
        || repetitionLoopStops < 0) {
      throw new IllegalArgumentException("continuous batching metrics must not be negative");
    }
  }

  /**
   * Creates metrics without a repetition-loop count, preserving the original signature.
   *
   * @param completedRequests requests that completed normally
   * @param failedRequests requests that failed or were rejected
   * @param rejectedRequests rejected requests
   * @param batchInvocations backend batch decode calls
   * @param sessionSteps session decode steps
   * @param largestBatch largest batch decoded
   * @param activeRequests requests currently admitted
   * @param queuedRequests requests currently waiting
   */
  public ContinuousBatchingMetrics(
      long completedRequests,
      long failedRequests,
      long rejectedRequests,
      long batchInvocations,
      long sessionSteps,
      int largestBatch,
      int activeRequests,
      int queuedRequests) {
    this(
        completedRequests,
        failedRequests,
        rejectedRequests,
        batchInvocations,
        sessionSteps,
        largestBatch,
        activeRequests,
        queuedRequests,
        0);
  }

  /** Mean number of independent sessions advanced per backend batch call. */
  public double meanBatchSize() {
    return batchInvocations == 0 ? 0 : (double) sessionSteps / batchInvocations;
  }
}
