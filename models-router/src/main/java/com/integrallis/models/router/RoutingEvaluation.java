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
package com.integrallis.models.router;

import java.util.List;

/** Immutable request and runtime facts supplied to application routing extensions. */
public record RoutingEvaluation(
    RoutingRequest request,
    RoutingPolicy policy,
    String taskType,
    ModelCandidate candidate,
    List<ModelCandidate> eligibleCandidates,
    RoutingModelStatus status,
    String previousModelId,
    RoutingContinuity continuity) {

  /** Defensively copies the eligible fleet. */
  public RoutingEvaluation {
    eligibleCandidates = List.copyOf(eligibleCandidates);
  }
}
