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

/** The activated branch's next-token evidence for calling or declining a tool. */
public record ToolDecisionScore(
    int callTokenId, float callLogit, int noCallTokenId, float noCallLogit) {

  public ToolDecisionScore {
    if (callTokenId < 0) {
      throw new IllegalArgumentException("callTokenId must be >= 0");
    }
    if (noCallTokenId < 0) {
      throw new IllegalArgumentException("noCallTokenId must be >= 0");
    }
    if (callTokenId == noCallTokenId) {
      throw new IllegalArgumentException("call and no-call token IDs must differ");
    }
    if (!Float.isFinite(callLogit) || !Float.isFinite(noCallLogit)) {
      throw new IllegalArgumentException("tool-decision logits must be finite");
    }
  }

  /** Positive values favor a call; negative values favor abstention. */
  public float callMargin() {
    return callLogit - noCallLogit;
  }

  /** Returns whether the call margin strictly exceeds a calibrated threshold. */
  public boolean shouldCall(float minimumCallMargin) {
    if (!Float.isFinite(minimumCallMargin)) {
      throw new IllegalArgumentException("minimumCallMargin must be finite");
    }
    return callMargin() > minimumCallMargin;
  }
}
