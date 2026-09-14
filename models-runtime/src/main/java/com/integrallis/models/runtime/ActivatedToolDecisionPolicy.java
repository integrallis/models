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

import java.util.Objects;

/** A qualified call/abstention policy over physically shared base and activated branches. */
public record ActivatedToolDecisionPolicy(
    int callTokenId,
    int noCallTokenId,
    float primarySpecialistMargin,
    float rescueSpecialistMargin,
    float rescueBaseSpecialistGap,
    float rescueMaximumBaseMargin) {

  /** Why a policy selected the activated tool branch or the exact base branch. */
  public enum Decision {
    CALL_PRIMARY(true),
    CALL_RESCUE(true),
    NO_CALL(false);

    private final boolean call;

    Decision(boolean call) {
      this.call = call;
    }

    public boolean shouldCall() {
      return call;
    }
  }

  public ActivatedToolDecisionPolicy {
    if (callTokenId < 0 || noCallTokenId < 0 || callTokenId == noCallTokenId) {
      throw new IllegalArgumentException(
          "call and no-call token IDs must be distinct and nonnegative");
    }
    requireFinite(primarySpecialistMargin, "primarySpecialistMargin");
    requireFinite(rescueSpecialistMargin, "rescueSpecialistMargin");
    requireFinite(rescueBaseSpecialistGap, "rescueBaseSpecialistGap");
    requireFinite(rescueMaximumBaseMargin, "rescueMaximumBaseMargin");
    if (rescueSpecialistMargin >= primarySpecialistMargin) {
      throw new IllegalArgumentException(
          "rescueSpecialistMargin must be below primarySpecialistMargin");
    }
    if (rescueBaseSpecialistGap <= 0) {
      throw new IllegalArgumentException("rescueBaseSpecialistGap must be > 0");
    }
    if (rescueMaximumBaseMargin <= rescueSpecialistMargin + rescueBaseSpecialistGap) {
      throw new IllegalArgumentException("rescue thresholds do not define a reachable margin band");
    }
  }

  /** Selects the tool specialist or base branch using strict, qualification-bound thresholds. */
  public Decision decide(ToolDecisionScore base, ToolDecisionScore specialist) {
    requireCompatible(Objects.requireNonNull(base, "base"));
    requireCompatible(Objects.requireNonNull(specialist, "specialist"));
    float specialistMargin = specialist.callMargin();
    if (specialistMargin > primarySpecialistMargin) {
      return Decision.CALL_PRIMARY;
    }
    float baseMargin = base.callMargin();
    if (specialistMargin > rescueSpecialistMargin
        && baseMargin - specialistMargin > rescueBaseSpecialistGap
        && baseMargin < rescueMaximumBaseMargin) {
      return Decision.CALL_RESCUE;
    }
    return Decision.NO_CALL;
  }

  private void requireCompatible(ToolDecisionScore score) {
    if (score.callTokenId() != callTokenId || score.noCallTokenId() != noCallTokenId) {
      throw new IllegalArgumentException("tool-decision score uses different token IDs");
    }
  }

  private static void requireFinite(float value, String name) {
    if (!Float.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite");
    }
  }
}
