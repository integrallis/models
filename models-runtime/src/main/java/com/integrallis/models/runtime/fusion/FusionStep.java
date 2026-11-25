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
package com.integrallis.models.runtime.fusion;

import java.util.Arrays;
import java.util.Objects;

/**
 * What one decode step of a {@link FusedInferenceBackend} did.
 *
 * @param position zero-based position of the token whose logits this step produced
 * @param consulted whether the auxiliaries voted
 * @param leadEntropyNats Shannon entropy of the lead's distribution, in nats
 * @param catchUpTokens per auxiliary, how many tokens it consumed to catch up this step (all zero
 *     when not consulted)
 * @param leadNanos time in the lead's forward or prefill
 * @param auxiliaryNanos time in the auxiliaries' catch-up (zero when not consulted)
 * @param fusionNanos time in the entropy and combination arithmetic
 */
public record FusionStep(
    int position,
    boolean consulted,
    double leadEntropyNats,
    int[] catchUpTokens,
    long leadNanos,
    long auxiliaryNanos,
    long fusionNanos) {

  /** Defensively copies the per-auxiliary counts. */
  public FusionStep {
    Objects.requireNonNull(catchUpTokens, "catchUpTokens");
    catchUpTokens = catchUpTokens.clone();
  }

  @Override
  public int[] catchUpTokens() {
    return catchUpTokens.clone();
  }

  @Override
  public String toString() {
    return "FusionStep[position="
        + position
        + ", consulted="
        + consulted
        + ", leadEntropyNats="
        + leadEntropyNats
        + ", catchUpTokens="
        + Arrays.toString(catchUpTokens)
        + "]";
  }
}
