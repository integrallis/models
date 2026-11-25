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

/**
 * Decides, per decode step, whether the auxiliaries are consulted.
 *
 * <p>The lead model always steps. A gate reads the Shannon entropy of the lead's next-token
 * distribution and the step's position, and answers whether the auxiliaries should catch up and
 * vote on this token. {@link #never()} makes a {@link FusedInferenceBackend} behave as its lead
 * alone; {@link #always()} fuses at every token.
 */
@FunctionalInterface
public interface EnsembleGate {

  /**
   * Returns whether the auxiliaries vote on the token at {@code position}.
   *
   * @param leadEntropyNats Shannon entropy, in nats, of the lead's distribution for this step
   * @param position zero-based position of the token about to be emitted
   */
  boolean consult(double leadEntropyNats, int position);

  /** A gate that never consults the auxiliaries. */
  static EnsembleGate never() {
    return (entropy, position) -> false;
  }

  /** A gate that consults the auxiliaries at every token. */
  static EnsembleGate always() {
    return (entropy, position) -> true;
  }
}
