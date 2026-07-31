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

/**
 * Optional inference capability for verifying speculative token continuations.
 *
 * <p>Verification consumes every proposed token and advances backend state. Callers retain the
 * accepted prefix by rewinding to {@code checkpoint + acceptedTokenCount}.
 */
public interface SpeculativeInferenceBackend extends RewindableInferenceBackend {

  /**
   * Consumes contiguous proposed tokens and returns the logits produced after each token.
   *
   * <p>Row {@code i} predicts the token following {@code tokens[i]}. The start position must equal
   * the current checkpoint. The token array is non-null and nonempty and is not modified. The
   * returned batch is a stable snapshot.
   */
  LogitBatch verify(int[] tokens, int startPosition);

  /**
   * Consumes proposed tokens and may return storage reused by the next backend call.
   *
   * <p>The default preserves the stable snapshot returned by {@link #verify(int[], int)}. An
   * overridden transient result remains valid until the next backend invocation.
   */
  default LogitBatch verifyTransient(int[] tokens, int startPosition) {
    return verify(tokens, startPosition);
  }
}
