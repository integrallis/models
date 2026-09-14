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

/** Optional access to the final normalized hidden state before vocabulary projection. */
public interface HiddenStateInferenceBackend extends InferenceBackend {

  /** Returns whether the loaded architecture exposes final normalized hidden states. */
  boolean supportsHiddenState();

  /** Runs one step and returns a stable final normalized hidden-state snapshot. */
  float[] forwardHiddenState(int token, int position);

  /** Runs one hidden-state step whose returned storage may be reused by the next backend call. */
  default float[] forwardHiddenStateTransient(int token, int position) {
    return forwardHiddenState(token, position);
  }

  /**
   * Processes a contiguous sequence and returns its final normalized hidden state. Returned storage
   * may be reused by the next backend invocation.
   */
  default float[] prefillHiddenState(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition < 0) {
      throw new IllegalArgumentException("startPosition must be >= 0");
    }
    float[] hidden = null;
    for (int index = 0; index < tokens.length; index++) {
      hidden = forwardHiddenStateTransient(tokens[index], Math.addExact(startPosition, index));
    }
    return hidden;
  }
}
