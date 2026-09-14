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

import com.integrallis.models.api.SharedInferencePrefix;

/**
 * Immutable token and KV-cache prefix prepared by one {@link InferencePipeline}.
 *
 * <p>The handle does not own the loaded model. It remains usable until its pipeline is closed and
 * cannot be transferred to another pipeline.
 */
public final class SharedPromptPrefix {
  final InferencePipeline owner;
  final SharedInferencePrefix backendPrefix;
  final int[] promptTokens;
  private final long sharedBytes;

  SharedPromptPrefix(
      InferencePipeline owner,
      SharedInferencePrefix backendPrefix,
      int[] promptTokens,
      long sharedBytes) {
    this.owner = owner;
    this.backendPrefix = backendPrefix;
    this.promptTokens = promptTokens.clone();
    this.sharedBytes = sharedBytes;
  }

  /** Returns the number of immutable prompt tokens shared by every fork. */
  public int tokenCount() {
    return promptTokens.length;
  }

  /** Returns the key/value and bookkeeping bytes physically shared by every fork. */
  public long sharedBytes() {
    return sharedBytes;
  }
}
