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

import java.time.Duration;
import java.util.Objects;

/** Measurements for preparing one prompt prefix without decoding output tokens. */
public record PromptPrefillMetrics(
    Duration tokenization,
    Duration cachePreparation,
    Duration prefill,
    Duration total,
    PromptCacheMetrics promptCache) {

  public PromptPrefillMetrics {
    tokenization = requireDuration(tokenization, "tokenization");
    cachePreparation = requireDuration(cachePreparation, "cachePreparation");
    prefill = requireDuration(prefill, "prefill");
    total = requireDuration(total, "total");
    promptCache = Objects.requireNonNull(promptCache, "promptCache");
  }

  private static Duration requireDuration(Duration value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isNegative()) {
      throw new IllegalArgumentException(name + " must not be negative");
    }
    return value;
  }
}
