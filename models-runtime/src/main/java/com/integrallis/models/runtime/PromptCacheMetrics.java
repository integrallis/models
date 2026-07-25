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

/** Token-level measurements for cross-request prompt KV reuse. */
public record PromptCacheMetrics(
    boolean supported, int inputTokens, int cacheReadInputTokens, int cacheWriteInputTokens) {

  public PromptCacheMetrics {
    if (inputTokens < 0
        || cacheReadInputTokens < 0
        || cacheWriteInputTokens < 0
        || cacheReadInputTokens + cacheWriteInputTokens > inputTokens) {
      throw new IllegalArgumentException(
          "cache token counts must be non-negative and not exceed inputTokens");
    }
    if (!supported && (cacheReadInputTokens != 0 || cacheWriteInputTokens != 0)) {
      throw new IllegalArgumentException("unsupported prompt caches cannot report cache activity");
    }
  }

  /** Returns the fraction of input tokens restored from the retained KV prefix. */
  public double hitRate() {
    return inputTokens == 0 ? 0 : (double) cacheReadInputTokens / inputTokens;
  }

  static PromptCacheMetrics unavailable() {
    return new PromptCacheMetrics(false, 0, 0, 0);
  }
}
