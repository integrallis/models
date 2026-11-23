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

/**
 * What a pipeline's prompt-prefix cache has done so far.
 *
 * @param lookups prompts looked up
 * @param hits lookups that reused a prefix
 * @param misses lookups that reused nothing
 * @param reusedTokens prompt tokens served from a cached prefix, summed over hits
 * @param evaluatedTokens prompt tokens the backend had to evaluate, summed over lookups
 * @param insertions prefixes added
 * @param evictions prefixes dropped to stay within the byte budget
 * @param entries prefixes held now
 * @param heldBytes bytes the held prefixes report, summed per entry; storage two entries share is
 *     counted in each
 */
public record PromptPrefixCacheMetrics(
    long lookups,
    long hits,
    long misses,
    long reusedTokens,
    long evaluatedTokens,
    long insertions,
    long evictions,
    int entries,
    long heldBytes) {

  /** Hits over lookups, or zero before the first lookup. */
  public double hitRate() {
    return lookups == 0 ? 0.0 : (double) hits / lookups;
  }
}
