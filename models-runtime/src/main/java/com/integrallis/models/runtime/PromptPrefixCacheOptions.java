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
 * Settings for a pipeline's automatic prompt-prefix cache (#474).
 *
 * <p>With the cache on, every {@code generate} call on the pipeline looks its prompt up against the
 * prompts that came before it and reuses the longest matching key/value prefix instead of
 * prefilling it again: an agent that replays its instructions, tools and history on every turn, or
 * several conversations sharing one system prompt, pay for the new tokens only. The cache is off
 * unless a pipeline is built with these options.
 *
 * @param maximumBytes the key/value bytes the cache may hold across its entries, as each entry
 *     reports them; the least recently used entries go first when it is exceeded. One entry larger
 *     than the budget is kept, so the budget bounds the cache without refusing a request.
 * @param minimumReusableTokens the shortest shared prefix worth reusing; a shorter match is a miss
 */
public record PromptPrefixCacheOptions(long maximumBytes, int minimumReusableTokens) {

  public PromptPrefixCacheOptions {
    if (maximumBytes <= 0) {
      throw new IllegalArgumentException("maximumBytes must be > 0: " + maximumBytes);
    }
    if (minimumReusableTokens < 1) {
      throw new IllegalArgumentException(
          "minimumReusableTokens must be >= 1: " + minimumReusableTokens);
    }
  }

  /** Options with the given byte budget and no minimum beyond one token. */
  public static PromptPrefixCacheOptions withMaximumBytes(long maximumBytes) {
    return new PromptPrefixCacheOptions(maximumBytes, 1);
  }
}
