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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * The prompts a pipeline has seen, as immutable shared prefixes, looked up by their tokens.
 *
 * <p>An entry is a {@link SharedPromptPrefix} ending just before a prompt's last token: the last
 * token is the generation cue and is always evaluated, because evaluating it is what yields the
 * first logits. A lookup finds the entry sharing the longest token prefix with the new prompt,
 * never longer than the new prompt minus its last token. A match that covers an entry whole is
 * usable on any backend that shares prefixes; a match that stops inside an entry needs a truncated
 * view of it, which only a backend with portable prefixes provides, so on other backends such a
 * match is a miss.
 *
 * <p>Entries are held in access order and dropped least recently used first when their reported
 * bytes exceed the budget. Dropping an entry only drops this cache's reference; a session forked
 * from it keeps the storage alive until it closes. Callers serialize access through the pipeline's
 * backend monitor.
 */
final class PromptPrefixCache {

  /** A cached prefix and how many of its tokens the prompt shares. */
  record Match(SharedPromptPrefix entry, int reusableTokens) {
    boolean coversWholeEntry() {
      return reusableTokens == entry.tokenCount();
    }
  }

  private final long maximumBytes;
  private final int minimumReusableTokens;
  private final boolean truncatable;
  private final LinkedHashMap<SharedPromptPrefix, Long> entries =
      new LinkedHashMap<>(16, 0.75f, true);
  private long heldBytes;
  private long lookups;
  private long hits;
  private long misses;
  private long reusedTokens;
  private long evaluatedTokens;
  private long insertions;
  private long evictions;

  PromptPrefixCache(PromptPrefixCacheOptions options, boolean truncatable) {
    Objects.requireNonNull(options, "options");
    this.maximumBytes = options.maximumBytes();
    this.minimumReusableTokens = options.minimumReusableTokens();
    this.truncatable = truncatable;
  }

  /** The entry sharing the longest usable prefix with {@code promptTokens}, if any. */
  Optional<Match> lookup(int[] promptTokens) {
    Objects.requireNonNull(promptTokens, "promptTokens");
    lookups++;
    int limit = promptTokens.length - 1;
    SharedPromptPrefix best = null;
    int bestLength = 0;
    for (SharedPromptPrefix entry : entries.keySet()) {
      int common = Math.min(entry.commonPrefixLength(promptTokens), limit);
      if (common < minimumReusableTokens || common <= bestLength) {
        continue;
      }
      if (common < entry.tokenCount() && !truncatable) {
        continue;
      }
      best = entry;
      bestLength = common;
    }
    if (best == null) {
      misses++;
      evaluatedTokens += promptTokens.length;
      return Optional.empty();
    }
    entries.get(best);
    hits++;
    reusedTokens += bestLength;
    evaluatedTokens += promptTokens.length - bestLength;
    return Optional.of(new Match(best, bestLength));
  }

  /**
   * Adds a prefix, unless an entry already covers its tokens whole. An entry that is a strict
   * prefix of the new one is dropped when truncated views can serve it from the longer entry.
   */
  void insert(SharedPromptPrefix prefix) {
    Objects.requireNonNull(prefix, "prefix");
    List<SharedPromptPrefix> redundant = new ArrayList<>();
    SharedPromptPrefix covering = null;
    for (Map.Entry<SharedPromptPrefix, Long> candidate : entries.entrySet()) {
      SharedPromptPrefix entry = candidate.getKey();
      int common = entry.commonPrefixLength(prefix.promptTokens);
      if (common == prefix.tokenCount() && entry.tokenCount() >= prefix.tokenCount()) {
        covering = entry;
        break;
      }
      if (truncatable && common == entry.tokenCount()) {
        redundant.add(entry);
      }
    }
    if (covering != null) {
      entries.get(covering);
      return;
    }
    for (SharedPromptPrefix entry : redundant) {
      heldBytes -= entries.remove(entry);
    }
    long bytes = prefix.sharedBytes();
    entries.put(prefix, bytes);
    heldBytes += bytes;
    insertions++;
    while (heldBytes > maximumBytes && entries.size() > 1) {
      var eldest = entries.entrySet().iterator().next();
      heldBytes -= eldest.getValue();
      entries.remove(eldest.getKey());
      evictions++;
    }
  }

  PromptPrefixCacheMetrics metrics() {
    return new PromptPrefixCacheMetrics(
        lookups,
        hits,
        misses,
        reusedTokens,
        evaluatedTokens,
        insertions,
        evictions,
        entries.size(),
        heldBytes);
  }

  void clear() {
    entries.clear();
    heldBytes = 0;
  }
}
