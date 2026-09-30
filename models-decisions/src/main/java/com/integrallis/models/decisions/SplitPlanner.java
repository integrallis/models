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
package com.integrallis.models.decisions;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Draws a sample from a corpus and divides it, assigning whole passages rather than loose
 * questions.
 *
 * <p>Two properties matter more than tidy split sizes. The draw preserves the corpus's natural base
 * rate instead of rebalancing it, so the majority-class floor reported beside a result is the real
 * one. And every question about a passage lands in the same split, so a head cannot be scored on a
 * context it was fitted on. Whole-passage assignment means the split sizes land near their targets
 * rather than exactly on them; the sizes actually drawn are what get reported.
 */
public final class SplitPlanner {

  private SplitPlanner() {}

  /**
   * Draws {@code sampleSize} items and divides them.
   *
   * @param pool every candidate item from the corpus
   * @param sampleSize how many items to draw, uniformly and without rebalancing
   * @param seed the draw's seed, so the same split can be produced again
   * @param trainTarget the desired training-split size
   * @param calibrationTarget the desired calibration-split size
   * @param sealedTarget the desired sealed-split size
   */
  public static SplitPlan plan(
      List<CorpusItem> pool,
      int sampleSize,
      long seed,
      int trainTarget,
      int calibrationTarget,
      int sealedTarget) {
    Objects.requireNonNull(pool, "pool");
    if (sampleSize <= 0) {
      throw new IllegalArgumentException("sampleSize must be positive");
    }
    if (pool.size() < sampleSize) {
      throw new IllegalArgumentException(
          "pool holds " + pool.size() + " items, fewer than the " + sampleSize + " requested");
    }
    if (trainTarget < 0 || calibrationTarget < 0 || sealedTarget < 0) {
      throw new IllegalArgumentException("split targets must not be negative");
    }
    if (trainTarget + calibrationTarget + sealedTarget != sampleSize) {
      throw new IllegalArgumentException("split targets must sum to the sample size");
    }

    List<CorpusItem> shuffled = new ArrayList<>(pool);
    deterministicShuffle(shuffled, seed);
    List<CorpusItem> sample = shuffled.subList(0, sampleSize);

    // Group the drawn items by passage, preserving first-seen order so the grouping is stable.
    Map<String, List<CorpusItem>> byPassage = new LinkedHashMap<>();
    for (CorpusItem item : sample) {
      byPassage.computeIfAbsent(item.passageId(), key -> new ArrayList<>()).add(item);
    }

    List<CorpusItem> train = new ArrayList<>();
    List<CorpusItem> calibration = new ArrayList<>();
    List<CorpusItem> sealed = new ArrayList<>();
    int[] targets = {trainTarget, calibrationTarget, sealedTarget};
    List<List<CorpusItem>> splits = List.of(train, calibration, sealed);

    // Assign each passage whole, to whichever split is furthest below its target.
    for (List<CorpusItem> group : byPassage.values()) {
      int chosen = 0;
      int bestDeficit = Integer.MIN_VALUE;
      for (int index = 0; index < splits.size(); index++) {
        int deficit = targets[index] - splits.get(index).size();
        if (deficit > bestDeficit) {
          bestDeficit = deficit;
          chosen = index;
        }
      }
      splits.get(chosen).addAll(group);
    }

    return new SplitPlan(train, calibration, sealed);
  }

  /** A seeded Fisher-Yates, so a seed reproduces a draw exactly on any host. */
  private static void deterministicShuffle(List<CorpusItem> items, long seed) {
    long state = seed;
    for (int index = items.size() - 1; index > 0; index--) {
      state = state * 6364136223846793005L + 1442695040888963407L;
      int target = (int) Long.remainderUnsigned(state >>> 16, index + 1L);
      CorpusItem swap = items.get(index);
      items.set(index, items.get(target));
      items.set(target, swap);
    }
  }
}
