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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The split is where a result gets quietly invalidated. If one passage reaches both the training
 * and the sealed split, the head has seen the answer's context before it is scored, and the
 * agreement figure means nothing. These tests hold the planner to passage-level separation, to
 * determinism, and to reporting the base rate it actually drew rather than the one that was hoped
 * for.
 */
@Tag("unit")
class SplitPlannerTest {

  private static final long SEED = 20260920L;

  @Test
  void noPassageEverReachesTwoSplits() {
    SplitPlan plan = SplitPlanner.plan(pool(400, 4), 200, SEED, 100, 40, 60);

    Set<String> train = passages(plan.train());
    Set<String> calibration = passages(plan.calibration());
    Set<String> sealed = passages(plan.sealedTest());

    assertThat(train).doesNotContainAnyElementsOf(calibration);
    assertThat(train).doesNotContainAnyElementsOf(sealed);
    assertThat(calibration).doesNotContainAnyElementsOf(sealed);
  }

  @Test
  void theSameSeedDrawsTheIdenticalSplit() {
    SplitPlan first = SplitPlanner.plan(pool(400, 4), 200, SEED, 100, 40, 60);
    SplitPlan second = SplitPlanner.plan(pool(400, 4), 200, SEED, 100, 40, 60);

    assertThat(ids(second.train())).isEqualTo(ids(first.train()));
    assertThat(ids(second.calibration())).isEqualTo(ids(first.calibration()));
    assertThat(ids(second.sealedTest())).isEqualTo(ids(first.sealedTest()));
  }

  @Test
  void adifferentSeedDrawsADifferentSplit() {
    SplitPlan first = SplitPlanner.plan(pool(400, 4), 200, SEED, 100, 40, 60);
    SplitPlan other = SplitPlanner.plan(pool(400, 4), 200, SEED + 1, 100, 40, 60);

    assertThat(ids(other.train())).isNotEqualTo(ids(first.train()));
  }

  @Test
  void everySampledItemLandsInExactlyOneSplit() {
    SplitPlan plan = SplitPlanner.plan(pool(400, 4), 200, SEED, 100, 40, 60);

    List<String> all = new ArrayList<>();
    all.addAll(ids(plan.train()));
    all.addAll(ids(plan.calibration()));
    all.addAll(ids(plan.sealedTest()));

    assertThat(all).hasSize(200);
    assertThat(new HashSet<>(all)).hasSize(200);
  }

  @Test
  void splitSizesLandNearTheirTargetsDespiteWholePassageAssignment() {
    SplitPlan plan = SplitPlanner.plan(pool(400, 4), 200, SEED, 100, 40, 60);

    assertThat(plan.train()).hasSizeBetween(92, 108);
    assertThat(plan.calibration()).hasSizeBetween(32, 48);
    assertThat(plan.sealedTest()).hasSizeBetween(52, 68);
  }

  @Test
  void theRealizedBaseRateIsReportedRatherThanAssumed() {
    SplitPlan plan = SplitPlanner.plan(pool(400, 4), 200, SEED, 100, 40, 60);

    double answerable = plan.sealedTest().stream().filter(CorpusItem::answerable).count();
    double expected = answerable / plan.sealedTest().size();

    assertThat(plan.sealedTestMajorityFloor()).isBetween(0.5, 1.0);
    assertThat(plan.sealedTestMajorityFloor()).isEqualTo(Math.max(expected, 1.0 - expected));
  }

  @Test
  void aPoolTooSmallForTheRequestedSampleIsRefused() {
    assertThatThrownBy(() -> SplitPlanner.plan(pool(50, 4), 200, SEED, 100, 40, 60))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void samplingPreservesTheNaturalBaseRateRatherThanRebalancingIt() {
    // Pool is deliberately skewed 75/25, as CUAD's is. Sampling must not quietly even it out.
    List<CorpusItem> skewed = new ArrayList<>();
    for (int index = 0; index < 800; index++) {
      skewed.add(
          new CorpusItem("q" + index, "p" + (index / 4), "question " + index, index % 4 == 0));
    }

    SplitPlan plan = SplitPlanner.plan(skewed, 400, SEED, 200, 80, 120);

    long answerable =
        plan.train().stream().filter(CorpusItem::answerable).count()
            + plan.calibration().stream().filter(CorpusItem::answerable).count()
            + plan.sealedTest().stream().filter(CorpusItem::answerable).count();

    assertThat(answerable / 400.0).isBetween(0.20, 0.30);
  }

  private static List<CorpusItem> pool(int items, int perPassage) {
    List<CorpusItem> pool = new ArrayList<>(items);
    for (int index = 0; index < items; index++) {
      pool.add(
          new CorpusItem(
              "q" + index, "p" + (index / perPassage), "question " + index, index % 2 == 0));
    }
    return pool;
  }

  private static Set<String> passages(List<CorpusItem> items) {
    Set<String> passages = new HashSet<>();
    for (CorpusItem item : items) {
      passages.add(item.passageId());
    }
    return passages;
  }

  private static List<String> ids(List<CorpusItem> items) {
    return items.stream().map(CorpusItem::id).toList();
  }
}
