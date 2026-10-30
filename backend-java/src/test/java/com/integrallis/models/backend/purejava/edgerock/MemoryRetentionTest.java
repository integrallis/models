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
package com.integrallis.models.backend.purejava.edgerock;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Whether one early, distinctive anchor survives a long flood of duller ones.
 *
 * <p>Written to diagnose a specific failure rather than to decorate the suite. On the publisher's
 * {@code memory_two_keys_8k} case our decoder answered with the room access code from the middle of
 * the prompt instead of the project identifier in its first sentence. That prompt is one
 * distinctive fact, then about 10400 tokens of near-identical filler, then the question. If a bank
 * of 1024 slots cannot hold a high-scoring anchor across the forty commits that flood produces, the
 * retrieval failure is here and needs no model to see.
 *
 * <p>This exercises the retention policy with synthetic scores, so it says nothing about whether
 * the learned admission scorer gives the needle a high score in the first place. It isolates one
 * half of that question: given that it does, is the anchor kept.
 */
@Tag("unit")
class MemoryRetentionTest {

  private static final int GROUPS = 2;
  private static final int CAPACITY = 1_024;
  private static final int CHUNK = 256;
  private static final int DIM = 4;

  @Test
  void aHighScoringEarlyAnchorSurvivesTenThousandDullerTokens() {
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);

    // The needle at position 0, scored above everything that follows.
    stage(bank, 10.0f, 1.0f);
    // Then the flood: identical, lower-scoring, and far more of it than the bank can hold.
    for (int token = 1; token < 10_400; token++) {
      stage(bank, 0.5f, 0.0f);
    }
    bank.commit();

    assertThat(bank.total()).isEqualTo(10_400);
    int[] kept = bank.selection(0).positionsCopy();
    assertThat(kept)
        .as("the position-0 needle must still be in a bank of %d slots", CAPACITY)
        .contains(0);
  }

  @Test
  void theNeedleIsFoundByItsOwnKeyAndNotByTheFloodsKey() {
    // Retention is necessary but not sufficient: the kept anchor also has to be the one a query
    // for it selects. A bank that keeps the needle but whose read returns the flood's value would
    // produce exactly the observed failure -- a confident answer taken from the wrong record.
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    stage(bank, 10.0f, 1.0f);
    for (int token = 1; token < 2_000; token++) {
      stage(bank, 0.5f, 0.0f);
    }
    bank.commit();

    int[] kept = bank.selection(0).positionsCopy();
    int needleSlot = -1;
    for (int slot = 0; slot < kept.length; slot++) {
      if (kept[slot] == 0) {
        needleSlot = slot;
        break;
      }
    }
    assertThat(needleSlot).as("needle retained").isNotNegative();
    // Its stored key must still be the needle's key, not overwritten by a later record.
    assertThat(bank.key(0, needleSlot)[0]).isEqualTo(1.0f);
    assertThat(bank.value(0, needleSlot)[0]).isEqualTo(1.0f);
  }

  @Test
  void eachGroupKeepsItsOwnAnchorsBecauseTheyScoreThemDifferently() {
    // The test that was missing. admission_up is [rank, groups] and emits a score per key/value
    // group, so the groups disagree about what is worth keeping -- that disagreement is what
    // "independent" memory means. Every fixture above hands all groups the same score, and a
    // uniform fixture cannot fail on the axis it holds constant: with equal scores a bank that
    // collapses the groups into one shared selection looks perfectly correct.
    int groups = 2;
    int capacity = 2;
    // The chunk cannot exceed the capacity, so this is two commits of two rather than one of four.
    int chunk = 2;
    MemoryBank bank = new MemoryBank(groups, capacity, chunk, 1);

    // Group 0 prefers the first two tokens, group 1 the last two.
    float[][] perGroupScores = {{9f, 1f}, {8f, 2f}, {1f, 8f}, {2f, 9f}};
    for (int token = 0; token < 4; token++) {
      float[][] keys = new float[groups][1];
      float[][] values = new float[groups][1];
      float[] scores = new float[groups];
      for (int group = 0; group < groups; group++) {
        // Distinct per group as well as per token, so a record taken from the wrong group shows up.
        keys[group][0] = (token + 1) * 10f + group;
        values[group][0] = (token + 1) * 10f + group;
        scores[group] = perGroupScores[token][group];
      }
      bank.stage(keys, values, scores);
    }

    assertThat(bank.selection(0).positionsCopy())
        .as("group 0 ranked tokens 0 and 1 highest")
        .containsExactly(0, 1);
    assertThat(bank.selection(1).positionsCopy())
        .as("group 1 ranked tokens 3 and 2 highest")
        .containsExactly(3, 2);

    // And each group's records must be its own, in its own slot order.
    assertThat(bank.key(0, 0)[0]).isEqualTo(10f);
    assertThat(bank.key(0, 1)[0]).isEqualTo(20f);
    assertThat(bank.key(1, 0)[0]).isEqualTo(41f);
    assertThat(bank.key(1, 1)[0]).isEqualTo(31f);
  }

  private static void stage(MemoryBank bank, float score, float payload) {
    float[][] keys = new float[GROUPS][DIM];
    float[][] values = new float[GROUPS][DIM];
    float[] scores = new float[GROUPS];
    for (int group = 0; group < GROUPS; group++) {
      keys[group][0] = payload;
      values[group][0] = payload;
      scores[group] = score;
    }
    bank.stage(keys, values, scores);
  }
}
