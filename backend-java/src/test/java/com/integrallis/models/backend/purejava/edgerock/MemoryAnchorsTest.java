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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.edgerock.MemoryAnchors.Selection;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MemoryAnchorsTest {

  @Test
  void keepsTheHighestScoresAndPadsTheRest() {
    Selection kept =
        MemoryAnchors.select(
            new float[] {0.1f, 0.9f, 0.5f}, new int[] {0, 1, 2}, new int[] {0, 1, 2}, 5);

    assertThat(kept.capacity()).isEqualTo(5);
    assertThat(kept.referencesCopy()).containsExactly(1, 2, 0, -1, -1);
    assertThat(kept.positionsCopy()).containsExactly(1, 2, 0, -1, -1);
    assertThat(kept.score(3)).isEqualTo(Float.NEGATIVE_INFINITY);
  }

  @Test
  void truncatesToCapacityByScore() {
    Selection kept =
        MemoryAnchors.select(
            new float[] {0.1f, 0.9f, 0.5f, 0.7f},
            new int[] {0, 1, 2, 3},
            new int[] {0, 1, 2, 3},
            2);

    assertThat(kept.referencesCopy()).containsExactly(1, 3);
  }

  @Test
  void breaksTiesByEarlierPosition() {
    // The behaviour the whole class exists for. Equal scores, positions deliberately out of index
    // order: the earlier position must win, because the position pass runs first and the score
    // pass is stable. An unstable sort passes the other tests and fails this one.
    Selection kept =
        MemoryAnchors.select(
            new float[] {0.5f, 0.5f, 0.5f}, new int[] {40, 10, 20}, new int[] {100, 101, 102}, 2);

    assertThat(kept.positionsCopy()).containsExactly(10, 20);
    assertThat(kept.referencesCopy()).containsExactly(101, 102);
  }

  @Test
  void treatsANegativeReferenceAsAbsentWhateverItsScore() {
    // A bank that is not yet full carries empty slots. They must never be selected, even when
    // their score field would otherwise win.
    Selection kept =
        MemoryAnchors.select(new float[] {9.0f, 0.1f}, new int[] {0, 1}, new int[] {-1, 7}, 2);

    assertThat(kept.referencesCopy()).containsExactly(7, -1);
    assertThat(kept.positionsCopy()).containsExactly(1, -1);
  }

  @Test
  void mergeIsTheCommitStepAndKeepsTheBestOfBoth() {
    Selection bank =
        MemoryAnchors.select(new float[] {0.8f, 0.2f}, new int[] {0, 1}, new int[] {0, 1}, 3);
    Selection pending =
        MemoryAnchors.select(new float[] {0.9f, 0.1f}, new int[] {2, 3}, new int[] {10, 11}, 3);

    Selection committed = MemoryAnchors.merge(bank, pending, 3);

    assertThat(committed.referencesCopy()).containsExactly(10, 0, 1);
    assertThat(committed.score(0)).isEqualTo(0.9f);
  }

  @Test
  void mergeIsAssociativeSoPrefillAndDecodeAgree() {
    // Prefill reduces over a tree, decode merges once per commit. They must reach the same bank,
    // or a cached prefix would disagree with a freshly decoded one.
    Random random = new Random(42);
    Selection a = random(random, 4, 0);
    Selection b = random(random, 4, 4);
    Selection c = random(random, 4, 8);

    Selection leftFirst = MemoryAnchors.merge(MemoryAnchors.merge(a, b, 4), c, 4);
    Selection rightFirst = MemoryAnchors.merge(a, MemoryAnchors.merge(b, c, 4), 4);

    assertThat(leftFirst.referencesCopy()).containsExactly(rightFirst.referencesCopy());
    assertThat(leftFirst.positionsCopy()).containsExactly(rightFirst.positionsCopy());
  }

  @Test
  void mergingAnEmptyBankIsTheIdentity() {
    Selection pending =
        MemoryAnchors.select(new float[] {0.4f, 0.6f}, new int[] {5, 6}, new int[] {2, 3}, 4);

    Selection committed = MemoryAnchors.merge(MemoryAnchors.empty(4), pending, 4);

    assertThat(committed.referencesCopy()).containsExactly(pending.referencesCopy());
    assertThat(committed.positionsCopy()).containsExactly(pending.positionsCopy());
  }

  @Test
  void rejectsMismatchedInputsAndNonPositiveCapacity() {
    assertThatThrownBy(
            () -> MemoryAnchors.select(new float[] {1f}, new int[] {1, 2}, new int[] {0}, 1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> MemoryAnchors.select(new float[] {1f}, new int[] {1}, new int[] {0}, 0))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static Selection random(Random random, int width, int positionBase) {
    float[] scores = new float[width];
    int[] positions = new int[width];
    int[] references = new int[width];
    for (int i = 0; i < width; i++) {
      scores[i] = random.nextFloat();
      positions[i] = positionBase + i;
      references[i] = positionBase + i;
    }
    return MemoryAnchors.select(scores, positions, references, width);
  }
}
