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

@Tag("unit")
class MemoryBankTest {

  private static final int GROUPS = 2;
  private static final int CAPACITY = 4;
  private static final int CHUNK = 2;
  private static final int DIM = 3;

  @Test
  void stagedRecordsAreInvisibleUntilTheChunkIsFull() {
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);

    boolean committed = bank.stage(keys(1f), values(1f), scores(0.9f));

    assertThat(committed).isFalse();
    assertThat(bank.staged()).isEqualTo(1);
    // The bank the read path sees is still empty: a token cannot attend to itself.
    assertThat(bank.selection().occupancy()).isZero();
  }

  @Test
  void commitHappensExactlyAtTheChunkBoundary() {
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);

    assertThat(bank.stage(keys(1f), values(1f), scores(0.9f))).isFalse();
    assertThat(bank.stage(keys(2f), values(2f), scores(0.5f))).isTrue();

    assertThat(bank.staged()).isZero();
    assertThat(bank.selection().occupancy()).isEqualTo(2);
    assertThat(bank.total()).isEqualTo(2);
  }

  @Test
  void commitKeepsTheHighestScoringRecordsAcrossChunks() {
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    // Four tokens, two chunks, scores chosen so the second chunk partly evicts the first.
    bank.stage(keys(1f), values(1f), scores(0.10f));
    bank.stage(keys(2f), values(2f), scores(0.90f));
    bank.stage(keys(3f), values(3f), scores(0.20f));
    bank.stage(keys(4f), values(4f), scores(0.80f));

    assertThat(bank.selection().occupancy()).isEqualTo(4);
    assertThat(bank.selection().scoresCopy()).containsExactly(0.9f, 0.8f, 0.2f, 0.1f);
    // Positions travel with their scores, so the bank still knows when each record was written.
    assertThat(bank.selection().positionsCopy()).containsExactly(1, 3, 2, 0);
  }

  @Test
  void aFullBankEvictsTheWeakestRatherThanGrowing() {
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    for (int token = 0; token < CAPACITY; token++) {
      bank.stage(keys(token + 1f), values(token + 1f), scores(0.5f));
    }
    assertThat(bank.selection().occupancy()).isEqualTo(CAPACITY);

    // Two more tokens, both better than everything held.
    bank.stage(keys(99f), values(99f), scores(0.95f));
    bank.stage(keys(98f), values(98f), scores(0.96f));

    assertThat(bank.selection().occupancy()).isEqualTo(CAPACITY);
    assertThat(bank.selection().score(0)).isEqualTo(0.96f);
    assertThat(bank.selection().score(1)).isEqualTo(0.95f);
    // The survivors of the earlier ties are the earliest positions, which is the tie-break rule.
    assertThat(bank.selection().positionsCopy()).containsExactly(5, 4, 0, 1);
  }

  @Test
  void positionsAreDerivedFromTheSequenceNotTheChunk() {
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    for (int token = 0; token < 4; token++) {
      bank.stage(keys(1f), values(1f), scores(1f - token * 0.1f));
    }

    // Four tokens seen, so positions 0..3 -- not 0..1 twice, which a chunk-local counter gives.
    assertThat(bank.selection().positionsCopy()).containsExactly(0, 1, 2, 3);
    assertThat(bank.total()).isEqualTo(4);
  }

  @Test
  void resetClearsTheBankForANewSequence() {
    MemoryBank bank = new MemoryBank(GROUPS, CAPACITY, CHUNK, DIM);
    bank.stage(keys(1f), values(1f), scores(0.9f));
    bank.stage(keys(2f), values(2f), scores(0.5f));

    bank.reset();

    assertThat(bank.selection().occupancy()).isZero();
    assertThat(bank.staged()).isZero();
    assertThat(bank.total()).isZero();
  }

  private static float[][] keys(float fill) {
    return fill(fill);
  }

  private static float[][] values(float fill) {
    return fill(fill);
  }

  private static float[][] fill(float value) {
    float[][] out = new float[GROUPS][DIM];
    for (float[] row : out) {
      java.util.Arrays.fill(row, value);
    }
    return out;
  }

  private static float[] scores(float value) {
    float[] out = new float[GROUPS];
    java.util.Arrays.fill(out, value);
    return out;
  }
}
