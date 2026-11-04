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
package com.integrallis.models.backend.purejava.cache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/**
 * The one growth policy every cache and score buffer here shares.
 *
 * <p>Worth its own test because it replaced eight separate copies of the same arithmetic. The
 * copies were the problem: #357 was four architectures allocating their session's ceiling at open
 * while the two shared caches already grew, and nothing could have caught that divergence while the
 * rule existed in eight places.
 */
class SequenceGrowthTest {

  /** Qwen3.5's declared context, the one that produced #357. */
  private static final int LONG_CONTEXT = 262_144;

  @Test
  void theFirstAllocationIsSmallWhateverTheCeiling() {
    assertThat(SequenceGrowth.nextCapacity(0, 1, LONG_CONTEXT))
        .isEqualTo(SequenceGrowth.INITIAL_CAPACITY);
    // The whole point: a long declaration costs the same at open as a short one.
    assertThat(SequenceGrowth.nextCapacity(0, 1, 1_024_000))
        .isEqualTo(SequenceGrowth.nextCapacity(0, 1, 512));
  }

  @Test
  void aCeilingBelowTheInitialCapacityIsNeverExceeded() {
    assertThat(SequenceGrowth.nextCapacity(0, 1, 4)).isEqualTo(4);
    assertThat(SequenceGrowth.nextCapacity(0, 1, 1)).isEqualTo(1);
  }

  @Test
  void growthDoublesAndThenClampsToTheCeiling() {
    assertThat(SequenceGrowth.nextCapacity(16, 17, LONG_CONTEXT)).isEqualTo(32);
    assertThat(SequenceGrowth.nextCapacity(16, 100, LONG_CONTEXT)).isEqualTo(128);
    // The last step lands exactly on the ceiling rather than overshooting it.
    assertThat(SequenceGrowth.nextCapacity(200_000, 262_000, LONG_CONTEXT)).isEqualTo(LONG_CONTEXT);
    assertThat(SequenceGrowth.nextCapacity(LONG_CONTEXT, LONG_CONTEXT, LONG_CONTEXT))
        .isEqualTo(LONG_CONTEXT);
  }

  @Test
  void whatIsAlreadyHeldIsKeptWhenItSuffices() {
    assertThat(SequenceGrowth.nextCapacity(1024, 1024, LONG_CONTEXT)).isEqualTo(1024);
    assertThat(SequenceGrowth.nextCapacity(1024, 1, LONG_CONTEXT)).isEqualTo(1024);
  }

  @Test
  void advancingOnePositionAtATimeCopiesALinearAmount() {
    // The property doubling is for, asserted rather than asserted-by-comment: reaching the ceiling
    // one position at a time must reallocate a logarithmic number of times, not once per position.
    int allocated = 0;
    int reallocations = 0;
    long copied = 0;
    for (int position = 1; position <= LONG_CONTEXT; position++) {
      int next = SequenceGrowth.nextCapacity(allocated, position, LONG_CONTEXT);
      if (next != allocated) {
        reallocations++;
        copied += allocated;
        allocated = next;
      }
    }
    assertThat(allocated).isEqualTo(LONG_CONTEXT);
    assertThat(reallocations).isLessThan(32);
    // Geometric growth copies under twice the final size in total.
    assertThat(copied).isLessThan(2L * LONG_CONTEXT);
  }

  @Test
  void askingForMoreThanTheCeilingIsRefusedRatherThanClamped() {
    // Clamping would hand back a capacity too small for the caller's write and turn a clear
    // failure into an array index exception somewhere further in.
    assertThatThrownBy(() -> SequenceGrowth.nextCapacity(16, 4097, 4096))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("4096")
        .hasMessageContaining("4097");
  }
}
