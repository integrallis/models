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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.cache.SequenceGrowth;
import org.junit.jupiter.api.Test;

/**
 * The retained-KV store's two obligations: never hold the declared context before a sequence
 * reaches it, and keep every stored row addressable after the stride that addresses it has moved.
 */
class HeadMajorKvStoreTest {

  private static final int HEADS = 3;
  private static final int WIDTH = 4;

  /** A value that identifies its own head, position and lane, so a misplaced row is visible. */
  private static float marker(int head, int position, int lane) {
    return head * 100_000 + position * 100 + lane;
  }

  private static void storeOne(HeadMajorKvStore store, int position) {
    float[] key = new float[HEADS * WIDTH];
    float[] value = new float[HEADS * WIDTH];
    for (int head = 0; head < HEADS; head++) {
      for (int lane = 0; lane < WIDTH; lane++) {
        key[head * WIDTH + lane] = marker(head, position, lane);
        value[head * WIDTH + lane] = -marker(head, position, lane);
      }
    }
    store.store(key, value, position, 1);
  }

  private static void assertRowsIntact(HeadMajorKvStore store, int through) {
    for (int position = 0; position <= through; position++) {
      for (int head = 0; head < HEADS; head++) {
        for (int lane = 0; lane < WIDTH; lane++) {
          int at = (head * store.stride() + position) * WIDTH + lane;
          assertThat(store.keys()[at])
              .describedAs("key head %s position %s lane %s", head, position, lane)
              .isEqualTo(marker(head, position, lane));
          assertThat(store.values()[at])
              .describedAs("value head %s position %s lane %s", head, position, lane)
              .isEqualTo(-marker(head, position, lane));
        }
      }
    }
  }

  @Test
  void holdsTheInitialCapacityRatherThanTheDeclaredContext() {
    HeadMajorKvStore store = new HeadMajorKvStore(32, 128, 4096);

    storeOne32x128(store);

    assertThat(store.stride()).isEqualTo(SequenceGrowth.INITIAL_CAPACITY);
    // What the preallocating form held for one position, in bytes: 4096 positions of both buffers.
    long declared = 2L * 32 * 128 * 4096 * Float.BYTES;
    assertThat(store.allocatedBytes()).isEqualTo(2L * 32 * 128 * 16 * Float.BYTES);
    assertThat(store.allocatedBytes()).isLessThan(declared / 100);
  }

  private static void storeOne32x128(HeadMajorKvStore store) {
    float[] rows = new float[32 * 128];
    store.store(rows, rows, 0, 1);
  }

  @Test
  void growthKeepsEveryStoredRowAtItsNewStride() {
    HeadMajorKvStore store = new HeadMajorKvStore(HEADS, WIDTH, 100);

    for (int position = 0; position < 100; position++) {
      storeOne(store, position);
      // Checked after every single step: a restride that drops or shifts a row shows up at the
      // boundary it happened on, not 80 positions later.
      assertRowsIntact(store, position);
      assertThat(store.positions()).isEqualTo(position + 1);
      assertThat(store.stride()).isGreaterThanOrEqualTo(position + 1);
    }
  }

  @Test
  void growthStopsAtTheCeiling() {
    HeadMajorKvStore store = new HeadMajorKvStore(HEADS, WIDTH, 100);

    for (int position = 0; position < 100; position++) {
      storeOne(store, position);
      assertThat(store.stride()).isLessThanOrEqualTo(100);
    }

    assertThat(store.stride()).isEqualTo(100);
    assertThatThrownBy(() -> storeOne(store, 100))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("exceed");
  }

  @Test
  void aBatchWiderThanTheCurrentStrideGrowsOnce() {
    HeadMajorKvStore store = new HeadMajorKvStore(HEADS, WIDTH, 4096);
    float[] rows = new float[40 * HEADS * WIDTH];

    store.store(rows, rows, 0, 40);

    assertThat(store.positions()).isEqualTo(40);
    assertThat(store.stride()).isEqualTo(SequenceGrowth.nextCapacity(0, 40, 4096));
  }

  @Test
  void rewindKeepsAllocationAndForgetsPositions() {
    HeadMajorKvStore store = new HeadMajorKvStore(HEADS, WIDTH, 100);
    for (int position = 0; position < 40; position++) {
      storeOne(store, position);
    }
    long held = store.allocatedBytes();

    store.rewind(10);

    assertThat(store.positions()).isEqualTo(10);
    assertThat(store.allocatedBytes()).isEqualTo(held);
    assertRowsIntact(store, 9);
  }
}
