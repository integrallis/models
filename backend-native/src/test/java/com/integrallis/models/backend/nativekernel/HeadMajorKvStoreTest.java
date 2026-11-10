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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The retained-KV store's two obligations: never hold the declared context before a sequence
 * reaches it, and keep every stored row addressable after the stride that addresses it has moved.
 */
@Tag("unit")
class HeadMajorKvStoreTest {

  private static final int HEADS = 3;
  private static final int WIDTH = 4;

  /** A value identifying its own head, position and lane, so a misplaced row is visible. */
  private static float marker(int head, int position, int lane) {
    return head * 100_000 + position * 100 + lane;
  }

  /**
   * Writes one position exactly as NativeCausalAttention.storeRows does, by the returned stride.
   */
  private static void storeOne(HeadMajorKvStore store, int position) {
    int stride = store.ensureCapacity(position + 1);
    for (int head = 0; head < HEADS; head++) {
      int destination = (head * stride + position) * WIDTH;
      for (int lane = 0; lane < WIDTH; lane++) {
        store.keys()[destination + lane] = marker(head, position, lane);
        store.values()[destination + lane] = -marker(head, position, lane);
      }
    }
    store.positions = position + 1;
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
    // LLaVA 1.5 7B geometry: 32 heads of 128, declared context 4096.
    HeadMajorKvStore store = new HeadMajorKvStore(32, 128, 4096);

    assertThat(store.ensureCapacity(1)).isEqualTo(SequenceGrowth.INITIAL_CAPACITY);

    long declared = 2L * 32 * 128 * 4096 * Float.BYTES;
    assertThat(declared).isEqualTo(128L * 1024 * 1024); // 128 MiB per layer, 4 GiB over 32 layers
    assertThat(store.allocatedBytes()).isEqualTo(2L * 32 * 128 * 16 * Float.BYTES);
    assertThat(store.allocatedBytes()).isLessThan(declared / 100);
  }

  @Test
  void growthKeepsEveryStoredRowAtItsNewStride() {
    HeadMajorKvStore store = new HeadMajorKvStore(HEADS, WIDTH, 100);

    for (int position = 0; position < 100; position++) {
      storeOne(store, position);
      // Checked after every step: a restride that drops or shifts a row shows up at the boundary
      // it happened on rather than eighty positions later.
      assertRowsIntact(store, position);
      assertThat(store.positions).isEqualTo(position + 1);
      assertThat(store.stride()).isGreaterThanOrEqualTo(position + 1);
    }
    assertThat(store.stride()).isEqualTo(100);
  }

  @Test
  void growthStopsAtTheCeiling() {
    HeadMajorKvStore store = new HeadMajorKvStore(HEADS, WIDTH, 100);
    for (int position = 0; position < 100; position++) {
      storeOne(store, position);
      assertThat(store.stride()).isLessThanOrEqualTo(100);
    }

    assertThatThrownBy(() -> store.ensureCapacity(101))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("exceed");
  }

  @Test
  void aBatchWiderThanTheCurrentStrideGrowsOnce() {
    HeadMajorKvStore store = new HeadMajorKvStore(HEADS, WIDTH, 4096);

    assertThat(store.ensureCapacity(40)).isEqualTo(SequenceGrowth.nextCapacity(0, 40, 4096));
  }

  @Test
  void rejectsDegenerateGeometry() {
    assertThatThrownBy(() -> new HeadMajorKvStore(0, 4, 16))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("heads");
    assertThatThrownBy(() -> new HeadMajorKvStore(3, 0, 16))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("width");
    assertThatThrownBy(() -> new HeadMajorKvStore(3, 4, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("ceiling");
  }
}
