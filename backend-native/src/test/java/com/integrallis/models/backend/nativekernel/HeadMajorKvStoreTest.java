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

  /**
   * A value identifying its own head, position and lane, so a misplaced row is visible.
   *
   * <p>Kept under 2048, where every integer is exactly representable in F16, so a marker survives
   * the rounding the store's caller applies and a shifted row cannot be mistaken for a rounded one.
   */
  private static float marker(int head, int position, int lane) {
    return head * 600 + position * 6 + lane;
  }

  /**
   * Writes one position exactly as NativeCausalAttention.storeRows does, by the returned stride.
   */
  private static void storeOne(HeadMajorKvStore store, int position) {
    int stride = store.ensureCapacity(position + 1);
    for (int head = 0; head < HEADS; head++) {
      int destination = (head * stride + position) * WIDTH;
      for (int lane = 0; lane < WIDTH; lane++) {
        store.keys()[destination + lane] = Float.floatToFloat16(marker(head, position, lane));
        store.values()[destination + lane] = Float.floatToFloat16(-marker(head, position, lane));
      }
    }
    store.positions = position + 1;
  }

  private static void assertRowsIntact(HeadMajorKvStore store, int through) {
    for (int position = 0; position <= through; position++) {
      for (int head = 0; head < HEADS; head++) {
        for (int lane = 0; lane < WIDTH; lane++) {
          int at = (head * store.stride() + position) * WIDTH + lane;
          assertThat(Float.float16ToFloat(store.keys()[at]))
              .describedAs("key head %s position %s lane %s", head, position, lane)
              .isEqualTo(marker(head, position, lane));
          assertThat(Float.float16ToFloat(store.values()[at]))
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

    long declared = 2L * 32 * 128 * 4096 * Short.BYTES;
    assertThat(declared).isEqualTo(64L * 1024 * 1024); // 64 MiB per layer, 2 GiB over 32 layers
    assertThat(store.allocatedBytes()).isEqualTo(2L * 32 * 128 * 16 * Short.BYTES);
    assertThat(store.allocatedBytes()).isLessThan(declared / 100);
  }

  @Test
  void holdsF16BitsRatherThanWidenedValues() {
    HeadMajorKvStore store = new HeadMajorKvStore(1, 4, 16);
    int stride = store.ensureCapacity(1);
    assertThat(stride).isEqualTo(SequenceGrowth.INITIAL_CAPACITY);

    // 0.1 has no exact F16 representation, so it shows what the store keeps and what it discards.
    float rounded = Float.float16ToFloat(Float.floatToFloat16(0.1f));
    store.keys()[0] = Float.floatToFloat16(0.1f);

    assertThat(store.keys()[0]).isEqualTo(Float.floatToFloat16(rounded));
    assertThat(Float.float16ToFloat(store.keys()[0])).isEqualTo(rounded).isNotEqualTo(0.1f);
    // Two bytes an element, which is the whole point of storing the bits.
    assertThat(store.allocatedBytes()).isEqualTo(2L * 4 * stride * Short.BYTES);
    assertThat(store.allocatedBytes()).isEqualTo(2L * 4 * stride * Float.BYTES / 2);
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
  void javaRoundingIsClosedOverEveryF16Encoding() {
    // The store's filling loop writes Float.floatToFloat16 bits and the native kernel widens them.
    // That makes Java and the kernel agree on an encoding rather than on a value, so Java's half of
    // the boundary is pinned here: re-rounding a widened value must return the encoding it came
    // from, for all 65536 of them. The kernel's half, including that cvtph_ps and the scalar
    // widening agree on every non-NaN encoding, is asserted in the Rust crate's
    // every_f16_encoding_widens_the_same_way_through_both_paths.
    int finite = 0;
    for (int encoding = 0; encoding <= 0xFFFF; encoding++) {
      float widened = Float.float16ToFloat((short) encoding);
      if (Float.isNaN(widened)) {
        continue;
      }
      finite++;
      assertThat(Float.floatToFloat16(widened))
          .describedAs("rounding is not closed on encoding 0x%04X", encoding)
          .isEqualTo((short) encoding);
    }
    assertThat(finite).isEqualTo(63490);
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
