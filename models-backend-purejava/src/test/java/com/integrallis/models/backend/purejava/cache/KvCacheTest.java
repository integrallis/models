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

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class KvCacheTest {

  @Nested
  class StoreAndRetrieve {

    @Test
    void storeAndRetrieveSinglePosition() {
      var cache = new KvCache(2, 10, 4);
      float[] key = {1.0f, 2.0f, 3.0f, 4.0f};
      float[] value = {5.0f, 6.0f, 7.0f, 8.0f};

      cache.store(0, 0, key, value);

      assertThat(cache.key(0, 0)).containsExactly(1.0f, 2.0f, 3.0f, 4.0f);
      assertThat(cache.value(0, 0)).containsExactly(5.0f, 6.0f, 7.0f, 8.0f);
    }

    @Test
    void sequentialPositions() {
      var cache = new KvCache(1, 10, 2);
      cache.store(0, 0, new float[] {1, 2}, new float[] {10, 20});
      cache.store(0, 1, new float[] {3, 4}, new float[] {30, 40});
      cache.store(0, 2, new float[] {5, 6}, new float[] {50, 60});

      assertThat(cache.key(0, 1)).containsExactly(3, 4);
      assertThat(cache.value(0, 2)).containsExactly(50, 60);
    }

    @Test
    void storesVectorsFromBatchOffsets() {
      var cache = new KvCache(1, 10, 2, 3);
      float[] keys = {99, 1, 2, 98};
      float[] values = {97, 96, 10, 20, 30, 95};

      cache.store(0, 1, keys, 1, values, 2);

      assertThat(cache.key(0, 1)).containsExactly(1, 2);
      assertThat(cache.value(0, 1)).containsExactly(10, 20, 30);
    }

    @Test
    void storesVectorsInContiguousFlatBuffers() {
      var cache = new KvCache(2, 3, 2);
      cache.store(0, 0, new float[] {1, 2}, new float[] {10, 20});
      cache.store(0, 1, new float[] {3, 4}, new float[] {30, 40});
      cache.store(1, 0, new float[] {5, 6}, new float[] {50, 60});

      assertThat(cache.vectorOffset(0, 0)).isZero();
      assertThat(cache.vectorOffset(0, 1)).isEqualTo(2);
      assertThat(cache.vectorOffset(1, 0)).isEqualTo(6);
      assertThat(cache.keyBuffer()).containsExactly(1, 2, 3, 4, 0, 0, 5, 6, 0, 0, 0, 0);
      assertThat(cache.valueBuffer()).containsExactly(10, 20, 30, 40, 0, 0, 50, 60, 0, 0, 0, 0);
    }

    @Test
    void retrievedVectorsDoNotAliasFlatStorage() {
      var cache = new KvCache(1, 2, 2);
      cache.store(0, 0, new float[] {1, 2}, new float[] {10, 20});

      float[] key = cache.key(0, 0);
      float[] value = cache.value(0, 0);
      key[0] = 99;
      value[0] = 99;

      assertThat(cache.keyBuffer()).startsWith(1, 2);
      assertThat(cache.valueBuffer()).startsWith(10, 20);
    }
  }

  @Nested
  class Slicing {

    @Test
    void sliceCorrectRange() {
      var cache = new KvCache(1, 10, 2);
      cache.store(0, 0, new float[] {1, 2}, new float[] {10, 20});
      cache.store(0, 1, new float[] {3, 4}, new float[] {30, 40});
      cache.store(0, 2, new float[] {5, 6}, new float[] {50, 60});

      float[] keySlice = cache.keySlice(0, 0, 3);
      assertThat(keySlice).containsExactly(1, 2, 3, 4, 5, 6);

      float[] valueSlice = cache.valueSlice(0, 1, 3);
      assertThat(valueSlice).containsExactly(30, 40, 50, 60);
    }
  }

  @Nested
  class MultipleLayer {

    @Test
    void layersAreIndependent() {
      var cache = new KvCache(2, 10, 2);
      cache.store(0, 0, new float[] {1, 2}, new float[] {10, 20});
      cache.store(1, 0, new float[] {3, 4}, new float[] {30, 40});

      assertThat(cache.key(0, 0)).containsExactly(1, 2);
      assertThat(cache.key(1, 0)).containsExactly(3, 4);
    }
  }

  @Nested
  class Clear {

    @Test
    void clearResetsAllEntries() {
      var cache = new KvCache(1, 10, 2);
      cache.store(0, 0, new float[] {1, 2}, new float[] {3, 4});

      cache.clear();

      assertThat(cache.key(0, 0)).isNull();
      assertThat(cache.value(0, 0)).isNull();

      cache.store(0, 0, new float[] {5, 6}, new float[] {7, 8});
      assertThat(cache.key(0, 0)).containsExactly(5, 6);
      assertThat(cache.value(0, 0)).containsExactly(7, 8);
    }
  }

  @Nested
  class Growth {

    @Test
    void largeLogicalContextStartsWithSmallPhysicalCapacity() {
      var cache = new KvCache(2, 10_000, 4);

      assertThat(cache.allocatedSequenceCapacity()).isEqualTo(16);
      assertThat(cache.keyBuffer()).hasSize(2 * 16 * 4);
      assertThat(cache.valueBuffer()).hasSize(2 * 16 * 4);
    }

    @Test
    void growthPreservesEntriesAndLayerIsolation() {
      var cache = new KvCache(2, 40, 2, 3);
      cache.store(0, 0, new float[] {1, 2}, new float[] {10, 20, 30});
      cache.store(1, 0, new float[] {3, 4}, new float[] {40, 50, 60});

      cache.store(0, 20, new float[] {5, 6}, new float[] {70, 80, 90});

      assertThat(cache.allocatedSequenceCapacity()).isEqualTo(32);
      assertThat(cache.key(0, 0)).containsExactly(1, 2);
      assertThat(cache.value(1, 0)).containsExactly(40, 50, 60);
      assertThat(cache.key(0, 20)).containsExactly(5, 6);
      assertThat(cache.keyOffset(1, 0)).isEqualTo(32 * 2);
      assertThat(cache.valueOffset(1, 0)).isEqualTo(32 * 3);
    }
  }

  @Nested
  class BoundsChecking {

    @Test
    void positionBeyondMaxThrows() {
      var cache = new KvCache(1, 5, 2);
      assertThatThrownBy(() -> cache.store(0, 5, new float[] {1, 2}, new float[] {3, 4}))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("position");
    }

    @Test
    void negativeLayerThrows() {
      var cache = new KvCache(2, 10, 2);
      assertThatThrownBy(() -> cache.store(-1, 0, new float[] {1, 2}, new float[] {3, 4}))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void layerBeyondRangeThrows() {
      var cache = new KvCache(2, 10, 2);
      assertThatThrownBy(() -> cache.store(2, 0, new float[] {1, 2}, new float[] {3, 4}))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }
}
