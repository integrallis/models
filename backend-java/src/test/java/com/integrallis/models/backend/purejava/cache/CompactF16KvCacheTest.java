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

import org.junit.jupiter.api.Test;

/**
 * Issue #408 compact F16 KV cache coverage.
 *
 * <p>These tests use the package-visible {@code KvCache(int,int,int,int,boolean)} constructor so
 * they run regardless of the global {@code models.purejava.halfPrecisionKvCache} setting.
 */
class CompactF16KvCacheTest {

  @Test
  void roundTripsThroughF16Rounding() {
    // A float that changes when rounded to F16 and back.
    float original = 1.0001f;
    float rounded = Float.float16ToFloat(Float.floatToFloat16(original));
    assertThat(original).isNotEqualTo(rounded);

    var cache = new KvCache(2, 8, 4, 3, true);
    cache.store(0, 0, new float[] {original, 2.0f, 3.0f, 4.0f}, new float[] {5.0f, 6.0f, 7.0f});

    float[] key = cache.key(0, 0);
    float[] value = cache.value(0, 0);

    assertThat(key[0]).isEqualTo(rounded);
    assertThat(key[1]).isEqualTo(2.0f);
    assertThat(value[0]).isEqualTo(5.0f);
  }

  @Test
  void allocatesHalfTheBytesOfF32() {
    int layers = 4;
    int seqLen = 16;
    int keyDim = 8;
    int valueDim = 6;

    var compact = new KvCache(layers, seqLen, keyDim, valueDim, true);
    var normal = new KvCache(layers, seqLen, keyDim, valueDim, false);

    // Populate one token so ensureCapacity allocates backing stores.
    compact.store(0, 0, new float[keyDim], new float[valueDim]);
    normal.store(0, 0, new float[keyDim], new float[valueDim]);

    long compactBytes = compact.allocatedBytes();
    long normalBytes = normal.allocatedBytes();

    // Occupancy arrays are the same size; keys+values should be roughly half.
    // With 2-element initial capacity from SequenceGrowth.INITIAL_CAPACITY = 2,
    // plus the allocated slot array, the ratio should be very close to 0.5.
    assertThat((double) compactBytes / normalBytes).isBetween(0.45, 0.55);
  }

  @Test
  void freezeAndForkPreserveCompactMode() {
    var cache = new KvCache(2, 8, 4, 3, true);
    for (int layer = 0; layer < 2; layer++) {
      cache.store(layer, 0, new float[] {1.0f, 2.0f, 3.0f, 4.0f}, new float[] {5.0f, 6.0f, 7.0f});
      cache.store(
          layer, 1, new float[] {8.0f, 9.0f, 10.0f, 11.0f}, new float[] {12.0f, 13.0f, 14.0f});
    }

    var prefix = cache.freezePrefix(2);
    var fork = prefix.fork();

    fork.store(0, 2, new float[] {15.0f, 16.0f, 17.0f, 18.0f}, new float[] {19.0f, 20.0f, 21.0f});

    assertThat(fork.key(0, 0)).containsExactly(1.0f, 2.0f, 3.0f, 4.0f);
    assertThat(fork.key(0, 2)).containsExactly(15.0f, 16.0f, 17.0f, 18.0f);
  }

  @Test
  void sliceReadsBackMultiplePositions() {
    var cache = new KvCache(2, 8, 4, 3, true);
    cache.store(0, 0, new float[] {1.0f, 2.0f, 3.0f, 4.0f}, new float[] {5.0f, 6.0f, 7.0f});
    cache.store(0, 1, new float[] {8.0f, 9.0f, 10.0f, 11.0f}, new float[] {12.0f, 13.0f, 14.0f});

    float[] slice = cache.keySlice(0, 0, 2);
    assertThat(slice).containsExactly(1.0f, 2.0f, 3.0f, 4.0f, 8.0f, 9.0f, 10.0f, 11.0f);
  }

  @Test
  void attentionViewBuildsSpans() {
    var cache = new KvCache(1, 8, 4, 3, true);
    cache.store(0, 0, new float[] {1.0f, 2.0f, 3.0f, 4.0f}, new float[] {5.0f, 6.0f, 7.0f});
    cache.store(0, 1, new float[] {8.0f, 9.0f, 10.0f, 11.0f}, new float[] {12.0f, 13.0f, 14.0f});

    var view = cache.attentionView(0, 0, 2);
    assertThat(view.spanCount()).isEqualTo(1);
    assertThat(view.span(0).keyBuffer())
        .containsExactly(1.0f, 2.0f, 3.0f, 4.0f, 8.0f, 9.0f, 10.0f, 11.0f);
  }
}
