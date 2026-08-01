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
package com.integrallis.models.backend.purejava.gemma4;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Gemma4KvCacheTest {

  @Test
  void mapsSlidingAndFullAttentionLayersToTheirExactDimensionsAndRetention() {
    Gemma4Config config = config();

    LayeredKvCache cache = Gemma4KvCache.create(config, 8_192, 128);

    assertThat(cache.numLayers()).isEqualTo(30);
    assertThat(cache.maxSeqLen()).isEqualTo(8_192);
    assertThat(cache.keyDim(0)).isEqualTo(2_048);
    assertThat(cache.valueDim(0)).isEqualTo(2_048);
    assertThat(cache.physicalSequenceCapacity(0)).isEqualTo(1_152);
    assertThat(cache.keyDim(5)).isEqualTo(1_024);
    assertThat(cache.valueDim(5)).isEqualTo(1_024);
    assertThat(cache.physicalSequenceCapacity(5)).isEqualTo(8_192);
    assertThat(cache.allocatedBytes()).isZero();
    assertThat(cache.maximumBytes()).isEqualTo(807_682_560L);
    assertThat(cache.maximumBytes()).isLessThan(3_691_970_560L / 4);
  }

  @Test
  void capsSlidingStorageWhenTheRuntimeContextIsShorterThanTheWindowAndChunk() {
    LayeredKvCache cache = Gemma4KvCache.create(config(), 512, 128);

    assertThat(cache.physicalSequenceCapacity(0)).isEqualTo(512);
    assertThat(cache.physicalSequenceCapacity(5)).isEqualTo(512);
  }

  @Test
  void rejectsInvalidRuntimeContextAndPrefillCapacity() {
    assertThatThrownBy(() -> Gemma4KvCache.create(config(), 0, 128))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("runtimeContextLength must be > 0");
    assertThatThrownBy(() -> Gemma4KvCache.create(config(), 32_769, 128))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("exceeds model context length 32768");
    assertThatThrownBy(() -> Gemma4KvCache.create(config(), 8_192, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("prefillCapacity must be > 0");
  }

  private static Gemma4Config config() {
    return new Gemma4Config(
        2_816,
        30,
        16,
        IntStream.range(0, 30).mapToObj(layer -> isFull(layer) ? 2 : 8).toList(),
        512,
        256,
        512,
        256,
        262_144,
        32_768,
        2_112,
        704,
        128,
        8,
        1_000_000.0f,
        10_000.0f,
        512,
        256,
        1.0e-6f,
        1_024,
        IntStream.range(0, 30).mapToObj(layer -> !isFull(layer)).toList(),
        30.0f);
  }

  private static boolean isFull(int layer) {
    return (layer + 1) % 6 == 0;
  }
}
