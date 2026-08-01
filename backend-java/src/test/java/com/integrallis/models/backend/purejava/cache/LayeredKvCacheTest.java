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

import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LayeredKvCacheTest {

  @Test
  void linearLayerGrowsLazilyAndPreservesAbsoluteHistory() {
    LayeredKvCache cache = new LayeredKvCache(64, LayerSpec.linear(2, 1));

    assertThat(cache.allocatedSequenceCapacity(0)).isZero();

    cache.store(0, 0, new float[] {1, 2}, new float[] {3});
    cache.store(0, 20, new float[] {4, 5}, new float[] {6});

    assertThat(cache.allocatedSequenceCapacity(0)).isGreaterThanOrEqualTo(21);
    assertThat(cache.contains(0, 0)).isTrue();
    assertThat(cache.contains(0, 20)).isTrue();
    assertThat(cache.keyOffset(0, 20)).isEqualTo(40);
    assertThat(cache.valueOffset(0, 20)).isEqualTo(20);
    assertThat(cache.keyBuffer(0)).containsSequence(1, 2);
    assertThat(cache.keyBuffer(0)).containsSequence(4, 5);
    assertThat(cache.oldestRetainedPosition(0)).hasValue(0);
    assertThat(cache.newestRetainedPosition(0)).hasValue(20);
  }

  @Test
  void ringLayerRejectsOverwrittenPositionsAndSplitsAViewAtWraparound() {
    LayeredKvCache cache = new LayeredKvCache(16, LayerSpec.ring(2, 1, 4));
    for (int position = 0; position < 6; position++) {
      cache.store(
          0,
          position,
          new float[] {position * 10.0f, position * 10.0f + 1},
          new float[] {position * 100.0f});
    }

    assertThat(cache.contains(0, 0)).isFalse();
    assertThat(cache.contains(0, 1)).isFalse();
    assertThat(cache.oldestRetainedPosition(0)).hasValue(2);
    assertThat(cache.newestRetainedPosition(0)).hasValue(5);

    LayeredKvCache.AttentionView view = cache.attentionView(0, 2, 6);
    assertThat(view.positionCount()).isEqualTo(4);
    assertThat(view.spanCount()).isEqualTo(2);
    assertThat(view.span(0)).isEqualTo(new LayeredKvCache.AttentionSpan(2, 2, 4, 2));
    assertThat(view.span(1)).isEqualTo(new LayeredKvCache.AttentionSpan(4, 2, 0, 0));

    assertThat(cache.keyBuffer(0)).containsExactly(40, 41, 50, 51, 20, 21, 30, 31);
    assertThat(cache.valueBuffer(0)).containsExactly(400, 500, 200, 300);
    assertThatThrownBy(() -> cache.attentionView(0, 1, 5))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("position 1")
        .hasMessageContaining("overwritten or absent");
  }

  @Test
  void ringCapacityCanRetainAWindowWhileAnEntirePrefillChunkIsWritten() {
    int window = 3;
    int prefillChunk = 2;
    LayeredKvCache cache = new LayeredKvCache(32, LayerSpec.ring(1, 1, window + prefillChunk));

    for (int position = 2; position <= 6; position++) {
      cache.store(0, position, new float[] {position}, new float[] {-position});
    }

    assertThat(cache.attentionView(0, 2, 5).positionCount()).isEqualTo(window);
    assertThat(cache.attentionView(0, 4, 7).positionCount()).isEqualTo(window);
    assertThat(cache.contains(0, 2)).isTrue();
    assertThat(cache.contains(0, 6)).isTrue();
  }

  @Test
  void mixedLayerShapesUseIndependentLinearAndRingStorage() {
    LayeredKvCache cache = new LayeredKvCache(32, LayerSpec.ring(4, 2, 8), LayerSpec.linear(2, 3));

    cache.store(0, 3, sequence(4, 10), sequence(2, 20));
    cache.store(1, 3, sequence(2, 30), sequence(3, 40));

    assertThat(cache.keyDim(0)).isEqualTo(4);
    assertThat(cache.valueDim(0)).isEqualTo(2);
    assertThat(cache.keyDim(1)).isEqualTo(2);
    assertThat(cache.valueDim(1)).isEqualTo(3);
    assertThat(cache.keyBuffer(0)).containsSequence(10, 11, 12, 13);
    assertThat(cache.valueBuffer(1)).containsSequence(40, 41, 42);
    assertThat(cache.allocatedBytes())
        .isEqualTo(
            (8L * (4 + 2) * Float.BYTES + 8L * Integer.BYTES)
                + (16L * (2 + 3) * Float.BYTES + 16L * Integer.BYTES));
  }

  @Test
  void discardFromPreservesTheAcceptedPrefixAcrossLinearAndRingLayers() {
    LayeredKvCache cache = new LayeredKvCache(32, LayerSpec.ring(1, 1, 6), LayerSpec.linear(1, 1));
    for (int position = 0; position < 8; position++) {
      cache.store(0, position, new float[] {position}, new float[] {position});
      cache.store(1, position, new float[] {position}, new float[] {position});
    }

    cache.discardFrom(6);

    assertThat(cache.contains(0, 5)).isTrue();
    assertThat(cache.contains(0, 6)).isFalse();
    assertThat(cache.contains(0, 7)).isFalse();
    assertThat(cache.contains(1, 5)).isTrue();
    assertThat(cache.contains(1, 6)).isFalse();
    assertThat(cache.contains(1, 7)).isFalse();

    cache.store(0, 6, new float[] {60}, new float[] {61});
    cache.store(1, 6, new float[] {60}, new float[] {61});
    assertThat(cache.attentionView(0, 2, 7).positionCount()).isEqualTo(5);
    assertThat(cache.attentionView(1, 0, 7).positionCount()).isEqualTo(7);

    cache.clear();
    assertThat(cache.oldestRetainedPosition(0)).isEmpty();
    assertThat(cache.oldestRetainedPosition(1)).isEmpty();
  }

  @Test
  void validatesCoordinatesVectorRangesAndAttentionRanges() {
    LayeredKvCache cache = new LayeredKvCache(8, LayerSpec.ring(2, 1, 4));

    assertThatThrownBy(() -> cache.store(0, 0, new float[] {1}, new float[] {2}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("key range");
    assertThatThrownBy(() -> cache.store(1, 0, new float[] {1, 2}, new float[] {3}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("layer out of range");
    assertThatThrownBy(() -> cache.store(0, 8, new float[] {1, 2}, new float[] {3}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("position out of range");
    assertThatThrownBy(() -> cache.attentionView(0, 0, 5))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("exceeds ring capacity 4");
  }

  private static float[] sequence(int size, int start) {
    float[] result = new float[size];
    for (int index = 0; index < size; index++) {
      result[index] = start + index;
    }
    return result;
  }
}
