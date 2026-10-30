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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SlidingWindowAttentionTest {

  private static final int DIM = 4;
  private static final float TOLERANCE = 1.0e-5f;

  @Test
  void aSingleVisibleKeyTakesAllTheWeightUnlikeTheMemoryRead() {
    // The asymmetry worth pinning. Both layers softmax; only the memory read appends a zero sink.
    // Same setup there reads 0.5; here it reads 1.0.
    SlidingWindowAttention.Cache cache = new SlidingWindowAttention.Cache(1, 4, DIM);
    cache.append(group(new float[DIM]), group(fill(7f)), 0);
    float[][] out = {new float[DIM]};

    SlidingWindowAttention.read(cache, new float[][] {new float[DIM]}, 0, 1, out);

    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(7f);
  }

  @Test
  void theWindowSpansExactlyWindowKeysIncludingTheQuery() {
    int window = 3;
    SlidingWindowAttention.Cache cache = new SlidingWindowAttention.Cache(1, window, DIM);
    // Positions 0..2 with distinct values; a zero query makes every visible key tie.
    cache.append(group(new float[DIM]), group(fill(1f)), 0);
    cache.append(group(new float[DIM]), group(fill(2f)), 1);
    cache.append(group(new float[DIM]), group(fill(3f)), 2);
    float[][] out = {new float[DIM]};

    SlidingWindowAttention.read(cache, new float[][] {new float[DIM]}, 2, 1, out);

    // All three visible and tied: (1 + 2 + 3) / 3 = 2.
    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(2f);
  }

  @Test
  void theKeyOneStepOutsideTheWindowIsExcluded() {
    // The off-by-one. Window 2 at position 2 admits positions 1 and 2 only; position 0 is out,
    // even though the ring buffer still holds it until overwritten.
    int window = 2;
    SlidingWindowAttention.Cache cache = new SlidingWindowAttention.Cache(1, window, DIM);
    cache.append(group(new float[DIM]), group(fill(100f)), 0);
    cache.append(group(new float[DIM]), group(fill(2f)), 1);
    float[][] out = {new float[DIM]};

    // Position 2 has not been appended, so only position 1 is visible: weight 1.0 on value 2.
    SlidingWindowAttention.read(cache, new float[][] {new float[DIM]}, 2, 1, out);

    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(2f);
  }

  @Test
  void futureKeysAreNeverVisible() {
    // Causality. The ring buffer may hold a later position after wrapping; it must not be read.
    SlidingWindowAttention.Cache cache = new SlidingWindowAttention.Cache(1, 4, DIM);
    cache.append(group(new float[DIM]), group(fill(1f)), 0);
    cache.append(group(new float[DIM]), group(fill(99f)), 3);
    float[][] out = {new float[DIM]};

    SlidingWindowAttention.read(cache, new float[][] {new float[DIM]}, 1, 1, out);

    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(1f);
  }

  @Test
  void theRingBufferEvictsExactlyWindowPositionsBack() {
    int window = 2;
    SlidingWindowAttention.Cache cache = new SlidingWindowAttention.Cache(1, window, DIM);
    cache.append(group(new float[DIM]), group(fill(1f)), 0);
    cache.append(group(new float[DIM]), group(fill(2f)), 1);
    // Position 2 lands in slot 0, overwriting position 0 -- which is also the first position the
    // window excludes, so eviction and masking agree rather than one covering for the other.
    cache.append(group(new float[DIM]), group(fill(3f)), 2);

    assertThat(cache.positionAt(0)).isEqualTo(2);
    assertThat(cache.positionAt(1)).isEqualTo(1);

    float[][] out = {new float[DIM]};
    SlidingWindowAttention.read(cache, new float[][] {new float[DIM]}, 2, 1, out);

    // Positions 1 and 2 tie: (2 + 3) / 2 = 2.5.
    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(2.5f);
  }

  @Test
  void weightsSumToOneSoTiedKeysAverage() {
    SlidingWindowAttention.Cache cache = new SlidingWindowAttention.Cache(1, 4, DIM);
    cache.append(group(new float[DIM]), group(fill(4f)), 0);
    cache.append(group(new float[DIM]), group(fill(8f)), 1);
    float[][] out = {new float[DIM]};

    SlidingWindowAttention.read(cache, new float[][] {new float[DIM]}, 1, 1, out);

    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(6f);
  }

  @Test
  void headsInAGroupShareKeysAndValues() {
    SlidingWindowAttention.Cache cache = new SlidingWindowAttention.Cache(1, 4, DIM);
    cache.append(group(new float[DIM]), group(fill(5f)), 0);
    float[][] out = {new float[DIM], new float[DIM]};

    SlidingWindowAttention.read(cache, new float[][] {new float[DIM], new float[DIM]}, 0, 2, out);

    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(5f);
    assertThat(out[1]).usingComparatorWithPrecision(TOLERANCE).containsOnly(5f);
  }

  @Test
  void readingBeforeAppendingIsACallerBug() {
    SlidingWindowAttention.Cache cache = new SlidingWindowAttention.Cache(1, 4, DIM);

    assertThatThrownBy(
            () ->
                SlidingWindowAttention.read(
                    cache, new float[][] {new float[DIM]}, 0, 1, new float[][] {new float[DIM]}))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("append before reading");
  }

  private static float[][] group(float[] row) {
    return new float[][] {row};
  }

  private static float[] fill(float value) {
    float[] out = new float[DIM];
    java.util.Arrays.fill(out, value);
    return out;
  }
}
