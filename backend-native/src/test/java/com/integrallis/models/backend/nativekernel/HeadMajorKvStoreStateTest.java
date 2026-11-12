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

import org.junit.jupiter.api.Test;

class HeadMajorKvStoreStateTest {

  @Test
  void restorePreservesIdentityAndPositions() {
    int heads = 3, width = 16, ceiling = 64, positions = 8, stride = 16;
    short[] keys = new short[stride * heads * width];
    short[] values = new short[stride * heads * width];
    for (int i = 0; i < keys.length; i++) {
      keys[i] = (short) (i + 1);
      values[i] = (short) -(i + 1);
    }

    HeadMajorKvStore store =
        HeadMajorKvStore.restore(heads, width, ceiling, positions, stride, keys, values);

    assertThat(store.heads()).isEqualTo(heads);
    assertThat(store.dimension()).isEqualTo(heads * width);
    assertThat(store.ceiling()).isEqualTo(ceiling);
    assertThat(store.stride()).isEqualTo(stride);
    assertThat(store.keys()).containsExactly(keys);
    assertThat(store.values()).containsExactly(values);
    // allocatedBytes includes the whole stride, not just used positions
    assertThat(store.allocatedBytes()).isEqualTo(keys.length * 2L + values.length * 2L);
  }

  @Test
  void restoreValidatesShape() {
    short[] keys = new short[10];
    short[] values = new short[10];
    assertThatThrownBy(() -> HeadMajorKvStore.restore(2, 4, 8, 10, 0, keys, values))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("stride < positions");
  }

  @Test
  void restoreRejectsLengthMismatch() {
    short[] keys = new short[10];
    short[] values = new short[12];
    assertThatThrownBy(() -> HeadMajorKvStore.restore(2, 4, 8, 4, 4, keys, values))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("length mismatch");
  }
}
