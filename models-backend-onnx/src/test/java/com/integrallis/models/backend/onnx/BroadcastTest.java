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
package com.integrallis.models.backend.onnx;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BroadcastTest {

  @Test
  void computesBroadcastShapesAndRejectsMismatches() {
    assertThat(Broadcast.shape(new long[] {2, 1, 3}, new long[] {4, 1})).containsExactly(2, 4, 3);
    assertThat(Broadcast.shape(new long[0], new long[] {5})).containsExactly(5);
    assertThat(Broadcast.same(new long[] {1}, new long[] {1})).isTrue();
    assertThatThrownBy(() -> Broadcast.shape(new long[] {2}, new long[] {3}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void mapsOutputIndicesBackToBroadcastInputs() {
    assertThat(Broadcast.indexMap(new long[] {3, 1}, new long[] {3, 2}))
        .containsExactly(0, 0, 1, 1, 2, 2);
    assertThat(Broadcast.indexMap(new long[] {2}, new long[] {3, 2}))
        .containsExactly(0, 1, 0, 1, 0, 1);
    assertThat(Broadcast.indexMap(new long[0], new long[] {2, 2})).containsExactly(0, 0, 0, 0);
    assertThat(Broadcast.strides(new long[] {1, 2}, new long[] {3, 1, 2})).containsExactly(0, 2, 1);
    assertThat(Broadcast.strides(new long[] {1, 2}, new long[] {3, 4, 2})).containsExactly(0, 0, 1);
  }
}
