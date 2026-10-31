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
package com.integrallis.models.backend.purejava.qwen35;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The attended span under a sliding window.
 *
 * <p>EdgeRock's eight attention layers are windowed at 1024 where the three older Qwen3.5
 * architectures attend to everything. The window is applied as a base offset into a key cache that
 * is contiguous in position, so an error here does not throw or produce a NaN -- it attends to the
 * wrong tokens and returns a fluent answer.
 */
@Tag("unit")
class Qwen35SlidingWindowSpanTest {

  @Test
  void anUnwindowedModelAlwaysSeesThePrefixFromZero() {
    assertThat(Qwen35ForwardPass.firstVisiblePosition(0, 0)).isZero();
    assertThat(Qwen35ForwardPass.firstVisiblePosition(5_000, 0)).isZero();
  }

  @Test
  void aSequenceShorterThanTheWindowIsNotTrimmed() {
    // Up to and including the position where the span exactly fills the window, nothing is
    // dropped. This is the boundary an off-by-one would cross first.
    assertThat(Qwen35ForwardPass.firstVisiblePosition(0, 1_024)).isZero();
    assertThat(Qwen35ForwardPass.firstVisiblePosition(1_022, 1_024)).isZero();
    assertThat(Qwen35ForwardPass.firstVisiblePosition(1_023, 1_024)).isZero();
  }

  @Test
  void theSpanCoversExactlyTheWindowOnceTheSequenceIsLonger() {
    assertThat(Qwen35ForwardPass.firstVisiblePosition(1_024, 1_024)).isEqualTo(1);
    assertThat(Qwen35ForwardPass.firstVisiblePosition(4_095, 1_024)).isEqualTo(3_072);

    // The span is inclusive at both ends, so its width is the window and not one either side.
    for (int position : new int[] {1_023, 1_024, 2_000, 262_143}) {
      int first = Qwen35ForwardPass.firstVisiblePosition(position, 1_024);
      int width = position - first + 1;
      assertThat(width).isEqualTo(Math.min(position + 1, 1_024));
    }
  }

  @Test
  void aWindowOfOneSeesOnlyTheCurrentToken() {
    assertThat(Qwen35ForwardPass.firstVisiblePosition(7, 1)).isEqualTo(7);
  }

  @Test
  void nonsenseArgumentsAreRefusedRatherThanProducingANegativeOffset() {
    // A negative first position becomes a negative array offset, which would either throw far
    // from the cause or read another layer's cache.
    assertThatThrownBy(() -> Qwen35ForwardPass.firstVisiblePosition(-1, 1_024))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Qwen35ForwardPass.firstVisiblePosition(5, -1))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
