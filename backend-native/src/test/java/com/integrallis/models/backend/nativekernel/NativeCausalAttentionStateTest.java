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

import org.junit.jupiter.api.Test;

/**
 * What the retained attention cache holds, and that appending across a growth boundary stays
 * continuous.
 *
 * <p>{@code append} stores rows without calling either native kernel, so these run without the
 * shim: the question here is allocation and continuity, not arithmetic. Parity of the attention
 * output itself is {@code NativeAttentionParityTest}'s job.
 */
class NativeCausalAttentionStateTest {

  // LLaVA-1.5-7B: 32 layers, 32 heads, 4096 embedding, declared context 4096.
  private static final int LAYERS = 32;
  private static final int HEADS = 32;
  private static final int WIDTH = 128;
  private static final int DECLARED_CONTEXT = 4096;

  private static NativeCausalAttention attention() {
    return new NativeCausalAttention(null, null);
  }

  private static boolean appendOne(NativeCausalAttention attention, int layer, int position) {
    float[] rows = new float[HEADS * WIDTH];
    return attention.append(
        rows, rows, layer, position, 1, HEADS, HEADS, WIDTH, WIDTH, DECLARED_CONTEXT, 0);
  }

  @Test
  void aShortSequenceDoesNotHoldTheDeclaredContext() {
    NativeCausalAttention attention = attention();

    for (int layer = 0; layer < LAYERS; layer++) {
      assertThat(appendOne(attention, layer, 0)).isTrue();
    }

    long declared = 2L * LAYERS * HEADS * WIDTH * DECLARED_CONTEXT * Float.BYTES;
    assertThat(declared).isEqualTo(4L * 1024 * 1024 * 1024); // 4 GiB, what it held before
    assertThat(attention.allocatedStateBytes()).isEqualTo(declared / (DECLARED_CONTEXT / 16));
  }

  @Test
  void appendStaysEligibleAndGrowsAcrossTheBoundary() {
    NativeCausalAttention attention = attention();

    for (int position = 0; position < 200; position++) {
      assertThat(appendOne(attention, 0, position))
          .describedAs("eligible at position %s", position)
          .isTrue();
    }

    // One layer, 200 positions: doubling from 16 reaches 256, and nothing beyond it is held.
    assertThat(attention.allocatedStateBytes()).isEqualTo(2L * HEADS * WIDTH * 256 * Float.BYTES);
  }

  @Test
  void aDiscontinuousAppendIsStillRefused() {
    NativeCausalAttention attention = attention();

    assertThat(appendOne(attention, 0, 0)).isTrue();
    assertThat(appendOne(attention, 0, 5)).isFalse();
  }

  @Test
  void resetForgetsPositionsAndRewindTruncatesThem() {
    NativeCausalAttention attention = attention();
    for (int position = 0; position < 40; position++) {
      appendOne(attention, 0, position);
    }
    long held = attention.allocatedStateBytes();

    attention.rewind(10);
    assertThat(appendOne(attention, 0, 10)).isTrue();

    attention.reset();
    assertThat(appendOne(attention, 0, 0)).isTrue();
    assertThat(attention.allocatedStateBytes()).isEqualTo(held);
  }
}
