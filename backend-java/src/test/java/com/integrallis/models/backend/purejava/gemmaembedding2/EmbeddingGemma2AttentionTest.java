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
package com.integrallis.models.backend.purejava.gemmaembedding2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class EmbeddingGemma2AttentionTest {
  @Test
  void symmetricWindowIncludesBothBoundariesAndFutureTokens() {
    // Equal logits turn attention into an arithmetic mean. Values outside either endpoint
    // are deliberately large, so an off-by-one or a causal mask cannot accidentally agree.
    float[] out = new float[7];
    EmbeddingGemma2Encoder.attend(
        new float[7],
        new float[7],
        new float[] {1000, 1, 2, 3, 4, 5, 2000},
        out,
        new float[7],
        7,
        1,
        1,
        1,
        4);
    assertThat(out[3]).isCloseTo(3, offset(1e-6f));
    assertThat(out[0]).isCloseTo(1003f / 3, offset(1e-4f));
    assertThat(out[6]).isCloseTo(2009f / 3, offset(1e-4f));
  }

  @Test
  void fullAttentionSeesTheWholeSequenceAndGroupsQueryHeads() {
    float[] out = new float[8];
    EmbeddingGemma2Encoder.attend(
        new float[8], new float[4], new float[] {2, 10, 4, 30}, out, new float[2], 2, 4, 2, 1, 0);
    assertThat(out).containsExactly(3, 3, 20, 20, 3, 3, 20, 20);
  }

  @Test
  void learnedQueryScaleIsNotDividedByTheSquareRootOfHeadWidth() {
    float[] out = new float[4];
    EmbeddingGemma2Encoder.attend(
        new float[] {1, 0, 1, 0},
        new float[] {0, 0, 1, 0},
        new float[] {0, 0, 1, 2},
        out,
        new float[2],
        2,
        1,
        1,
        2,
        0);
    assertThat(out[0]).isCloseTo((float) (Math.E / (1 + Math.E)), offset(1e-6f));
    assertThat(out[1]).isCloseTo(2 * out[0], offset(1e-6f));
  }
}
