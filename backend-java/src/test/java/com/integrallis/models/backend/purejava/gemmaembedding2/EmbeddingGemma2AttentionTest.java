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
    attend(new float[7], new float[7], new float[] {1000, 1, 2, 3, 4, 5, 2000}, out, 7, 1, 1, 1, 4);
    assertThat(out[3]).isCloseTo(3, offset(1e-6f));
    assertThat(out[0]).isCloseTo(1003f / 3, offset(1e-4f));
    assertThat(out[6]).isCloseTo(2009f / 3, offset(1e-4f));
  }

  @Test
  void fullAttentionSeesTheWholeSequenceAndGroupsQueryHeads() {
    float[] out = new float[8];
    attend(new float[8], new float[4], new float[] {2, 10, 4, 30}, out, 2, 4, 2, 1, 0);
    assertThat(out).containsExactly(3, 3, 20, 20, 3, 3, 20, 20);
  }

  @Test
  void learnedQueryScaleIsNotDividedByTheSquareRootOfHeadWidth() {
    float[] out = new float[4];
    attend(
        new float[] {1, 0, 1, 0},
        new float[] {0, 0, 1, 0},
        new float[] {0, 0, 1, 2},
        out,
        2,
        1,
        1,
        2,
        0);
    assertThat(out[0]).isCloseTo((float) (Math.E / (1 + Math.E)), offset(1e-6f));
    assertThat(out[1]).isCloseTo(2 * out[0], offset(1e-6f));
  }

  @Test
  void headPartitioningKeepsEveryBitAcrossShapesAndReusedScratch() {
    int heads = 4;
    var attention = new EmbeddingGemma2Attention(heads);
    var random = new java.util.Random(390);
    for (int count : new int[] {3, 127, 128, 7, 129, 1}) {
      for (int kvHeads : new int[] {1, 2, 4}) {
        for (int width : new int[] {16, 32}) {
          for (int window : new int[] {0, 4, 9}) {
            float[] q = randomValues(random, count * heads * width);
            float[] k = randomValues(random, count * kvHeads * width);
            float[] v = randomValues(random, k.length);
            float[] expected = new float[q.length + 3];
            float[] actual = new float[expected.length];
            java.util.Arrays.fill(expected, 17.0f);
            java.util.Arrays.fill(actual, 17.0f);
            reference(q, k, v, expected, new float[count], count, heads, kvHeads, width, window);
            attention.compute(q, k, v, actual, count, kvHeads, width, window);
            assertThat(actual).containsExactly(expected);
            // Reusing the output and score storage must not accumulate a previous sequence.
            attention.compute(q, k, v, actual, count, kvHeads, width, window);
            assertThat(actual).containsExactly(expected);
          }
        }
      }
    }
  }

  private static float[] randomValues(java.util.Random random, int count) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) values[i] = random.nextFloat() * 2 - 1;
    return values;
  }

  private static void attend(
      float[] q,
      float[] k,
      float[] v,
      float[] out,
      int count,
      int heads,
      int kvHeads,
      int width,
      int window) {
    new EmbeddingGemma2Attention(heads).compute(q, k, v, out, count, kvHeads, width, window);
  }

  /** Attention uses scale 1: the learned query norm already supplies the scale. */
  private static void reference(
      float[] queries,
      float[] keys,
      float[] values,
      float[] out,
      float[] scores,
      int count,
      int heads,
      int kvHeads,
      int headDim,
      int window) {
    int queryDim = heads * headDim;
    int keyDim = kvHeads * headDim;
    java.util.Arrays.fill(out, 0, count * queryDim, 0);
    for (int position = 0; position < count; position++) {
      int start = window == 0 ? 0 : Math.max(0, position - window / 2);
      int end = window == 0 ? count : Math.min(count, position + window / 2 + 1);
      for (int head = 0; head < heads; head++) {
        int kvHead = head / (heads / kvHeads);
        int queryOffset = position * queryDim + head * headDim;
        for (int keyPosition = start; keyPosition < end; keyPosition++) {
          scores[keyPosition] =
              com.integrallis.models.backend.purejava.ops.PinnedReduction.dot(
                  queries, queryOffset, keys, keyPosition * keyDim + kvHead * headDim, headDim);
        }
        com.integrallis.models.backend.purejava.ops.TensorOps.softmax(scores, start, end - start);
        for (int keyPosition = start; keyPosition < end; keyPosition++) {
          float weight = scores[keyPosition];
          int valueOffset = keyPosition * keyDim + kvHead * headDim;
          for (int i = 0; i < headDim; i++)
            out[queryOffset + i] += weight * values[valueOffset + i];
        }
      }
    }
  }
}
