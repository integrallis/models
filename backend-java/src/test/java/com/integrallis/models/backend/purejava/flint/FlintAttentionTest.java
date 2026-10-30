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
package com.integrallis.models.backend.purejava.flint;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FlintAttentionTest {
  @Test
  void matchesScalarChronologicalArithmeticForLinearAndWrappedGroupedHeads() {
    for (int width : new int[] {4, 17, 128}) {
      for (int count : new int[] {3, 257, 513}) {
        verify(width, count, false);
        verify(width, count, true);
      }
    }
  }

  private void verify(int width, int count, boolean ring) {
    int heads = 32, kvHeads = 4, stride = kvHeads * width;
    Random random = new Random(42);
    var cache =
        new LayeredKvCache(
            count + 5,
            ring ? LayerSpec.ring(stride, stride, count) : LayerSpec.linear(stride, stride));
    float[] query = values(random, heads * width), sinks = values(random, heads);
    int last = ring ? count + 4 : count - 1;
    for (int position = 0; position <= last; position++) {
      cache.store(0, position, values(random, stride), values(random, stride));
    }
    var view = cache.attentionView(0, last + 1 - count, last + 1);
    float[] keys = cache.keyBuffer(0), values = cache.valueBuffer(0);
    float[] expected = new float[heads * width], actual = new float[heads * width];
    float scale = (float) (1.0 / Math.sqrt(width));
    for (int head = 0; head < heads; head++) {
      float[] scores = new float[count];
      int kvOffset = head / (heads / kvHeads) * width;
      int index = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        var span = view.span(spanIndex);
        for (int row = 0; row < span.positionCount(); row++) {
          scores[index++] =
              scale
                  * PinnedReduction.dot(
                      query, head * width, keys, span.keyOffset() + row * stride + kvOffset, width);
        }
      }
      TensorOps.softmaxWithZeroValueSink(scores, 0, count, sinks[head]);
      index = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        var span = view.span(spanIndex);
        for (int row = 0; row < span.positionCount(); row++) {
          float weight = scores[index++];
          for (int dimension = 0; dimension < width; dimension++) {
            expected[head * width + dimension] +=
                weight * values[span.valueOffset() + row * stride + kvOffset + dimension];
          }
        }
      }
    }
    float[][] scratch = new float[heads][count];
    for (int repeat = 0; repeat < 3; repeat++) {
      FlintAttention.compute(
          query, keys, values, scratch, actual, sinks, view, heads, kvHeads, width);
      for (int i = 0; i < expected.length; i++) {
        assertThat(Float.floatToRawIntBits(actual[i]))
            .as("width=%s positions=%s ring=%s element=%s", width, count, ring, i)
            .isEqualTo(Float.floatToRawIntBits(expected[i]));
      }
    }
  }

  private float[] values(Random random, int length) {
    float[] result = new float[length];
    for (int i = 0; i < length; i++) result[i] = (random.nextFloat() - 0.5f) * 4;
    return result;
  }
}
