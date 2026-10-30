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

import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionView;
import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.util.Arrays;
import java.util.stream.IntStream;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Independent heads; each key reduction and chronological value fold retains Flint's order. */
final class FlintAttention {
  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
  private static final long PARALLEL_MIN_ELEMENTS = 1L << 20;

  private FlintAttention() {}

  static void compute(
      float[] query,
      float[] keys,
      float[] values,
      float[][] scores,
      float[] output,
      float[] sinks,
      AttentionView view,
      int heads,
      int kvHeads,
      int width) {
    Arrays.fill(output, 0.0f);
    IntStream work = IntStream.range(0, heads);
    if ((long) heads * width * view.positionCount() >= PARALLEL_MIN_ELEMENTS) {
      work = work.parallel();
    }
    work.forEach(
        head ->
            computeHead(
                query,
                keys,
                values,
                scores[head],
                output,
                sinks[head],
                view,
                head,
                heads / kvHeads,
                kvHeads * width,
                width));
  }

  private static void computeHead(
      float[] query,
      float[] keys,
      float[] values,
      float[] scores,
      float[] output,
      float sink,
      AttentionView view,
      int head,
      int headsPerKv,
      int stride,
      int width) {
    int queryOffset = head * width;
    int kvOffset = head / headsPerKv * width;
    float scale = (float) (1.0 / Math.sqrt(width));
    int index = 0;
    for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
      var span = view.span(spanIndex);
      for (int row = 0; row < span.positionCount(); row++) {
        scores[index++] =
            scale
                * PinnedReduction.dot(
                    query, queryOffset, keys, span.keyOffset() + row * stride + kvOffset, width);
      }
    }
    TensorOps.softmaxWithZeroValueSink(scores, 0, view.positionCount(), sink);
    index = 0;
    for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
      var span = view.span(spanIndex);
      for (int row = 0; row < span.positionCount(); row++) {
        float weight = scores[index++];
        int rowOffset = span.valueOffset() + row * stride + kvOffset;
        int dimension = 0;
        for (; dimension < SPECIES.loopBound(width); dimension += SPECIES.length()) {
          // Deliberately separate multiply and add, matching the original scalar fold (not FMA).
          FloatVector.fromArray(SPECIES, values, rowOffset + dimension)
              .mul(weight)
              .add(FloatVector.fromArray(SPECIES, output, queryOffset + dimension))
              .intoArray(output, queryOffset + dimension);
        }
        for (; dimension < width; dimension++) {
          output[queryOffset + dimension] += weight * values[rowOffset + dimension];
        }
      }
    }
  }
}
