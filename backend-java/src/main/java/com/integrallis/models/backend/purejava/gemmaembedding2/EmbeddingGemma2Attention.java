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

import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufStagePlan;
import java.util.Arrays;

/** Bidirectional attention with independent heads sharing the configured GGUF worker pool. */
final class EmbeddingGemma2Attention {
  // Publishing four short head tasks costs more than their work at 32 tokens. Keep the same
  // arithmetic inline for short inputs; larger contexts amortize the shared-pool publication.
  private static final int MIN_PARALLEL_TOKENS = 64;
  private final int heads;
  private final GgufStagePlan plan;
  private final float[][] scores;
  private float[] queries;
  private float[] keys;
  private float[] values;
  private float[] output;
  private int count;
  private int kvHeads;
  private int headDim;
  private int window;

  EmbeddingGemma2Attention(int heads) {
    if (heads < 1) throw new IllegalArgumentException("heads must be positive");
    this.heads = heads;
    scores = new float[heads][0];
    plan = GgufStagePlan.of(GgufStagePlan.stage(heads, this::attendHeads));
  }

  /** The owning encoder serializes calls; each worker writes only its assigned heads. */
  void compute(
      float[] queries,
      float[] keys,
      float[] values,
      float[] output,
      int count,
      int kvHeads,
      int headDim,
      int window) {
    for (int head = 0; head < heads; head++) {
      if (scores[head].length < count) scores[head] = new float[count];
    }
    this.queries = queries;
    this.keys = keys;
    this.values = values;
    this.output = output;
    this.count = count;
    this.kvHeads = kvHeads;
    this.headDim = headDim;
    this.window = window;
    try {
      if (count < MIN_PARALLEL_TOKENS) attendHeads(0, heads);
      else plan.execute();
    } finally {
      this.queries = null;
      this.keys = null;
      this.values = null;
      this.output = null;
      this.count = 0;
    }
  }

  private void attendHeads(int first, int last) {
    int queryDim = heads * headDim;
    int keyDim = kvHeads * headDim;
    for (int head = first; head < last; head++) {
      float[] rowScores = scores[head];
      int kvHead = head / (heads / kvHeads);
      for (int position = 0; position < count; position++) {
        int start = window == 0 ? 0 : Math.max(0, position - window / 2);
        int end = window == 0 ? count : Math.min(count, position + window / 2 + 1);
        int queryOffset = position * queryDim + head * headDim;
        Arrays.fill(output, queryOffset, queryOffset + headDim, 0.0f);
        for (int keyPosition = start; keyPosition < end; keyPosition++) {
          // Scale 1: the learned query norm already supplies the scale.
          rowScores[keyPosition] =
              PinnedReduction.dot(
                  queries, queryOffset, keys, keyPosition * keyDim + kvHead * headDim, headDim);
        }
        TensorOps.softmax(rowScores, start, end - start);
        for (int keyPosition = start; keyPosition < end; keyPosition++) {
          float weight = rowScores[keyPosition];
          int valueOffset = keyPosition * keyDim + kvHead * headDim;
          // Preserve the original separate multiply/add and increasing-key accumulation order.
          for (int i = 0; i < headDim; i++)
            output[queryOffset + i] += weight * values[valueOffset + i];
        }
      }
    }
  }
}
