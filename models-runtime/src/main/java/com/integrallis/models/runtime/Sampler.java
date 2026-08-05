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
package com.integrallis.models.runtime;

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.SamplingOptions;
import java.util.List;
import java.util.Random;

/** Samples the next token from logits using configurable strategies. */
public final class Sampler {

  private final SamplingOptions options;
  private final Random rng;

  public Sampler(SamplingOptions options) {
    this.options = options;
    this.rng = options.seed() != null ? new Random(options.seed()) : new Random();
  }

  /** Samples the next token ID from the given logits array. */
  public int sample(float[] logits, List<Integer> previousTokens) {
    if (options.temperature() == 0.0f && options.repetitionPenalty() == 1.0f) {
      return argmax(logits);
    }
    float[] adjusted = logits.clone();
    applyRepetitionPenalty(adjusted, previousTokens);
    return sampleAdjusted(adjusted);
  }

  /** Samples one row without copying transient logits for unpenalized greedy generation. */
  public int sample(LogitBatch logits, int tokenIndex, List<Integer> previousTokens) {
    if (options.temperature() == 0.0f && options.repetitionPenalty() == 1.0f) {
      return logits.argmax(tokenIndex);
    }
    float[] adjusted = logits.copyRow(tokenIndex);
    applyRepetitionPenalty(adjusted, previousTokens);
    return sampleAdjusted(adjusted);
  }

  private void applyRepetitionPenalty(float[] adjusted, List<Integer> previousTokens) {
    if (options.repetitionPenalty() > 1.0f && previousTokens != null) {
      for (int tokenId : previousTokens) {
        if (tokenId >= 0 && tokenId < adjusted.length) {
          if (adjusted[tokenId] > 0) {
            adjusted[tokenId] /= options.repetitionPenalty();
          } else {
            adjusted[tokenId] *= options.repetitionPenalty();
          }
        }
      }
    }
  }

  private int sampleAdjusted(float[] adjusted) {
    // Greedy (temperature = 0)
    if (options.temperature() == 0.0f) {
      return argmax(adjusted);
    }

    // Temperature scaling
    float temp = options.temperature();
    for (int i = 0; i < adjusted.length; i++) {
      adjusted[i] /= temp;
    }

    // Convert to probabilities via softmax
    float max = Float.NEGATIVE_INFINITY;
    for (float v : adjusted) {
      if (v > max) max = v;
    }
    float sum = 0;
    for (int i = 0; i < adjusted.length; i++) {
      adjusted[i] = (float) Math.exp(adjusted[i] - max);
      sum += adjusted[i];
    }
    for (int i = 0; i < adjusted.length; i++) {
      adjusted[i] /= sum;
    }

    // Top-K selection. A bounded heap keeps this O(vocab log k) instead of sorting the whole
    // vocabulary; at Qwen3's 151,936 tokens the full sort dominated per-token cost.
    int topK = Math.min(options.topK(), adjusted.length);
    int[] candidates = selectTopK(adjusted, topK);

    // Top-P (nucleus) filtering
    float cumulative = 0;
    int cutoff = candidates.length;
    for (int i = 0; i < candidates.length; i++) {
      cumulative += adjusted[candidates[i]];
      if (cumulative >= options.topP()) {
        cutoff = i + 1;
        break;
      }
    }

    // Re-normalize
    float totalProb = 0;
    for (int i = 0; i < cutoff; i++) {
      totalProb += adjusted[candidates[i]];
    }

    // Sample from the filtered distribution
    float r = rng.nextFloat() * totalProb;
    float acc = 0;
    for (int i = 0; i < cutoff; i++) {
      acc += adjusted[candidates[i]];
      if (acc >= r) {
        return candidates[i];
      }
    }

    return candidates[cutoff - 1];
  }

  /**
   * Returns the {@code k} highest-probability token ids, most likely first.
   *
   * <p>Ties resolve to the lower token id, matching the stable descending sort this replaced. That
   * ordering is observable: it decides which tokens survive the top-k cut and the top-p boundary,
   * so seeded runs stay reproducible.
   */
  private static int[] selectTopK(float[] probabilities, int k) {
    int[] heap = new int[k];
    int size = 0;
    for (int id = 0; id < probabilities.length; id++) {
      if (size < k) {
        heap[size] = id;
        siftUp(heap, probabilities, size);
        size++;
      } else if (isWorse(probabilities, heap[0], id)) {
        heap[0] = id;
        siftDown(heap, probabilities, size);
      }
    }
    // The root is the weakest survivor, so draining it fills the result back to front.
    int count = size;
    int[] ordered = new int[count];
    for (int index = count - 1; index >= 0; index--) {
      ordered[index] = heap[0];
      heap[0] = heap[size - 1];
      size--;
      siftDown(heap, probabilities, size);
    }
    return ordered;
  }

  /** Strict order over token ids: lower probability is worse, and ties prefer the lower id. */
  private static boolean isWorse(float[] probabilities, int left, int right) {
    if (probabilities[left] != probabilities[right]) {
      return probabilities[left] < probabilities[right];
    }
    return left > right;
  }

  private static void siftUp(int[] heap, float[] probabilities, int index) {
    while (index > 0) {
      int parent = (index - 1) >>> 1;
      if (!isWorse(probabilities, heap[index], heap[parent])) {
        break;
      }
      swap(heap, index, parent);
      index = parent;
    }
  }

  private static void siftDown(int[] heap, float[] probabilities, int size) {
    int index = 0;
    while (true) {
      int left = 2 * index + 1;
      if (left >= size) {
        break;
      }
      int weakest = left;
      int right = left + 1;
      if (right < size && isWorse(probabilities, heap[right], heap[left])) {
        weakest = right;
      }
      if (!isWorse(probabilities, heap[weakest], heap[index])) {
        break;
      }
      swap(heap, index, weakest);
      index = weakest;
    }
  }

  private static void swap(int[] heap, int left, int right) {
    int value = heap[left];
    heap[left] = heap[right];
    heap[right] = value;
  }

  private static int argmax(float[] arr) {
    int best = 0;
    for (int i = 1; i < arr.length; i++) {
      if (arr[i] > arr[best]) {
        best = i;
      }
    }
    return best;
  }
}
