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

import com.integrallis.models.api.SamplingOptions;
import java.util.ArrayList;
import java.util.Comparator;
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
    float[] adjusted = logits.clone();

    // Apply repetition penalty
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

    // Build sorted index for top-k/top-p
    List<TokenProb> sorted = new ArrayList<>(adjusted.length);
    for (int i = 0; i < adjusted.length; i++) {
      sorted.add(new TokenProb(i, adjusted[i]));
    }
    sorted.sort(Comparator.comparingDouble(TokenProb::prob).reversed());

    // Top-K filtering
    int topK = Math.min(options.topK(), sorted.size());
    sorted = new ArrayList<>(sorted.subList(0, topK));

    // Top-P (nucleus) filtering
    float cumulative = 0;
    int cutoff = sorted.size();
    for (int i = 0; i < sorted.size(); i++) {
      cumulative += sorted.get(i).prob;
      if (cumulative >= options.topP()) {
        cutoff = i + 1;
        break;
      }
    }
    sorted = new ArrayList<>(sorted.subList(0, cutoff));

    // Re-normalize
    float totalProb = 0;
    for (TokenProb tp : sorted) {
      totalProb += tp.prob;
    }

    // Sample from the filtered distribution
    float r = rng.nextFloat() * totalProb;
    float acc = 0;
    for (TokenProb tp : sorted) {
      acc += tp.prob;
      if (acc >= r) {
        return tp.id;
      }
    }

    return sorted.getLast().id;
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

  private record TokenProb(int id, float prob) {}
}
