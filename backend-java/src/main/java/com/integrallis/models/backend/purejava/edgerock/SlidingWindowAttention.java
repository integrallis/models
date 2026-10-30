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
package com.integrallis.models.backend.purejava.edgerock;

import java.util.Arrays;
import java.util.Objects;

/**
 * Grouped-query attention over a sliding window, for EdgeRock's 8 full-attention layers.
 *
 * <p>The span is {@code [position - window + 1, position]} inclusive, exactly {@code window} keys
 * counting the query's own. The reference states both halves separately -- {@code key_pos <=
 * query_pos} and {@code key_pos >= query_pos - window + 1} -- and the second is where an off-by-one
 * hides: a window that admits one key too many still produces fluent output.
 *
 * <p><strong>There is no sink here.</strong> The memory read appends a zero logit before its
 * softmax so it can attend to nothing; this is a plain masked softmax whose weights sum to one. The
 * two live side by side on the same layer, so the asymmetry is deliberate and tested: a single
 * visible key takes weight 1.0 here and 0.5 there.
 */
final class SlidingWindowAttention {

  private SlidingWindowAttention() {}

  /**
   * A bounded key/value cache holding at most {@code window} entries per group.
   *
   * <p>A ring buffer rather than a growing list: the window is what makes this layer constant
   * memory, and EdgeRock's own case for itself is constant memory at 262144 context.
   */
  static final class Cache {

    private final int groups;
    private final int window;
    private final int dim;
    private final float[][][] keys;
    private final float[][][] values;
    private final int[] positions;
    private int written;

    Cache(int groups, int window, int dim) {
      if (groups <= 0 || window <= 0 || dim <= 0) {
        throw new IllegalArgumentException(
            "groups, window and dim must be positive: " + groups + ", " + window + ", " + dim);
      }
      this.groups = groups;
      this.window = window;
      this.dim = dim;
      this.keys = new float[groups][window][dim];
      this.values = new float[groups][window][dim];
      this.positions = new int[window];
      Arrays.fill(this.positions, -1);
    }

    int groups() {
      return groups;
    }

    int window() {
      return window;
    }

    int dim() {
      return dim;
    }

    int written() {
      return written;
    }

    /** Appends one token's key and value, evicting the entry {@code window} positions back. */
    void append(float[][] groupKeys, float[][] groupValues, int position) {
      Objects.requireNonNull(groupKeys, "groupKeys");
      Objects.requireNonNull(groupValues, "groupValues");
      if (position < 0) {
        throw new IllegalArgumentException("position must be non-negative: " + position);
      }
      int slot = Math.floorMod(position, window);
      for (int group = 0; group < groups; group++) {
        System.arraycopy(groupKeys[group], 0, keys[group][slot], 0, dim);
        System.arraycopy(groupValues[group], 0, values[group][slot], 0, dim);
      }
      positions[slot] = position;
      written++;
    }

    int positionAt(int slot) {
      return positions[slot];
    }

    float[] key(int group, int slot) {
      return keys[group][slot];
    }

    float[] value(int group, int slot) {
      return values[group][slot];
    }

    void reset() {
      Arrays.fill(positions, -1);
      written = 0;
      for (int group = 0; group < groups; group++) {
        for (int slot = 0; slot < window; slot++) {
          Arrays.fill(keys[group][slot], 0f);
          Arrays.fill(values[group][slot], 0f);
        }
      }
    }
  }

  /**
   * Attends one query position over the cached window.
   *
   * @param cache keys and values already appended, including this position's own
   * @param queries per-head query, {@code [head][dim]}
   * @param position the query's sequence position
   * @param headsPerGroup query heads sharing one key/value group
   * @param out per-head output, {@code [head][dim]}, overwritten
   */
  static void read(Cache cache, float[][] queries, int position, int headsPerGroup, float[][] out) {
    Objects.requireNonNull(cache, "cache");
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    int heads = queries.length;
    if (out.length != heads) {
      throw new IllegalArgumentException(
          "queries and out must agree on head count: " + heads + ", " + out.length);
    }
    if (headsPerGroup <= 0 || heads % headsPerGroup != 0) {
      throw new IllegalArgumentException(
          "heads " + heads + " must be a positive multiple of headsPerGroup " + headsPerGroup);
    }
    if (heads / headsPerGroup != cache.groups()) {
      throw new IllegalArgumentException(
          "heads/headsPerGroup "
              + (heads / headsPerGroup)
              + " must equal the cache's group count "
              + cache.groups());
    }

    int window = cache.window();
    int dim = cache.dim();
    float scale = (float) (1.0 / Math.sqrt(dim));
    // Inclusive on both ends: exactly `window` keys, the query's own included.
    int oldest = position - window + 1;
    float[] weights = new float[window];

    for (int head = 0; head < heads; head++) {
      int group = head / headsPerGroup;
      float[] query = queries[head];
      Arrays.fill(out[head], 0f);

      float max = Float.NEGATIVE_INFINITY;
      int visible = 0;
      for (int slot = 0; slot < window; slot++) {
        int keyPosition = cache.positionAt(slot);
        if (keyPosition < 0 || keyPosition > position || keyPosition < oldest) {
          weights[slot] = Float.NEGATIVE_INFINITY;
          continue;
        }
        float dot = 0f;
        float[] key = cache.key(group, slot);
        for (int i = 0; i < dim; i++) {
          dot += query[i] * key[i];
        }
        weights[slot] = dot * scale;
        max = Math.max(max, weights[slot]);
        visible++;
      }
      if (visible == 0) {
        // Only reachable if nothing has been appended for this position, which is a caller bug
        // rather than a state the model reaches: the query's own key is always in the window.
        throw new IllegalStateException(
            "no visible keys for position " + position + "; append before reading");
      }

      float total = 0f;
      for (int slot = 0; slot < window; slot++) {
        if (weights[slot] == Float.NEGATIVE_INFINITY) {
          continue;
        }
        float weight = (float) Math.exp(weights[slot] - max);
        weights[slot] = weight;
        total += weight;
      }
      for (int slot = 0; slot < window; slot++) {
        if (weights[slot] == Float.NEGATIVE_INFINITY) {
          continue;
        }
        float weight = weights[slot] / total;
        float[] value = cache.value(group, slot);
        for (int i = 0; i < dim; i++) {
          out[head][i] += weight * value[i];
        }
      }
    }
  }
}
