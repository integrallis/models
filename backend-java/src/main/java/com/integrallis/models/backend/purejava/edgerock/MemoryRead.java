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

import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.util.Objects;

/**
 * The grouped-query read over a {@link MemoryBank}.
 *
 * <p>One detail decides whether this is right: <strong>a zero logit is appended before the softmax
 * and dropped afterwards</strong>. The read weights therefore do not sum to one, and a layer whose
 * every record scores poorly contributes nearly nothing instead of being forced to average whatever
 * it has. Omit the sink and every weight changes, because the normaliser loses its {@code exp(0)}
 * term -- the output stays finite and fluent, which is why this has a test rather than a comment.
 *
 * <p>Arithmetic is {@code float}, which matches the reference: it casts query, keys and values to
 * float32 before the matmul whatever the model dtype. On this path that is the natural width rather
 * than a choice, but it is the reason a future bf16 fast path here would change results.
 */
final class MemoryRead {

  private MemoryRead() {}

  /**
   * Reads every head's attention over its group's committed records.
   *
   * @param bank the committed state; records are read, never written
   * @param queries per-head query, {@code [head][dim]}, already normed and rotated
   * @param gates per-head read gate, in {@code (-2, 2)}
   * @param headsPerGroup query heads sharing one key/value group
   * @param out per-head read output, {@code [head][dim]}, overwritten
   */
  static void read(
      MemoryBank bank, float[][] queries, float[] gates, int headsPerGroup, float[][] out) {
    Objects.requireNonNull(bank, "bank");
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(gates, "gates");
    Objects.requireNonNull(out, "out");
    int heads = queries.length;
    if (gates.length != heads || out.length != heads) {
      throw new IllegalArgumentException(
          "queries, gates and out must agree on head count: "
              + heads
              + ", "
              + gates.length
              + ", "
              + out.length);
    }
    if (headsPerGroup <= 0 || heads % headsPerGroup != 0) {
      throw new IllegalArgumentException(
          "heads " + heads + " must be a positive multiple of headsPerGroup " + headsPerGroup);
    }
    if (heads / headsPerGroup != bank.groups()) {
      throw new IllegalArgumentException(
          "heads/headsPerGroup "
              + (heads / headsPerGroup)
              + " must equal the bank's group count "
              + bank.groups());
    }

    int dim = bank.dim();
    int capacity = bank.capacity();
    float scale = (float) (1.0 / Math.sqrt(dim));
    MemoryAnchors.Selection selection = bank.selection();
    float[] logits = new float[capacity];

    for (int head = 0; head < heads; head++) {
      int group = head / headsPerGroup;
      float[] query = queries[head];
      java.util.Arrays.fill(out[head], 0f);

      // Absent slots are masked to -inf, which exp()s to zero inside the shared softmax. The
      // sink's logit is zero: TensorOps.softmaxWithZeroValueSink is the same primitive another
      // architecture already needed, and it rejects NaN and +inf scores, which a hand-rolled loop
      // here did not.
      int occupied = 0;
      for (int slot = 0; slot < capacity; slot++) {
        if (!selection.occupied(slot)) {
          logits[slot] = Float.NEGATIVE_INFINITY;
          continue;
        }
        float dot = 0f;
        float[] key = bank.key(group, slot);
        for (int i = 0; i < dim; i++) {
          dot += query[i] * key[i];
        }
        logits[slot] = dot * scale;
        occupied++;
      }
      if (occupied == 0) {
        // Nothing committed: the sink is the whole distribution and the read is zero. It falls out
        // of the sink rather than being special-cased, but an empty bank is every sequence's first
        // chunk, so it is asserted.
        continue;
      }
      TensorOps.softmaxWithZeroValueSink(logits, 0, capacity, 0.0f);

      float gate = gates[head];
      for (int slot = 0; slot < capacity; slot++) {
        if (!selection.occupied(slot)) {
          continue;
        }
        float weight = gate * logits[slot];
        float[] value = bank.value(group, slot);
        for (int i = 0; i < dim; i++) {
          out[head][i] += weight * value[i];
        }
      }
    }
  }

  /**
   * The per-head read gate: {@code tanh(scale) * 2 * sigmoid(dot(query, weight)/sqrt(dim) + bias)}.
   *
   * <p>Range is {@code (-2, 2)}, so a head can suppress its read or invert it. {@code readScale} is
   * zero-initialised in the reference, which makes an untrained gate exactly zero.
   */
  static void gates(
      float[][] queries, float[][] gateWeights, float[] gateBias, float[] readScale, float[] out) {
    int heads = queries.length;
    int dim = queries[0].length;
    float scale = (float) (1.0 / Math.sqrt(dim));
    for (int head = 0; head < heads; head++) {
      float dot = 0f;
      for (int i = 0; i < dim; i++) {
        dot += queries[head][i] * gateWeights[head][i];
      }
      float sigmoid = (float) (1.0 / (1.0 + Math.exp(-(dot * scale + gateBias[head]))));
      out[head] = (float) (Math.tanh(readScale[head]) * 2.0 * sigmoid);
    }
  }
}
