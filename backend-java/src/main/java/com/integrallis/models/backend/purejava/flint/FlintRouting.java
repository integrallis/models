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

/** Sigmoid routing: the optional bias affects selection only, never the mixture weights. */
final class FlintRouting {
  private FlintRouting() {}

  static void select(
      float[] logits,
      float[] bias,
      int[] selected,
      float[] weights,
      boolean normalize,
      float scale) {
    int used = selected.length;
    int filled = 0;
    for (int expert = 0; expert < logits.length; expert++) {
      logits[expert] = 1f / (1f + (float) Math.exp(-logits[expert]));
      float score = logits[expert] + bias[expert];
      if (filled == used && score <= weights[used - 1]) continue;
      int slot = Math.min(filled, used - 1);
      while (slot > 0 && weights[slot - 1] < score) {
        weights[slot] = weights[slot - 1];
        selected[slot] = selected[slot - 1];
        slot--;
      }
      weights[slot] = score;
      selected[slot] = expert;
      if (filled < used) filled++;
    }
    float sum = 0f;
    for (int i = 0; i < used; i++) {
      weights[i] = logits[selected[i]];
      sum += weights[i];
    }
    // Matches the GGUF reference's F16-normal minimum guard on the selected weight sum.
    float denominator = normalize ? Math.max(sum, 0x1.0p-14f) : 1f;
    for (int i = 0; i < used; i++) weights[i] = weights[i] / denominator * scale;
  }
}
