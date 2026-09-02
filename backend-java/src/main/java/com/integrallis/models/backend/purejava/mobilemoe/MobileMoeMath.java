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
package com.integrallis.models.backend.purejava.mobilemoe;

import java.util.Objects;

/** Architecture-specific scalar math used by the MobileMoE correctness path. */
final class MobileMoeMath {

  private MobileMoeMath() {}

  static void normalizeHeads(float[] values, int headDimension, float epsilon) {
    Objects.requireNonNull(values, "values");
    if (headDimension <= 0 || values.length % headDimension != 0) {
      throw new IllegalArgumentException("headDimension must divide the value buffer");
    }
    if (!(epsilon > 0.0f) || !Float.isFinite(epsilon)) {
      throw new IllegalArgumentException("epsilon must be finite and positive");
    }
    for (int offset = 0; offset < values.length; offset += headDimension) {
      double sumSquares = 0.0;
      for (int index = 0; index < headDimension; index++) {
        float value = values[offset + index];
        sumSquares += value * value;
      }
      float multiplier = (float) (1.0 / Math.sqrt(sumSquares / headDimension + epsilon));
      for (int index = 0; index < headDimension; index++) {
        values[offset + index] *= multiplier;
      }
    }
  }
}
