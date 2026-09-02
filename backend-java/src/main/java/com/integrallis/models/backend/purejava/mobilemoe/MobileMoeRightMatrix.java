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

import java.lang.foreign.SegmentAllocator;
import java.util.Locale;
import java.util.Objects;

/** Runtime execution layout for one MobileMoE routed-expert projection. */
final class MobileMoeRightMatrix {

  enum Layout {
    Q8,
    PACKED_INT4;

    static Layout configured() {
      String configured =
          System.getProperty("models.mobilemoe.runtimeLayout", "q8")
              .trim()
              .replace('-', '_')
              .toUpperCase(Locale.ROOT);
      try {
        return valueOf(configured);
      } catch (IllegalArgumentException exception) {
        throw new IllegalArgumentException(
            "models.mobilemoe.runtimeLayout must be q8 or packed-int4: " + configured, exception);
      }
    }

    String propertyValue() {
      return name().toLowerCase(Locale.ROOT).replace('_', '-');
    }
  }

  private final MobileMoePackedInt4RightMatrix packed;
  private final MobileMoeQ8RightMatrix q8;

  private MobileMoeRightMatrix(MobileMoePackedInt4RightMatrix packed, MobileMoeQ8RightMatrix q8) {
    this.packed = Objects.requireNonNull(packed, "packed");
    this.q8 = q8;
  }

  static MobileMoeRightMatrix prepare(
      MobileMoePackedInt4RightMatrix packed, Layout layout, SegmentAllocator allocator) {
    Objects.requireNonNull(layout, "layout");
    return new MobileMoeRightMatrix(
        packed, layout == Layout.Q8 ? packed.materializeQ8(allocator) : null);
  }

  void multiply(
      float[] input, float[] output, byte[] quantizedActivation, float[] activationScales) {
    if (q8 == null) {
      packed.multiply(input, output);
    } else {
      q8.multiply(input, output, quantizedActivation, activationScales);
    }
  }

  void multiplyBatch(
      float[] input,
      int batchSize,
      float[] output,
      byte[] quantizedActivation,
      float[] activationScales) {
    if (q8 == null) {
      packed.multiplyBatch(input, batchSize, output);
    } else {
      q8.multiplyBatch(input, batchSize, output, quantizedActivation, activationScales);
    }
  }
}
