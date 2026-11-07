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
package com.integrallis.models.backend.purejava.vision;

import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.vectors.core.GgufStagePlan;

/** Exact row reductions shared by the vision encoder and projector, using the GGUF worker pool. */
final class VisionProjection {
  private static final Thread ACCESS_PROBE = Thread.ofPlatform().unstarted(() -> {});

  private VisionProjection() {}

  static void multiply(float[] output, float[] input, int tokens, ClipVisionWeights.Matrix matrix) {
    GgufStagePlan.RangeOperation rows =
        (from, to) -> {
          float[] weight = new float[matrix.columns()];
          for (int row = from; row < to; row++) {
            GgufTensorValues.dequantizeRow(
                matrix.data(), matrix.type(), row, matrix.columns(), weight);
            for (int token = 0; token < tokens; token++) {
              output[token * matrix.rows() + row] =
                  PinnedReduction.dot(input, token * matrix.columns(), weight, 0, matrix.columns());
            }
          }
        };
    if ((long) tokens * matrix.rows() * matrix.columns() >= 131_072
        && matrix.data().isAccessibleBy(ACCESS_PROBE)) {
      GgufStagePlan.of(GgufStagePlan.stage(matrix.rows(), rows)).execute();
    } else {
      rows.execute(0, matrix.rows());
    }
  }
}
