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

  static com.integrallis.models.backend.purejava.spi.DenseProjectionKernels.Projection prepare(
      ClipVisionWeights.Matrix matrix,
      com.integrallis.models.backend.purejava.spi.DenseProjectionKernels kernels) {
    if (kernels == null) return (output, input, tokens) -> multiply(output, input, tokens, matrix);
    float[] weights = new float[Math.multiplyExact(matrix.rows(), matrix.columns())];
    float[] row = new float[matrix.columns()];
    for (int r = 0; r < matrix.rows(); r++) {
      GgufTensorValues.dequantizeRow(matrix.data(), matrix.type(), r, matrix.columns(), row);
      System.arraycopy(row, 0, weights, r * matrix.columns(), row.length);
    }
    return kernels.prepare(weights, matrix.rows(), matrix.columns());
  }

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
