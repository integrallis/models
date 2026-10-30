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

import com.integrallis.models.backend.purejava.flint.FlintWeights.Matrix;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/** Immutable small router snapshot shared by sessions; expert matrices stay mapped. */
final class FlintRouterProjection {
  private final float[] weights;
  private final int rows;
  private final int columns;

  private FlintRouterProjection(Matrix matrix) {
    rows = matrix.rows();
    columns = matrix.columns();
    weights = new float[Math.multiplyExact(rows, columns)];
    MemorySegment.copy(
        matrix.data(),
        ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN),
        0,
        weights,
        0,
        weights.length);
  }

  static FlintRouterProjection prepare(Matrix matrix, long remainingBytes) {
    long elements = (long) matrix.rows() * matrix.columns();
    if (matrix.type() != GgufTensorType.F32
        || elements > remainingBytes / Float.BYTES
        || elements > Integer.MAX_VALUE) return null;
    return new FlintRouterProjection(matrix);
  }

  long bytes() {
    return (long) weights.length * Float.BYTES;
  }

  void multiply(float[] input, float[] output) {
    for (int row = 0; row < rows; row++) {
      // Identical row reduction to TensorOps.f32Matmul, without its repeated row copy.
      output[row] = PinnedReduction.dot(input, 0, weights, row * columns, columns);
    }
  }
}
