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

import com.integrallis.models.backend.purejava.edgerock.EdgeRockMemoryLayer.Projection;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * A {@link Projection} over a quantized GGUF tensor, straight onto {@link TensorOps#ggufMatmul}.
 *
 * <p>This is the seam the memory layer was written against, now filled. It deliberately does not
 * reuse {@code Qwen35Weights.Matrix}: that type is package-private to {@code qwen35} and reaching
 * it would mean either widening its visibility or placing EdgeRock inside that package, and the
 * only thing shared is a tuple of (segment, type, rows, columns) that {@code ggufMatmul} already
 * accepts directly. The arithmetic is the same kernel either way.
 *
 * <p>No Hadamard transform is applied. {@code Qwen35ForwardPass} consults {@code
 * PrismHadamard.projection(name)} before projecting, because a Prism-transformed checkpoint stores
 * rotated weights. EdgeRock-4B publishes no {@code prism.hadamard.version} key, so there is nothing
 * to undo; a checkpoint that did carry one would need that lookup here, and {@link
 * #requireNoHadamard} exists so such a file fails rather than decoding into noise.
 */
record QuantizedProjection(
    MemorySegment data, GgufTensorType type, int rows, int columns, String name)
    implements Projection {

  QuantizedProjection {
    Objects.requireNonNull(data, "data");
    Objects.requireNonNull(type, "type");
    Objects.requireNonNull(name, "name");
    if (rows <= 0 || columns <= 0) {
      throw new IllegalArgumentException(
          "rows and columns must be positive: " + rows + "x" + columns);
    }
  }

  @Override
  public void apply(float[] input, float[] output) {
    if (input.length != columns) {
      throw new IllegalArgumentException(
          name + " expects an input of " + columns + " but got " + input.length);
    }
    if (output.length != rows) {
      throw new IllegalArgumentException(
          name + " produces " + rows + " but was given " + output.length);
    }
    TensorOps.ggufMatmul(output, input, data, type, rows, columns);
  }

  /**
   * Refuses a checkpoint whose weights are Hadamard-rotated.
   *
   * <p>Called at load time rather than per projection. A rotated weight projected without its
   * transform produces finite activations and fluent text, so the failure has to be raised where it
   * can still be seen.
   */
  static void requireNoHadamard(boolean present) {
    if (present) {
      throw new UnsupportedOperationException(
          "edgerock does not implement Prism Hadamard projections; this checkpoint carries "
              + "prism.hadamard.version and its weights are rotated");
    }
  }
}
