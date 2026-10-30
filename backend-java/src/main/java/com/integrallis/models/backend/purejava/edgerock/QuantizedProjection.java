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
public final class QuantizedProjection implements Projection {

  private final MemorySegment data;
  private final GgufTensorType type;
  private final int rows;
  private final int columns;
  private final String name;

  /**
   * Reusable Q8 activation scratch, allocated once per projection.
   *
   * <p>This was a record calling the short {@code ggufMatmul} overload, which allocates four arrays
   * and re-quantizes the activation on every call. One memory layer makes nine projections per
   * token and the model has eight of them, so that was on the order of 288 allocations per token
   * before anything was multiplied. {@code Qwen35ForwardPass} avoids it by carrying a {@code
   * ProjectionScratch}; this carries its own, sized from its own column count.
   *
   * <p>Not thread-safe, which matches the class that owns it: {@code EdgeRockMemoryLayer} already
   * holds per-layer activation buffers and runs one token at a time.
   */
  private final byte[] quantizedActivation;

  private final float[] quantizedActivationScales;

  public QuantizedProjection(
      MemorySegment data, GgufTensorType type, int rows, int columns, String name) {
    Objects.requireNonNull(data, "data");
    Objects.requireNonNull(type, "type");
    Objects.requireNonNull(name, "name");
    if (rows <= 0 || columns <= 0) {
      throw new IllegalArgumentException(
          "rows and columns must be positive: " + rows + "x" + columns);
    }
    this.data = data;
    this.type = type;
    this.rows = rows;
    this.columns = columns;
    this.name = name;
    // The K-quants quantize the activation in blocks of 256, the legacy quants in 32. Only the
    // types this projection can actually carry are sized here; anything else falls through to the
    // float path in ggufMatmul and never reads these.
    int block =
        switch (type) {
          case Q4_K, Q5_K, Q6_K -> 256;
          case Q4_0, Q5_0, Q8_0 -> 32;
          default -> 0;
        };
    this.quantizedActivation = block == 0 ? null : new byte[columns];
    this.quantizedActivationScales = block == 0 ? null : new float[columns / block];
  }

  public MemorySegment data() {
    return data;
  }

  public GgufTensorType type() {
    return type;
  }

  public int rows() {
    return rows;
  }

  public int columns() {
    return columns;
  }

  public String name() {
    return name;
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
    if (quantizedActivation == null) {
      TensorOps.ggufMatmul(output, input, data, type, rows, columns);
      return;
    }
    TensorOps.ggufMatmul(
        output, input, data, type, rows, columns, quantizedActivation, quantizedActivationScales);
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
