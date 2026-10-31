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
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
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

  /**
   * The accelerated projection route, or {@link GgufBatchedMatrixKernel#none()}.
   *
   * <p>Without this the memory layer called {@code TensorOps.ggufMatmul} directly, which for a
   * K-quant runs on the calling thread -- {@code packedProjectionThreads} only parallelises the
   * MXFP4 and PTQ1_0 float-activation kernels. That made the nine projections on each of the eight
   * memory layers single-threaded while the rest of the graph was free to use the Rust kernels, and
   * it dominated decode: EdgeRock measured 0.93 tok/s on the native route against 1.06 pure-Java,
   * i.e. the native kernels bought nothing, where an unrelated control model gained 3.9x. Routing
   * through the same seam {@code Qwen35ForwardPass.project} uses is what makes the memory layer
   * benefit from the kernels the rest of the model already uses.
   */
  private final GgufBatchedMatrixKernel kernel;

  public QuantizedProjection(
      MemorySegment data, GgufTensorType type, int rows, int columns, String name) {
    this(data, type, rows, columns, name, GgufBatchedMatrixKernel.none());
  }

  public QuantizedProjection(
      MemorySegment data,
      GgufTensorType type,
      int rows,
      int columns,
      String name,
      GgufBatchedMatrixKernel kernel) {
    this.kernel = Objects.requireNonNull(kernel, "kernel");
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
    // Batch size 1: the memory layer is sequential by construction, because each token reads the
    // bank before writing to it.
    if (kernel.isEligible(type, 1, rows, columns)) {
      kernel.multiply(output, input, data, type, 1, rows, columns);
      return;
    }
    if (quantizedActivation == null) {
      TensorOps.ggufMatmul(output, input, data, type, rows, columns);
      return;
    }
    TensorOps.ggufMatmul(
        output, input, data, type, rows, columns, quantizedActivation, quantizedActivationScales);
  }

  /**
   * Runs three projections of one shared input through a single activation preparation.
   *
   * <p>Borrowed from uzu, which projects the whole Gated DeltaNet input as one {@code in_proj} of
   * width {@code conv_dim + value_dim + 2 * num_v_heads} rather than as separate calls. The memory
   * layer has the same shape of opportunity: {@code q_proj}, {@code write_gate_proj} and {@code
   * write_up_proj} all read the normed state, all carry 2560 columns and all are Q4_K, so one
   * activation quantization can serve all three.
   *
   * @return whether the fused route ran; false means the caller must project individually, which is
   *     a capability fallback for an ineligible kernel and not a slower alternative to choose
   */
  public static boolean fusedTriple(
      Projection first,
      float[] firstOutput,
      Projection second,
      float[] secondOutput,
      Projection third,
      float[] thirdOutput,
      float[] input) {
    if (!(first instanceof QuantizedProjection a)
        || !(second instanceof QuantizedProjection b)
        || !(third instanceof QuantizedProjection c)) {
      return false;
    }
    // One shared activation means one shared input width, and one kernel to prepare it.
    if (a.columns != input.length || b.columns != input.length || c.columns != input.length) {
      return false;
    }
    if (a.kernel != b.kernel || a.kernel != c.kernel) {
      return false;
    }
    if (!a.kernel.isTripleEligible(a.type, a.rows, b.type, b.rows, c.type, c.rows, 1, a.columns)) {
      return false;
    }
    a.kernel.multiplyTriple(
        firstOutput,
        a.data,
        a.type,
        a.rows,
        secondOutput,
        b.data,
        b.type,
        b.rows,
        thirdOutput,
        c.data,
        c.type,
        c.rows,
        input,
        1,
        a.columns);
    return true;
  }

  /** Runs two projections of one shared input through a single activation preparation. */
  public static boolean fusedDual(
      Projection first,
      float[] firstOutput,
      Projection second,
      float[] secondOutput,
      float[] input) {
    if (!(first instanceof QuantizedProjection a) || !(second instanceof QuantizedProjection b)) {
      return false;
    }
    if (a.columns != input.length || b.columns != input.length || a.kernel != b.kernel) {
      return false;
    }
    if (!a.kernel.isDualEligible(a.type, a.rows, b.type, b.rows, 1, a.columns)) {
      return false;
    }
    a.kernel.multiplyDual(
        firstOutput,
        a.data,
        a.type,
        a.rows,
        secondOutput,
        b.data,
        b.type,
        b.rows,
        input,
        1,
        a.columns);
    return true;
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
