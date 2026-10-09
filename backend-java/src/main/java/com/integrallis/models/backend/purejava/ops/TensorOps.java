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
package com.integrallis.models.backend.purejava.ops;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.quant.Mxfp4Dequantizer;
import com.integrallis.models.backend.purejava.quant.Q4_1Dequantizer;
import com.integrallis.models.backend.purejava.quant.Q5_1Dequantizer;
import com.integrallis.vectors.core.BFloat16Matrix;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.GgufQ6BatchedKernel;
import com.integrallis.vectors.core.PanamaConstants;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/** Core tensor operations for transformer inference. */
public final class TensorOps {
  private static final VectorSpecies<Float> NORM_SPECIES =
      FloatVector.SPECIES_PREFERRED.vectorBitSize() <= PanamaConstants.MAX_BITS
          ? FloatVector.SPECIES_PREFERRED
          : VectorSpecies.of(float.class, VectorShape.forBitSize(PanamaConstants.MAX_BITS));

  private static final float TANH_TABLE_LIMIT = 10.0f;
  private static final int TANH_TABLE_SIZE = 1 << 16;
  private static final float TANH_TABLE_SCALE = TANH_TABLE_SIZE / (2.0f * TANH_TABLE_LIMIT);
  private static final float[] TANH_TABLE = createTanhTable();
  private static final ValueLayout.OfFloat LITTLE_ENDIAN_FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfShort LITTLE_ENDIAN_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  enum GroupedProjectionPlan {
    NONE,
    ALL,
    MIXED_Q4_K_Q4_K_Q6_K,
    FIRST_SECOND,
    FIRST_THIRD,
    SECOND_THIRD
  }

  private TensorOps() {}

  /** RMS normalization: out[i] = x[i] / rms(x) * weight[i]. */
  public static void rmsNorm(float[] out, float[] x, float[] weight, int size, float eps) {
    rmsNorm(out, 0, x, 0, weight, size, eps);
  }

  /** Offset-aware RMS normalization for attention heads stored in contiguous buffers. */
  public static void rmsNorm(
      float[] out, int outOffset, float[] x, int xOffset, float[] weight, int size, float eps) {
    // Host-independent: VectorUtil.dotProduct sizes its reduction from the host (four accumulators
    // of SPECIES_PREFERRED lanes, folded by a width switch), so this sum -- and therefore every
    // token this model selects -- differed between a 128-bit, 256-bit and 512-bit machine. See
    // PinnedReduction; at the 256-bit default this is bit-identical to what it replaces.
    float sumSq = PinnedReduction.sumOfSquares(x, xOffset, size);
    float rms = (float) Math.sqrt(sumSq / size + eps);
    float scale = 1.0f / rms;
    // (x * scale) * weight lanewise, the same two roundings in the same order as the scalar
    // loop, so the vector and tail paths and the pre-vectorization results are bit-identical.
    int lanes = NORM_SPECIES.length();
    int vectorLimit = NORM_SPECIES.loopBound(size);
    FloatVector scaleVector = FloatVector.broadcast(NORM_SPECIES, scale);
    int i = 0;
    for (; i < vectorLimit; i += lanes) {
      FloatVector.fromArray(NORM_SPECIES, x, xOffset + i)
          .mul(scaleVector)
          .mul(FloatVector.fromArray(NORM_SPECIES, weight, i))
          .intoArray(out, outOffset + i);
    }
    for (; i < size; i++) {
      out[outOffset + i] = x[xOffset + i] * scale * weight[i];
    }
  }

  /**
   * The bounded final-logit transform the Gemma family applies: {@code cap * tanh(logit / cap)}.
   *
   * <p>Shared because two architectures need the identical formula. Monotonic, so it cannot change
   * which token a greedy decode selects -- but it decides every logit's <i>value</i>, so anything
   * turning logits into probabilities, applying a temperature, or reporting a logprob is wrong
   * without it. gemma3n shipped without it: its published header carries no softcapping key at all,
   * and the reference's default of 30 applies, which was confirmed against the reference's own
   * graph dump (raw -15.6992 becoming -14.4074, and 30*tanh(-15.6992/30) = -14.41).
   */
  public static void softcap(float[] logits, float cap) {
    Objects.requireNonNull(logits, "logits");
    if (!(cap > 0.0f) || !Float.isFinite(cap)) {
      throw new IllegalArgumentException("cap must be finite and > 0: " + cap);
    }
    for (int index = 0; index < logits.length; index++) {
      logits[index] = cap * (float) Math.tanh(logits[index] / cap);
    }
  }

  /** Layer normalization with learned scale and bias over one contiguous row. */
  public static void layerNorm(
      float[] out,
      int outOffset,
      float[] x,
      int xOffset,
      float[] weight,
      float[] bias,
      int size,
      float eps) {
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(x, "x");
    Objects.requireNonNull(weight, "weight");
    Objects.requireNonNull(bias, "bias");
    Objects.checkFromIndexSize(outOffset, size, out.length);
    Objects.checkFromIndexSize(xOffset, size, x.length);
    if (weight.length != size || bias.length != size) {
      throw new IllegalArgumentException(
          "layer-norm scale and bias must match row size: "
              + weight.length
              + ", "
              + bias.length
              + " != "
              + size);
    }
    if (size == 0) {
      throw new IllegalArgumentException("layer-norm row must not be empty");
    }
    if (!(eps >= 0.0f) || !Float.isFinite(eps)) {
      throw new IllegalArgumentException("layer-norm epsilon must be finite and >= 0: " + eps);
    }

    float sum = 0.0f;
    for (int index = 0; index < size; index++) {
      sum += x[xOffset + index];
    }
    float mean = sum / size;
    float variance = 0.0f;
    for (int index = 0; index < size; index++) {
      float centered = x[xOffset + index] - mean;
      variance += centered * centered;
    }
    float scale = 1.0f / (float) Math.sqrt(variance / size + eps);
    for (int index = 0; index < size; index++) {
      float centered = x[xOffset + index] - mean;
      out[outOffset + index] = centered * scale * weight[index] + bias[index];
    }
  }

  /** Matrix-vector multiplication: out = weight * x where weight is [rows x cols] row-major. */
  public static void matmul(float[] out, float[] x, float[] weight, int rows, int cols) {
    VectorUtil.batchDotProduct(x, weight, rows, cols, out);
  }

  /**
   * Matrix-vector multiplication over a mapped GGUF tensor. Uses vectors-core fused kernels so rows
   * are not materialized as temporary F32 buffers.
   */
  public static void ggufMatmul(
      float[] out, float[] x, MemorySegment qWeight, GgufTensorType type, int rows, int cols) {
    if (type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q5_0
        || type == GgufTensorType.Q5_1
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K
        || type == GgufTensorType.Q6_K) {
      int activationBlockSize =
          type == GgufTensorType.Q4_K || type == GgufTensorType.Q5_K || type == GgufTensorType.Q6_K
              ? 256
              : 32;
      ggufMatmul(
          out,
          x,
          qWeight,
          type,
          rows,
          cols,
          new byte[cols],
          new float[cols / activationBlockSize],
          new int[(cols + 3) / 4],
          new short[(cols + 15) / 16],
          GgufQ4Kernel.WIDENED);
      return;
    }
    ggufMatmul(out, x, qWeight, type, rows, cols, null, null, null, null, GgufQ4Kernel.WIDENED);
  }

  /**
   * Matrix-vector multiplication with reusable Q8 activation scratch for GGML quantized kernels.
   */
  public static void ggufMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int rows,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales) {
    ggufMatmul(
        out,
        x,
        qWeight,
        type,
        rows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        new int[(cols + 3) / 4],
        type == GgufTensorType.Q4_K || type == GgufTensorType.Q5_K
            ? new short[(cols + 15) / 16]
            : null,
        GgufQ4Kernel.WIDENED);
  }

  /**
   * Matrix-vector product against MXFP4 weights.
   *
   * <p>Scalar and float-activation, unlike the K-quant kernels, which quantize the activation to Q8
   * and reduce in integer arithmetic. That is deliberate: MXFP4 arrives here to make the type
   * multipliable at all -- before this, an MXFP4 tensor could be decoded but not multiplied, and
   * {@code ggufMatmul} threw. Being correct first and fast second is the same order the MXFP4 and
   * ternary decoders were written in, and a Q8-activation MXFP4 kernel has to be proven identical
   * to this one, which needs this one to exist.
   *
   * <p>No per-row buffer: {@link Mxfp4Dequantizer#dotProduct} consumes the packed bytes directly,
   * so a row of any width costs nothing but the read.
   */
  private static void mxfp4Matmul(
      float[] out, float[] x, MemorySegment qWeight, int rows, int cols) {
    if (cols % Mxfp4Dequantizer.BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "MXFP4 row length must be a multiple of "
              + Mxfp4Dequantizer.BLOCK_SIZE
              + ", but was "
              + cols);
    }
    long rowBytes = (long) (cols / Mxfp4Dequantizer.BLOCK_SIZE) * Mxfp4Dequantizer.BLOCK_BYTES;
    int threads = mxfp4Threads(rows, rowBytes, qWeight);
    if (threads <= 1) {
      mxfp4Rows(out, x, qWeight, rowBytes, cols, 0, rows);
      return;
    }
    int chunk = (rows + threads - 1) / threads;
    List<Future<?>> pending = new ArrayList<>(threads);
    for (int first = chunk; first < rows; first += chunk) {
      int from = first;
      int to = Math.min(rows, first + chunk);
      pending.add(
          Mxfp4Threads.EXECUTOR.submit(() -> mxfp4Rows(out, x, qWeight, rowBytes, cols, from, to)));
    }
    // The calling thread takes the first chunk rather than waiting on all of them, so a matmul on a
    // one-core host does the same work with no handoff at all.
    mxfp4Rows(out, x, qWeight, rowBytes, cols, 0, Math.min(rows, chunk));
    for (Future<?> future : pending) {
      try {
        future.get();
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("Interrupted during an MXFP4 projection", interrupted);
      } catch (ExecutionException failed) {
        Throwable cause = failed.getCause();
        if (cause instanceof RuntimeException runtime) {
          throw runtime;
        }
        if (cause instanceof Error error) {
          throw error;
        }
        throw new IllegalStateException("MXFP4 projection failed", cause);
      }
    }
  }

  /** One contiguous band of output rows. Reads shared inputs, writes only {@code out[from..to)}. */
  private static void mxfp4Rows(
      float[] out, float[] x, MemorySegment qWeight, long rowBytes, int cols, int from, int to) {
    for (int row = from; row < to; row++) {
      out[row] = Mxfp4Dequantizer.dotProduct(qWeight, row * rowBytes, x, 0, cols);
    }
  }

  /**
   * Single-token projection for the quantizations that carry a per-block minimum.
   *
   * <p>Deliberately the batched routine at a batch of one rather than a hand-written scalar dot
   * product. A {@code q * d + m} format can be folded as {@code d * sum(q*x) + m * sum(x)}, which
   * is a different fold order from {@link PinnedReduction#dot}, so a decoder and an embedder
   * running the same weights would have disagreed in the last bits depending on which dispatch they
   * took. This codebase has already paid for a reduction whose order depended on something other
   * than the data; see the width switch {@code PinnedReduction} exists to remove. One routine for
   * both batch sizes makes them agree exactly rather than closely.
   */
  private static void exactSingleTokenMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int rows,
      int cols,
      int blockSize) {
    if (cols % blockSize != 0) {
      throw new IllegalArgumentException(
          type + " row length must be a multiple of " + blockSize + ", but was " + cols);
    }
    ggufExactBatchedMatmul(out, x, qWeight, type, 1, rows, cols, new float[cols]);
  }

  /**
   * How many threads to split an MXFP4 projection across, 1 meaning "stay on the calling thread".
   *
   * <p>Thread count cannot change the result. Each output element is its own dot product over its
   * own row, so no two threads contribute to one reduction and {@link Mxfp4Dequantizer#dotProduct}
   * folds each row in the same order it always did. That is what makes this safe in a backend where
   * a float reduction's order is pinned on purpose: this splits <i>which thread</i> runs a
   * reduction, never <i>how</i> one is folded. A test asserts the parallel and serial results are
   * bit-identical rather than close.
   *
   * <p>Only above a size threshold, because handing work to another thread costs more than a small
   * projection takes. The threshold is in weight bytes read, which is what the work actually is.
   */
  static int mxfp4Threads(int rows, long rowBytes, MemorySegment weight) {
    if (rows < 2 || Math.multiplyExact(rows, rowBytes) < MXFP4_PARALLEL_MIN_BYTES) {
      return 1;
    }
    // A segment from a confined arena is readable only by the thread that allocated it, and handing
    // its rows to a pool thread throws WrongThreadException from inside the dot product rather than
    // returning a wrong number. The probe is never started and never runs: asking whether some
    // OTHER
    // thread could read this segment is exactly the question, since a confined scope answers no for
    // every thread but its owner while a shared or global one answers yes for all of them. This is
    // the same property the execution planner already gates thread sharing on.
    if (!weight.isAccessibleBy(Mxfp4Threads.ACCESS_PROBE)) {
      return 1;
    }
    return Math.min(rows, Mxfp4Threads.COUNT);
  }

  /**
   * Weight bytes below which an MXFP4 projection stays on the calling thread.
   *
   * <p>Matches vectors-core's own {@code ggufParallelThreshold} of one mebibyte, so MXFP4 and the
   * K-quant kernels start using threads at the same amount of work rather than at two unrelated
   * sizes.
   */
  private static final long MXFP4_PARALLEL_MIN_BYTES = 1L << 20;

  /**
   * The pool, created on first use by an MXFP4 model and never by any other.
   *
   * <p>A holder class so that a process which loads no MXFP4 weights -- every K-quant model, which
   * is nearly all of them -- starts no threads at all. Daemon threads, so this never holds up JVM
   * exit.
   */
  private static final class Mxfp4Threads {
    static final int COUNT = Math.max(1, Runtime.getRuntime().availableProcessors());

    /**
     * A thread that is never started, used only to ask a segment whether a thread other than the
     * caller could read it. Deliberately not one of the pool threads: this question has to be
     * answerable before any work is submitted.
     */
    static final Thread ACCESS_PROBE = new Thread(() -> {}, "mxfp4-access-probe");

    static final ExecutorService EXECUTOR =
        Executors.newFixedThreadPool(
            COUNT - 1 > 0 ? COUNT - 1 : 1,
            runnable -> {
              Thread thread = new Thread(runnable, "mxfp4-projection");
              thread.setDaemon(true);
              return thread;
            });

    private Mxfp4Threads() {}
  }

  /**
   * F32 projection with one float reduction order in every JIT tier.
   *
   * <p>Each mapped weight row is copied to a reused heap row and scored with {@link
   * VectorUtil#dotProduct(float[], int, float[], int, int)}, whose lane sums are written out in a
   * fixed order. vectors-core's mapped F32 GEMV ({@code VectorUtil.ggufF32BatchDotProduct}) reduces
   * with {@code reduceLanes(ADD)}, whose float order differs between the interpreted/C1 fallback
   * and the C2 intrinsic: the same projection changed its low bits once C2 compiled it, which made
   * logits depend on warm-up and broke exact prefill comparisons (measured in
   * benchmark-results/2026-09-16-flaky-injected-attention).
   */
  private static void f32Matmul(float[] out, float[] x, MemorySegment weight, int rows, int cols) {
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(x, "x");
    Objects.requireNonNull(weight, "weight");
    if (x.length < cols
        || out.length < rows
        || weight.byteSize() < (long) rows * cols * Float.BYTES) {
      throw new IllegalArgumentException(
          "F32 projection dimensions do not match: rows="
              + rows
              + ", cols="
              + cols
              + ", input="
              + x.length
              + ", output="
              + out.length
              + ", weightBytes="
              + weight.byteSize());
    }
    float[] row = new float[cols];
    for (int r = 0; r < rows; r++) {
      MemorySegment.copy(weight, LITTLE_ENDIAN_FLOAT, (long) r * cols * Float.BYTES, row, 0, cols);
      // Pinned for the same reason as rmsNorm: an F32 row score feeds token selection.
      out[r] = PinnedReduction.dot(x, 0, row, 0, cols);
    }
  }

  /** Matrix-vector multiplication with caller-owned scratch and an explicit Q4 policy. */
  public static void ggufMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int rows,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      GgufQ4Kernel q4Kernel) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    switch (type) {
      case F32 -> f32Matmul(out, x, qWeight, rows, cols);
      // The same routine the batched path uses, at a batch of one, so the two cannot disagree. F16
      // was reachable only in batches until Gemma 4 E4B arrived carrying per_layer_model_proj as
      // F16
      // where the E2B file carries it as BF16 -- and the per-layer projection runs a token at a
      // time.
      case F16 -> multiplyF16Batch(out, x, qWeight, 1, rows, cols);
      case BF16 -> BFloat16Matrix.of(qWeight, rows, cols).multiply(x, out);
      case Q4_0 ->
          VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
              x,
              qWeight,
              rows,
              cols,
              out,
              quantizedActivation,
              quantizedActivationScales,
              quantizedActivationZeroPointCorrections,
              q4Kernel);
      case Q5_0 ->
          VectorUtil.ggufQ5_0Q8_0BatchDotProduct(
              x, qWeight, rows, cols, out, quantizedActivation, quantizedActivationScales);
      case Q8_0 ->
          VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
              x, qWeight, rows, cols, out, quantizedActivation, quantizedActivationScales);
      case Q4_K ->
          VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
              x,
              qWeight,
              rows,
              cols,
              out,
              quantizedActivation,
              quantizedActivationScales,
              quantizedActivationSums);
      case Q5_K ->
          VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
              x,
              qWeight,
              rows,
              cols,
              out,
              quantizedActivation,
              quantizedActivationScales,
              quantizedActivationSums);
      case Q6_K ->
          VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
              x, qWeight, rows, cols, out, quantizedActivation, quantizedActivationScales);
      case MXFP4 -> mxfp4Matmul(out, x, qWeight, rows, cols);
      // Exact, like the batched Q5_1 path: the reference's integer kernel pairs Q5_1 with Q8_1
      // activations because it needs their block sums for the minimum term, and nothing on this
      // side carries a Q8_1 block sum. Dequantizing the row costs the same read either way and
      // skips the activation quantization rather than approximating it.
      case Q4_1 ->
          exactSingleTokenMatmul(out, x, qWeight, type, rows, cols, Q4_1Dequantizer.BLOCK_SIZE);
      case Q5_1 ->
          exactSingleTokenMatmul(out, x, qWeight, type, rows, cols, Q5_1Dequantizer.BLOCK_SIZE);
      default -> throw new UnsupportedOperationException("GGUF matmul not supported for: " + type);
    }
  }

  /** Two projections sharing quantization and row dispatch under an explicit Q4 policy. */
  public static void ggufDualMatmul(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] input,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      GgufQ4Kernel q4Kernel) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    if (firstType == secondType && supportsGroupedMatmul(firstType)) {
      switch (firstType) {
        case Q4_0 ->
            VectorUtil.ggufQ4_0Q8_0DualBatchDotProduct(
                input,
                firstWeight,
                firstRows,
                firstOut,
                secondWeight,
                secondRows,
                secondOut,
                cols,
                quantizedActivation,
                quantizedActivationScales,
                quantizedActivationZeroPointCorrections,
                q4Kernel);
        case Q8_0 ->
            VectorUtil.ggufQ8_0Q8_0DualBatchDotProduct(
                input,
                firstWeight,
                firstRows,
                firstOut,
                secondWeight,
                secondRows,
                secondOut,
                cols,
                quantizedActivation,
                quantizedActivationScales);
        case Q4_K ->
            VectorUtil.ggufQ4_KQ8_KDualBatchDotProduct(
                input,
                firstWeight,
                firstRows,
                firstOut,
                secondWeight,
                secondRows,
                secondOut,
                cols,
                quantizedActivation,
                quantizedActivationScales,
                quantizedActivationSums);
        case Q5_K ->
            VectorUtil.ggufQ5_KQ8_KDualBatchDotProduct(
                input,
                firstWeight,
                firstRows,
                firstOut,
                secondWeight,
                secondRows,
                secondOut,
                cols,
                quantizedActivation,
                quantizedActivationScales,
                quantizedActivationSums);
        default -> throw new IllegalStateException("Unsupported grouped matmul type: " + firstType);
      }
      return;
    }

    ggufMatmul(
        firstOut,
        input,
        firstWeight,
        firstType,
        firstRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
    ggufMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        secondRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
  }

  /** Three projections with an explicit Q4 arithmetic policy. */
  public static void ggufTripleMatmul(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOut,
      MemorySegment thirdWeight,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      GgufQ4Kernel q4Kernel) {
    ggufTripleMatmul(
        firstOut,
        firstWeight,
        firstType,
        firstRows,
        secondOut,
        secondWeight,
        secondType,
        secondRows,
        thirdOut,
        thirdWeight,
        thirdType,
        thirdRows,
        input,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel,
        true);
  }

  /** Triple projection with explicit Q4 and heterogeneous K-quant policies. */
  public static void ggufTripleMatmul(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOut,
      MemorySegment thirdWeight,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      GgufQ4Kernel q4Kernel,
      boolean mixedKProjections) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    GroupedProjectionPlan projectionPlan = groupedProjectionPlan(firstType, secondType, thirdType);
    if (!mixedKProjections && projectionPlan == GroupedProjectionPlan.MIXED_Q4_K_Q4_K_Q6_K) {
      projectionPlan = GroupedProjectionPlan.FIRST_SECOND;
    }
    switch (projectionPlan) {
      case MIXED_Q4_K_Q4_K_Q6_K -> {
        VectorUtil.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchDotProduct(
            input,
            firstWeight,
            firstRows,
            firstOut,
            secondWeight,
            secondRows,
            secondOut,
            thirdWeight,
            thirdRows,
            thirdOut,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationSums);
        return;
      }
      case ALL -> {
        switch (firstType) {
          case Q4_0 ->
              VectorUtil.ggufQ4_0Q8_0TripleBatchDotProduct(
                  input,
                  firstWeight,
                  firstRows,
                  firstOut,
                  secondWeight,
                  secondRows,
                  secondOut,
                  thirdWeight,
                  thirdRows,
                  thirdOut,
                  cols,
                  quantizedActivation,
                  quantizedActivationScales,
                  quantizedActivationZeroPointCorrections,
                  q4Kernel);
          case Q4_K ->
              VectorUtil.ggufQ4_KQ8_KTripleBatchDotProduct(
                  input,
                  firstWeight,
                  firstRows,
                  firstOut,
                  secondWeight,
                  secondRows,
                  secondOut,
                  thirdWeight,
                  thirdRows,
                  thirdOut,
                  cols,
                  quantizedActivation,
                  quantizedActivationScales,
                  quantizedActivationSums);
          case Q5_K ->
              VectorUtil.ggufQ5_KQ8_KTripleBatchDotProduct(
                  input,
                  firstWeight,
                  firstRows,
                  firstOut,
                  secondWeight,
                  secondRows,
                  secondOut,
                  thirdWeight,
                  thirdRows,
                  thirdOut,
                  cols,
                  quantizedActivation,
                  quantizedActivationScales,
                  quantizedActivationSums);
          default ->
              throw new IllegalStateException("Unsupported grouped matmul type: " + firstType);
        }
        return;
      }
      case FIRST_SECOND -> {
        ggufDualMatmul(
            firstOut,
            firstWeight,
            firstType,
            firstRows,
            secondOut,
            secondWeight,
            secondType,
            secondRows,
            input,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        ggufMatmul(
            thirdOut,
            input,
            thirdWeight,
            thirdType,
            thirdRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        return;
      }
      case FIRST_THIRD -> {
        ggufDualMatmul(
            firstOut,
            firstWeight,
            firstType,
            firstRows,
            thirdOut,
            thirdWeight,
            thirdType,
            thirdRows,
            input,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        ggufMatmul(
            secondOut,
            input,
            secondWeight,
            secondType,
            secondRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        return;
      }
      case SECOND_THIRD -> {
        ggufDualMatmul(
            secondOut,
            secondWeight,
            secondType,
            secondRows,
            thirdOut,
            thirdWeight,
            thirdType,
            thirdRows,
            input,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        ggufMatmul(
            firstOut,
            input,
            firstWeight,
            firstType,
            firstRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        return;
      }
      case NONE -> {
        // Fall through to independent format-specific projections.
      }
    }

    ggufMatmul(
        firstOut,
        input,
        firstWeight,
        firstType,
        firstRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
    ggufMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        secondRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
    ggufMatmul(
        thirdOut,
        input,
        thirdWeight,
        thirdType,
        thirdRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
  }

  /** Two batched projections sharing quantization and row dispatch under an explicit Q4 policy. */
  public static void ggufDualBatchedMatmul(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] input,
      int batchSize,
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel) {
    ggufDualBatchedMatmul(
        firstOut,
        firstWeight,
        firstType,
        firstRows,
        secondOut,
        secondWeight,
        secondType,
        secondRows,
        input,
        batchSize,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK);
  }

  /** Two batched projections with explicit Q4 and Q6_K kernel policies. */
  public static void ggufDualBatchedMatmul(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] input,
      int batchSize,
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel,
      GgufQ6BatchedKernel q6BatchedKernel) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    Objects.requireNonNull(q6BatchedKernel, "q6BatchedKernel");
    if (firstType == secondType) {
      switch (firstType) {
        case Q4_0 -> {
          VectorUtil.ggufQ4_0Q8_0DualBatchedMatmul(
              input,
              firstWeight,
              firstRows,
              firstOut,
              secondWeight,
              secondRows,
              secondOut,
              batchSize,
              cols,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationZeroPointCorrections,
              q4LaneScratch,
              q4Kernel);
          return;
        }
        case Q4_K -> {
          VectorUtil.ggufQ4_KQ8_KDualBatchedMatmul(
              input,
              firstWeight,
              firstRows,
              firstOut,
              secondWeight,
              secondRows,
              secondOut,
              batchSize,
              cols,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationSums);
          return;
        }
        case Q8_0 -> {
          VectorUtil.ggufQ8_0Q8_0DualBatchedMatmul(
              input,
              firstWeight,
              firstRows,
              firstOut,
              secondWeight,
              secondRows,
              secondOut,
              batchSize,
              cols,
              quantizedActivations,
              quantizedActivationScales);
          return;
        }
        default -> {
          // Fall through to independent batched projections.
        }
      }
    }

    ggufBatchedMatmul(
        firstOut,
        input,
        firstWeight,
        firstType,
        batchSize,
        firstRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
    ggufBatchedMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        batchSize,
        secondRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
  }

  /** Three batched projections with explicit Q4 and heterogeneous K-quant policies. */
  public static void ggufTripleBatchedMatmul(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOut,
      MemorySegment thirdWeight,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int batchSize,
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel,
      boolean mixedKProjections) {
    ggufTripleBatchedMatmul(
        firstOut,
        firstWeight,
        firstType,
        firstRows,
        secondOut,
        secondWeight,
        secondType,
        secondRows,
        thirdOut,
        thirdWeight,
        thirdType,
        thirdRows,
        input,
        batchSize,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
        mixedKProjections);
  }

  /** Three batched projections with explicit Q4, Q6_K, and heterogeneous K-quant policies. */
  public static void ggufTripleBatchedMatmul(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOut,
      MemorySegment thirdWeight,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int batchSize,
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel,
      GgufQ6BatchedKernel q6BatchedKernel,
      boolean mixedKProjections) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    Objects.requireNonNull(q6BatchedKernel, "q6BatchedKernel");
    GroupedProjectionPlan projectionPlan = groupedProjectionPlan(firstType, secondType, thirdType);
    if (!mixedKProjections && projectionPlan == GroupedProjectionPlan.MIXED_Q4_K_Q4_K_Q6_K) {
      projectionPlan = GroupedProjectionPlan.FIRST_SECOND;
    }
    if (projectionPlan == GroupedProjectionPlan.MIXED_Q4_K_Q4_K_Q6_K) {
      VectorUtil.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
          input,
          firstWeight,
          firstRows,
          firstOut,
          secondWeight,
          secondRows,
          secondOut,
          thirdWeight,
          thirdRows,
          thirdOut,
          batchSize,
          cols,
          quantizedActivations,
          quantizedActivationScales,
          quantizedActivationSums,
          q6BatchedKernel);
      return;
    }

    switch (projectionPlan) {
      case ALL -> {
        if (firstType == GgufTensorType.Q4_0) {
          VectorUtil.ggufQ4_0Q8_0TripleBatchedMatmul(
              input,
              firstWeight,
              firstRows,
              firstOut,
              secondWeight,
              secondRows,
              secondOut,
              thirdWeight,
              thirdRows,
              thirdOut,
              batchSize,
              cols,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationZeroPointCorrections,
              q4LaneScratch,
              q4Kernel);
          return;
        }
        ggufDualBatchedMatmul(
            firstOut,
            firstWeight,
            firstType,
            firstRows,
            secondOut,
            secondWeight,
            secondType,
            secondRows,
            input,
            batchSize,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        ggufBatchedMatmul(
            thirdOut,
            input,
            thirdWeight,
            thirdType,
            batchSize,
            thirdRows,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        return;
      }
      case FIRST_SECOND -> {
        ggufDualBatchedMatmul(
            firstOut,
            firstWeight,
            firstType,
            firstRows,
            secondOut,
            secondWeight,
            secondType,
            secondRows,
            input,
            batchSize,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        ggufBatchedMatmul(
            thirdOut,
            input,
            thirdWeight,
            thirdType,
            batchSize,
            thirdRows,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        return;
      }
      case FIRST_THIRD -> {
        ggufDualBatchedMatmul(
            firstOut,
            firstWeight,
            firstType,
            firstRows,
            thirdOut,
            thirdWeight,
            thirdType,
            thirdRows,
            input,
            batchSize,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        ggufBatchedMatmul(
            secondOut,
            input,
            secondWeight,
            secondType,
            batchSize,
            secondRows,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        return;
      }
      case SECOND_THIRD -> {
        ggufDualBatchedMatmul(
            secondOut,
            secondWeight,
            secondType,
            secondRows,
            thirdOut,
            thirdWeight,
            thirdType,
            thirdRows,
            input,
            batchSize,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        ggufBatchedMatmul(
            firstOut,
            input,
            firstWeight,
            firstType,
            batchSize,
            firstRows,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        return;
      }
      case NONE -> {
        // Fall through to independent format-specific projections.
      }
      case MIXED_Q4_K_Q4_K_Q6_K -> throw new AssertionError("mixed K projection not handled");
    }

    ggufBatchedMatmul(
        firstOut,
        input,
        firstWeight,
        firstType,
        batchSize,
        firstRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
    ggufBatchedMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        batchSize,
        secondRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
    ggufBatchedMatmul(
        thirdOut,
        input,
        thirdWeight,
        thirdType,
        batchSize,
        thirdRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
  }

  /**
   * Returns whether equal-format projections can share activation quantization and row dispatch.
   */
  public static boolean supportsGroupedMatmul(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K;
  }

  /** Returns whether the format has a retained multi-projection batched prefill kernel. */
  public static boolean supportsGroupedBatchedMatmul(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K;
  }

  /** Returns whether three equal-format projections can share one row dispatch. */
  public static boolean supportsGroupedTripleMatmul(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K;
  }

  /** Returns whether three projection formats can share one activation and row dispatch. */
  public static boolean supportsGroupedTripleMatmul(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    return (firstType == secondType
            && firstType == thirdType
            && supportsGroupedTripleMatmul(firstType))
        || (firstType == GgufTensorType.Q4_K
            && secondType == GgufTensorType.Q4_K
            && thirdType == GgufTensorType.Q6_K);
  }

  static GroupedProjectionPlan groupedProjectionPlan(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    if (firstType == GgufTensorType.Q4_K
        && secondType == GgufTensorType.Q4_K
        && thirdType == GgufTensorType.Q6_K) {
      return GroupedProjectionPlan.MIXED_Q4_K_Q4_K_Q6_K;
    }
    if (supportsGroupedTripleMatmul(firstType)) {
      if (firstType == secondType && firstType == thirdType) {
        return GroupedProjectionPlan.ALL;
      }
      if (firstType == secondType) {
        return GroupedProjectionPlan.FIRST_SECOND;
      }
      if (firstType == thirdType) {
        return GroupedProjectionPlan.FIRST_THIRD;
      }
    }
    if (secondType == thirdType && supportsGroupedTripleMatmul(secondType)) {
      return GroupedProjectionPlan.SECOND_THIRD;
    }
    return GroupedProjectionPlan.NONE;
  }

  /** Returns whether the mapped tensor type has a weight-reusing batched prefill kernel. */
  public static boolean supportsBatchedMatmul(GgufTensorType type) {
    return type == GgufTensorType.F16
        || type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q4_1
        || type == GgufTensorType.BF16
        || type == GgufTensorType.Q5_0
        || type == GgufTensorType.Q5_1
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K
        || type == GgufTensorType.Q6_K;
  }

  /**
   * Exact mapped-weight batched projection for architectures whose output equivalence matters more
   * than Q8 activation reuse.
   *
   * <p>The supplied row buffer is reused for every weight row; the method never expands the model
   * weights into heap memory. Unlike the fast GGML Qx×Q8 path, activations remain F32.
   */
  public static void ggufExactBatchedMatmul(
      float[] output,
      float[] input,
      MemorySegment weights,
      GgufTensorType type,
      int batchSize,
      int rows,
      int columns,
      float[] decodedWeightRow) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(type, "type");
    Objects.requireNonNull(decodedWeightRow, "decodedWeightRow");
    if (decodedWeightRow.length < columns) {
      throw new IllegalArgumentException(
          "decoded weight row is too small: " + decodedWeightRow.length + " < " + columns);
    }
    for (int row = 0; row < rows; row++) {
      GgufTensorValues.dequantizeRow(weights, type, row, columns, decodedWeightRow);
      for (int batch = 0; batch < batchSize; batch++) {
        output[batch * rows + row] =
            PinnedReduction.dot(input, batch * columns, decodedWeightRow, 0, columns);
      }
    }
  }

  /** Batched matrix multiplication with caller-owned scratch and an explicit Q4 policy. */
  public static void ggufBatchedMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel) {
    ggufBatchedMatmul(
        out,
        x,
        qWeight,
        type,
        batchSize,
        rows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK);
  }

  /** Batched matrix multiplication with explicit Q4 and Q6_K kernel policies. */
  public static void ggufBatchedMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel,
      GgufQ6BatchedKernel q6BatchedKernel) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    Objects.requireNonNull(q6BatchedKernel, "q6BatchedKernel");
    if (!supportsBatchedMatmul(type)) {
      throw new UnsupportedOperationException("GGUF batched matmul not supported for: " + type);
    }
    switch (type) {
      case F16 -> multiplyF16Batch(out, x, qWeight, batchSize, rows, cols);
      case BF16 -> BFloat16Matrix.of(qWeight, rows, cols).multiplyBatch(x, 0, batchSize, out, 0);
      case Q4_0 ->
          VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationZeroPointCorrections,
              q4LaneScratch,
              q4Kernel);
      case Q5_0 ->
          VectorUtil.ggufQ5_0Q8_0BatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales);
      case Q8_0 ->
          VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales);
      case Q4_K ->
          VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationSums);
      case Q5_K ->
          VectorUtil.ggufQ5_KQ8_KBatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationSums);
      case Q6_K ->
          VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales,
              q6BatchedKernel);
      // Q5_1 carries a per-block minimum rather than a centred quant, so it has no Qx-by-Q8
      // kernel: the Q8 activation path folds a bias it cannot express. It therefore takes the
      // exact mapped-weight route, dequantizing one weight row at a time against F32 activations.
      // Correct and allocation-free, and it is what makes the mixed-quantization embedding
      // artifacts loadable at all -- all-MiniLM-L6-v2 and granite-embedding-107m carry Q5_1
      // attention tensors at Q4_K_S, Q5_K_S and Q5_K_M.
      case Q4_1, Q5_1 ->
          ggufExactBatchedMatmul(out, x, qWeight, type, batchSize, rows, cols, new float[cols]);
      default -> throw new AssertionError("unhandled batched matmul type: " + type);
    }
  }

  /** Multiplies mapped IEEE-754 half weights without materializing a full F32 copy. */
  private static void multiplyF16Batch(
      float[] output, float[] input, MemorySegment weights, int batchSize, int rows, int columns) {
    for (int batch = 0; batch < batchSize; batch++) {
      int inputOffset = batch * columns;
      int outputOffset = batch * rows;
      for (int row = 0; row < rows; row++) {
        long weightOffset = (long) row * columns * Short.BYTES;
        float sum = 0.0f;
        for (int column = 0; column < columns; column++) {
          sum +=
              Float.float16ToFloat(
                      weights.get(LITTLE_ENDIAN_SHORT, weightOffset + (long) column * Short.BYTES))
                  * input[inputOffset + column];
        }
        output[outputOffset + row] = sum;
      }
    }
  }

  /** Matrix-vector multiplication with a quantized GGUF weight. */
  public static void quantizedMatmul(
      float[] out, float[] x, MemorySegment qWeight, GgufTensorType type, int rows, int cols) {
    ggufMatmul(out, x, qWeight, type, rows, cols);
  }

  /** In-place numerically stable softmax over x[offset..offset+size). */
  public static void softmax(float[] x, int offset, int size) {
    Objects.requireNonNull(x, "x");
    if (size <= 0) {
      throw new IllegalArgumentException("size must be positive: " + size);
    }
    Objects.checkFromIndexSize(offset, size, x.length);
    float max = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < size; i++) {
      float value = x[offset + i];
      if (Float.isNaN(value)) {
        throw new IllegalArgumentException("softmax input contains NaN at index " + (offset + i));
      }
      if (value == Float.POSITIVE_INFINITY) {
        throw new IllegalArgumentException(
            "softmax input contains positive infinity at index " + (offset + i));
      }
      if (value > max) {
        max = value;
      }
    }
    if (!Float.isFinite(max)) {
      throw new IllegalArgumentException("softmax requires at least one finite input");
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
      x[offset + i] = (float) Math.exp(x[offset + i] - max);
      sum += x[offset + i];
    }
    float invSum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
      x[offset + i] *= invSum;
    }
  }

  /**
   * In-place stable softmax for attention scores with one learned sink logit whose value vector is
   * zero. The returned scores therefore sum to at most one; the remaining probability belongs to
   * the sink and contributes no value to the attention output.
   */
  public static void softmaxWithZeroValueSink(
      float[] scores, int offset, int size, float sinkLogit) {
    Objects.requireNonNull(scores, "scores");
    if (size <= 0) {
      throw new IllegalArgumentException("size must be positive: " + size);
    }
    Objects.checkFromIndexSize(offset, size, scores.length);
    if (!Float.isFinite(sinkLogit)) {
      throw new IllegalArgumentException("sink logit must be finite: " + sinkLogit);
    }

    float maximum = sinkLogit;
    for (int index = 0; index < size; index++) {
      float score = scores[offset + index];
      if (Float.isNaN(score)) {
        throw new IllegalArgumentException(
            "attention score contains NaN at index " + (offset + index));
      }
      if (score == Float.POSITIVE_INFINITY) {
        throw new IllegalArgumentException(
            "attention score contains positive infinity at index " + (offset + index));
      }
      maximum = Math.max(maximum, score);
    }

    float denominator = (float) Math.exp(sinkLogit - maximum);
    for (int index = 0; index < size; index++) {
      float probability = (float) Math.exp(scores[offset + index] - maximum);
      scores[offset + index] = probability;
      denominator += probability;
    }
    float inverseDenominator = 1.0f / denominator;
    for (int index = 0; index < size; index++) {
      scores[offset + index] *= inverseDenominator;
    }
  }

  /** In-place rotary position embedding on q and k vectors. */
  public static void rope(float[] q, float[] k, int position, int headDim, float ropeTheta) {
    rope(q, 0, k, 0, position, headDim, ropeTheta);
  }

  /** In-place rotary position embedding on q and k sub-vectors. */
  public static void rope(
      float[] q, int qOffset, float[] k, int kOffset, int position, int headDim, float ropeTheta) {
    rope(q, qOffset, k, kOffset, position, headDim, ropeTheta, 1.0f);
  }

  /** Offset-aware rotary embedding with a GGUF frequency scale. */
  public static void rope(
      float[] q,
      int qOffset,
      float[] k,
      int kOffset,
      int position,
      int headDim,
      float ropeTheta,
      float frequencyScale) {
    float scaledPosition = position * frequencyScale;
    for (int i = 0; i < headDim; i += 2) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) i / headDim));
      float angle = scaledPosition * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotatePair(q, qOffset + i, cos, sin);
      rotatePair(k, kOffset + i, cos, sin);
    }
  }

  /** In-place rotary position embedding on one sub-vector. */
  public static void rope(float[] vector, int offset, int position, int headDim, float ropeTheta) {
    rope(vector, offset, position, headDim, ropeTheta, 1.0f);
  }

  /** In-place rotary position embedding on one sub-vector with a GGUF frequency scale. */
  public static void rope(
      float[] vector,
      int offset,
      int position,
      int headDim,
      float ropeTheta,
      float frequencyScale) {
    float scaledPosition = position * frequencyScale;
    for (int i = 0; i < headDim; i += 2) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) i / headDim));
      float angle = scaledPosition * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotatePair(vector, offset + i, cos, sin);
    }
  }

  /** Applies standard rotary embedding using caller-precomputed pair factors. */
  public static void rope(float[] vector, int offset, float[] cosine, float[] sine) {
    rope(vector, offset, cosine, sine, 0, cosine.length);
  }

  /** Applies standard rotary embedding from an offset in precomputed pair factors. */
  public static void rope(
      float[] vector,
      int vectorOffset,
      float[] cosine,
      float[] sine,
      int factorOffset,
      int pairCount) {
    for (int pair = 0; pair < pairCount; pair++) {
      rotatePair(
          vector, vectorOffset + pair * 2, cosine[factorOffset + pair], sine[factorOffset + pair]);
    }
  }

  /** In-place NeoX rotary embedding whose coordinate pairs are separated by half a head. */
  public static void ropeNeox(
      float[] vector, int offset, int position, int headDim, float ropeTheta) {
    ropeNeox(vector, offset, position, headDim, ropeTheta, 1.0f);
  }

  /** In-place NeoX rotary embedding with a GGUF frequency scale. */
  public static void ropeNeox(
      float[] vector,
      int offset,
      int position,
      int headDim,
      float ropeTheta,
      float frequencyScale) {
    int half = headDim / 2;
    float scaledPosition = position * frequencyScale;
    for (int i = 0; i < half; i++) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) (2 * i) / headDim));
      float angle = scaledPosition * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotateSplitPair(vector, offset + i, offset + half + i, cos, sin);
    }
  }

  /** Applies NeoX rotary embedding using caller-precomputed pair factors. */
  public static void ropeNeox(float[] vector, int offset, float[] cosine, float[] sine) {
    ropeNeox(vector, offset, cosine, sine, 0, cosine.length);
  }

  /** Applies NeoX rotary embedding from an offset in precomputed pair factors. */
  public static void ropeNeox(
      float[] vector,
      int vectorOffset,
      float[] cosine,
      float[] sine,
      int factorOffset,
      int pairCount) {
    for (int pair = 0; pair < pairCount; pair++) {
      rotateSplitPair(
          vector,
          vectorOffset + pair,
          vectorOffset + pairCount + pair,
          cosine[factorOffset + pair],
          sine[factorOffset + pair]);
    }
  }

  /** SwiGLU activation: out[i] = silu(gate[i]) * up[i]. */
  public static void swiGlu(float[] out, float[] gate, float[] up, int size) {
    swiGlu(out, 0, gate, 0, up, 0, size);
  }

  /** Offset-aware SwiGLU activation over flat batch buffers. */
  public static void swiGlu(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int size) {
    for (int i = 0; i < size; i++) {
      float x = gate[gateOffset + i];
      float silu = x / (1.0f + (float) Math.exp(-x));
      out[outOffset + i] = silu * up[upOffset + i];
    }
  }

  /** GELU-gated activation using llama.cpp's tanh approximation. */
  public static void geluGlu(float[] out, float[] gate, float[] up, int size) {
    geluGlu(out, 0, gate, 0, up, 0, size);
  }

  /** GELU activation using llama.cpp's tanh approximation. */
  public static void gelu(float[] out, int outOffset, float[] input, int inputOffset, int size) {
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(input, "input");
    Objects.checkFromIndexSize(outOffset, size, out.length);
    Objects.checkFromIndexSize(inputOffset, size, input.length);
    for (int index = 0; index < size; index++) {
      float value = input[inputOffset + index];
      out[outOffset + index] =
          0.5f
              * value
              * (1.0f
                  + tableTanh(0.7978845608028654f * value * (1.0f + 0.044715f * value * value)));
    }
  }

  /** GELU activation using the erf formulation used by BERT's reference implementation. */
  public static void geluErf(float[] out, int outOffset, float[] input, int inputOffset, int size) {
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(input, "input");
    Objects.checkFromIndexSize(outOffset, size, out.length);
    Objects.checkFromIndexSize(inputOffset, size, input.length);
    for (int index = 0; index < size; index++) {
      float value = input[inputOffset + index];
      out[outOffset + index] = (float) (0.5 * value * (1.0 + erf(value * 0.7071067811865476)));
    }
  }

  /** High-accuracy error-function approximation; maximum error is about 1.2e-7. */
  private static double erf(double value) {
    if (value == 0.0) {
      return value;
    }
    double absolute = Math.abs(value);
    double t = 1.0 / (1.0 + 0.5 * absolute);
    double polynomial = 0.17087277;
    polynomial = -0.82215223 + t * polynomial;
    polynomial = 1.48851587 + t * polynomial;
    polynomial = -1.13520398 + t * polynomial;
    polynomial = 0.27886807 + t * polynomial;
    polynomial = -0.18628806 + t * polynomial;
    polynomial = 0.09678418 + t * polynomial;
    polynomial = 0.37409196 + t * polynomial;
    polynomial = 1.00002368 + t * polynomial;
    double tail = t * Math.exp(-absolute * absolute - 1.26551223 + t * polynomial);
    return value > 0.0 ? 1.0 - tail : tail - 1.0;
  }

  /** Offset-aware GELU-gated activation over flat batch buffers. */
  public static void geluGlu(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int size) {
    for (int index = 0; index < size; index++) {
      float value = gate[gateOffset + index];
      float gelu =
          0.5f
              * value
              * (1.0f
                  + tableTanh(0.7978845608028654f * value * (1.0f + 0.044715f * value * value)));
      out[outOffset + index] = gelu * up[upOffset + index];
    }
  }

  /**
   * Package-private so the table's last slot can be tested at the exact input that reaches it. The
   * boundary is one float ulp wide, so a sweep through plausible activations does not reliably land
   * on it -- a sweep written first missed it entirely and passed against the crashing version.
   */
  static float tableTanh(float value) {
    if (value <= -TANH_TABLE_LIMIT) {
      return -1.0f;
    }
    if (value >= TANH_TABLE_LIMIT) {
      return 1.0f;
    }
    float tablePosition = (value + TANH_TABLE_LIMIT) * TANH_TABLE_SCALE;
    int index = (int) tablePosition;
    // The bounds check above rejects value >= TANH_TABLE_LIMIT, but not every value below the limit
    // scales to a position below the last slot: for value just under the limit, (value + limit) *
    // scale
    // rounds to exactly TANH_TABLE_SIZE in float arithmetic, so index lands on the final entry and
    // the
    // interpolation below reads one past it. The table holds SIZE + 1 entries, so that read is
    // ArrayIndexOutOfBoundsException: Index 65537 out of bounds for length 65537 -- which is how
    // this
    // was found, crashing a gemma3n run 456 seconds in, from inside gelu.
    //
    // At the final slot there is nothing left to interpolate towards, and the tabulated value is
    // tanh(limit) to within the table's own resolution, so return it.
    if (index >= TANH_TABLE_SIZE) {
      return TANH_TABLE[TANH_TABLE_SIZE];
    }
    float lower = TANH_TABLE[index];
    return lower + (tablePosition - index) * (TANH_TABLE[index + 1] - lower);
  }

  private static float[] createTanhTable() {
    float[] table = new float[TANH_TABLE_SIZE + 1];
    for (int index = 0; index <= TANH_TABLE_SIZE; index++) {
      float value = -TANH_TABLE_LIMIT + index / TANH_TABLE_SCALE;
      table[index] = (float) Math.tanh(value);
    }
    return table;
  }

  private static void rotatePair(float[] vector, int offset, float cos, float sin) {
    float x0 = vector[offset];
    float x1 = vector[offset + 1];
    vector[offset] = x0 * cos - x1 * sin;
    vector[offset + 1] = x0 * sin + x1 * cos;
  }

  private static void rotateSplitPair(float[] vector, int first, int second, float cos, float sin) {
    float x0 = vector[first];
    float x1 = vector[second];
    vector[first] = x0 * cos - x1 * sin;
    vector[second] = x0 * sin + x1 * cos;
  }
}
