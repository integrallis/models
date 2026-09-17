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
package com.integrallis.models.backend.nativekernel;

import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliff;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliffs;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/** Versioned FFM access to Models-owned native inference kernels. */
@SuppressWarnings("restricted")
public final class NativeKernelLibrary implements AutoCloseable {
  public static final int ABI_VERSION = 5;
  public static final String THREAD_COUNT_PROPERTY = "models.native.kernels.threads";

  /**
   * Workers that take rows on single-token (decode) projections; defaults to the pool size. Batched
   * prefill keeps the whole pool. Honoured only when the library exports the active-thread setter
   * ({@link NativeKernelCapability#ACTIVE_THREADS}).
   */
  public static final String DECODE_THREAD_COUNT_PROPERTY = "models.native.kernels.decodeThreads";

  /**
   * Milliseconds the native workers keep polling for the next dispatch before they park. The
   * default (5 ms) is longer than any gap between two dispatches of one token, so the pool stays
   * hot for a whole generation and parks shortly after it ends; ggml's CPU backend runs the same
   * regime unbounded. Measured 2026-09-16: on a shared 16-vCPU host 4,000 spin rounds then park
   * gave 15.6-16.6 tok/s with ~800k context switches per run and a token-sized budget 24.0-26.0
   * tok/s with ~180k; on dedicated cores (c7a.4xlarge) 1 / 5 / 25 ms gave 25.8 / 25.9 / 26.1 tok/s
   * against 25.5. Five milliseconds keeps the whole gain and a fifth of the idle tail.
   */
  public static final String POLL_MILLIS_PROPERTY = "models.native.kernels.pollMillis";

  static final long DEFAULT_POLL_MILLIS = 5;

  private static final int STATUS_OK = 0;
  private static final int FORMAT_Q4_0 = 0;
  private static final int FORMAT_Q8_0 = 1;
  private static final int FORMAT_Q4_K = 2;
  private static final int FORMAT_Q6_K = 3;
  private static final int FORMAT_Q5_K = 4;
  private static final int FORMAT_Q5_0 = 5;
  private static final Linker LINKER = Linker.nativeLinker();
  private static final FunctionDescriptor ABI_VERSION_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.JAVA_INT);
  private static final FunctionDescriptor CAPABILITIES_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.JAVA_LONG);
  private static final FunctionDescriptor CONTEXT_CREATE_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT);
  private static final FunctionDescriptor SET_ACTIVE_THREADS_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);
  private static final FunctionDescriptor SET_POLL_NANOS_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
  private static final FunctionDescriptor GROUPED_ATTENTION_DESCRIPTOR =
      FunctionDescriptor.of(
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_FLOAT);
  private static final FunctionDescriptor CONTEXT_DESTROY_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS);
  private static final FunctionDescriptor QUANTIZED_BATCHED_WITH_CONTEXT_DESCRIPTOR =
      FunctionDescriptor.of(
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT);
  private static final FunctionDescriptor QUANTIZED_GROUPED_BATCHED_WITH_CONTEXT_DESCRIPTOR =
      FunctionDescriptor.of(
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT);
  private static final FunctionDescriptor QUANTIZED_INDEPENDENT_BATCHED_WITH_CONTEXT_DESCRIPTOR =
      FunctionDescriptor.of(
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_INT);
  private static final FunctionDescriptor GATED_DELTA_NET_WITH_CONTEXT_DESCRIPTOR =
      FunctionDescriptor.of(
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT);

  private final Arena libraryArena;
  private final long capabilities;
  private final MemorySegment context;
  private final MethodHandle contextDestroyHandle;
  private final MethodHandle quantizedBatchedHandle;
  private final MethodHandle quantizedGroupedBatchedHandle;
  private final MethodHandle quantizedIndependentBatchedHandle;
  private final MethodHandle gatedDeltaNetHandle;
  private final MethodHandle setActiveThreadsHandle;
  private final MethodHandle groupedAttentionHandle;
  private final int threadCount;
  private final int decodeThreadCount;
  private final long pollMillis;
  private int activeThreadCount;
  private boolean closed;

  private NativeKernelLibrary(
      Arena libraryArena,
      long capabilities,
      MemorySegment context,
      MethodHandle contextDestroyHandle,
      MethodHandle quantizedBatchedHandle,
      MethodHandle quantizedGroupedBatchedHandle,
      MethodHandle quantizedIndependentBatchedHandle,
      MethodHandle gatedDeltaNetHandle,
      MethodHandle setActiveThreadsHandle,
      MethodHandle groupedAttentionHandle,
      int threadCount,
      int decodeThreadCount,
      long pollMillis) {
    this.libraryArena = libraryArena;
    this.capabilities = capabilities;
    this.context = context;
    this.contextDestroyHandle = contextDestroyHandle;
    this.quantizedBatchedHandle = quantizedBatchedHandle;
    this.quantizedGroupedBatchedHandle = quantizedGroupedBatchedHandle;
    this.quantizedIndependentBatchedHandle = quantizedIndependentBatchedHandle;
    this.gatedDeltaNetHandle = gatedDeltaNetHandle;
    this.setActiveThreadsHandle = setActiveThreadsHandle;
    this.groupedAttentionHandle = groupedAttentionHandle;
    this.pollMillis = pollMillis;
    this.threadCount = threadCount;
    this.decodeThreadCount = decodeThreadCount;
    this.activeThreadCount = threadCount;
  }

  /** Opens a platform library and rejects incompatible ABI versions immediately. */
  public static NativeKernelLibrary open(Path libraryPath) {
    return open(libraryPath, configuredThreadCount());
  }

  static NativeKernelLibrary open(Path libraryPath, int threadCount) {
    Objects.requireNonNull(libraryPath, "libraryPath");
    validateThreadCount(threadCount, Integer.toString(threadCount));
    Path normalized = libraryPath.toAbsolutePath().normalize();
    if (!Files.isRegularFile(normalized)) {
      throw new IllegalArgumentException("native kernel library does not exist: " + normalized);
    }

    Arena arena = Arena.ofShared();
    try {
      SymbolLookup lookup = SymbolLookup.libraryLookup(normalized, arena);
      MethodHandle abiVersion =
          downcall(lookup, "jmodels_kernels_abi_version", ABI_VERSION_DESCRIPTOR);
      int actualAbiVersion = invokeInt(abiVersion, "read ABI version");
      if (actualAbiVersion != ABI_VERSION) {
        throw new IllegalArgumentException(
            "unsupported native kernel ABI " + actualAbiVersion + "; expected " + ABI_VERSION);
      }
      MethodHandle capabilities =
          downcall(lookup, "jmodels_kernels_capabilities", CAPABILITIES_DESCRIPTOR);
      long capabilityMask = invokeLong(capabilities, "read capabilities");
      if ((capabilityMask & NativeKernelCapability.PERSISTENT_WORKER_CONTEXT.mask()) == 0) {
        throw new IllegalArgumentException(
            "native kernel library does not provide a persistent worker context");
      }
      MethodHandle contextCreate =
          downcall(lookup, "jmodels_kernels_context_create", CONTEXT_CREATE_DESCRIPTOR);
      MethodHandle contextDestroy =
          downcall(lookup, "jmodels_kernels_context_destroy", CONTEXT_DESTROY_DESCRIPTOR);
      MethodHandle quantizedBatched =
          downcall(
              lookup,
              "jmodels_quantized_f32_batched_matmul_with_context",
              QUANTIZED_BATCHED_WITH_CONTEXT_DESCRIPTOR);
      MethodHandle quantizedGroupedBatched =
          downcall(
              lookup,
              "jmodels_quantized_f32_grouped_batched_matmul_with_context",
              QUANTIZED_GROUPED_BATCHED_WITH_CONTEXT_DESCRIPTOR);
      MethodHandle quantizedIndependentBatched =
          downcall(
              lookup,
              "jmodels_quantized_f32_independent_batched_matmul_with_context",
              QUANTIZED_INDEPENDENT_BATCHED_WITH_CONTEXT_DESCRIPTOR);
      MethodHandle gatedDeltaNet =
          downcallCritical(
              lookup,
              "jmodels_gated_delta_net_f32_with_context",
              GATED_DELTA_NET_WITH_CONTEXT_DESCRIPTOR);
      MethodHandle setActiveThreads =
          (capabilityMask & NativeKernelCapability.ACTIVE_THREADS.mask()) != 0
              ? downcall(
                  lookup,
                  "jmodels_kernels_context_set_active_threads",
                  SET_ACTIVE_THREADS_DESCRIPTOR)
              : null;
      MethodHandle groupedAttention =
          (capabilityMask & NativeKernelCapability.GROUPED_ATTENTION_F32.mask()) != 0
              ? downcallCritical(
                  lookup,
                  "jmodels_grouped_attention_f32_with_context",
                  GROUPED_ATTENTION_DESCRIPTOR)
              : null;
      MethodHandle setPollNanos =
          (capabilityMask & NativeKernelCapability.POLL_BUDGET.mask()) != 0
              ? downcall(
                  lookup, "jmodels_kernels_context_set_poll_nanos", SET_POLL_NANOS_DESCRIPTOR)
              : null;
      int decodeThreadCount = configuredDecodeThreadCount(threadCount);
      long pollMillis = configuredPollMillis();
      MemorySegment context = invokeAddress(contextCreate, threadCount, "create worker context");
      if (context.address() == 0) {
        throw new IllegalStateException("native kernel worker context creation failed");
      }
      applyPollBudget(setPollNanos, context, pollMillis);
      return new NativeKernelLibrary(
          arena,
          capabilityMask,
          context,
          contextDestroy,
          quantizedBatched,
          quantizedGroupedBatched,
          quantizedIndependentBatched,
          gatedDeltaNet,
          setActiveThreads,
          groupedAttention,
          threadCount,
          decodeThreadCount,
          pollMillis);
    } catch (RuntimeException | LinkageError failure) {
      arena.close();
      throw failure;
    }
  }

  /** Returns the ABI version accepted by this Java binding. */
  public int abiVersion() {
    return ABI_VERSION;
  }

  int threadCount() {
    return threadCount;
  }

  /** Returns whether the loaded library advertises the given operation. */
  public boolean supports(NativeKernelCapability capability) {
    Objects.requireNonNull(capability, "capability");
    return (capabilities & capability.mask()) != 0;
  }

  /** Applies the Qwen 3.5 float32 Gated DeltaNet recurrence in caller-owned arrays. */
  public void gatedDeltaNetF32(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] state,
      float[] output,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(value, "value");
    Objects.requireNonNull(logDecay, "logDecay");
    Objects.requireNonNull(beta, "beta");
    Objects.requireNonNull(state, "state");
    Objects.requireNonNull(output, "output");
    requirePositive(tokenCount, "tokenCount");
    requirePositive(keyHeadCount, "keyHeadCount");
    requirePositive(valueHeadCount, "valueHeadCount");
    requirePositive(keyDimension, "keyDimension");
    requirePositive(valueDimension, "valueDimension");
    if (valueHeadCount % keyHeadCount != 0) {
      throw new IllegalArgumentException("valueHeadCount must be divisible by keyHeadCount");
    }
    if (keyDimension > 256 || valueDimension > 256) {
      throw new IllegalArgumentException("Gated DeltaNet dimensions must not exceed 256");
    }
    int requiredQuery =
        Math.multiplyExact(Math.multiplyExact(tokenCount, keyHeadCount), keyDimension);
    int requiredValue =
        Math.multiplyExact(Math.multiplyExact(tokenCount, valueHeadCount), valueDimension);
    int requiredGates = Math.multiplyExact(tokenCount, valueHeadCount);
    int requiredState =
        Math.multiplyExact(Math.multiplyExact(valueHeadCount, keyDimension), valueDimension);
    requireCapacity(query, requiredQuery, "query");
    requireCapacity(key, requiredQuery, "key");
    requireCapacity(value, requiredValue, "value");
    requireCapacity(logDecay, requiredGates, "logDecay");
    requireCapacity(beta, requiredGates, "beta");
    requireCapacity(state, requiredState, "state");
    requireCapacity(output, requiredValue, "output");
    if (!supports(NativeKernelCapability.GATED_DELTA_NET_F32)) {
      throw new UnsupportedOperationException("loaded native library has no Gated DeltaNet kernel");
    }

    try {
      int status =
          (int)
              gatedDeltaNetHandle.invokeExact(
                  context,
                  MemorySegment.ofArray(query),
                  (long) query.length,
                  MemorySegment.ofArray(key),
                  (long) key.length,
                  MemorySegment.ofArray(value),
                  (long) value.length,
                  MemorySegment.ofArray(logDecay),
                  (long) logDecay.length,
                  MemorySegment.ofArray(beta),
                  (long) beta.length,
                  MemorySegment.ofArray(state),
                  (long) state.length,
                  MemorySegment.ofArray(output),
                  (long) output.length,
                  tokenCount,
                  keyHeadCount,
                  valueHeadCount,
                  keyDimension,
                  valueDimension);
      if (status != STATUS_OK) {
        throw new IllegalStateException(
            "native Gated DeltaNet kernel failed: " + statusName(status));
      }
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw bridgeFailure("Gated DeltaNet recurrence", failure);
    }
  }

  /** Computes a batch-major {@code input[batch, cols] * weights[rows, cols]} Q4_0 projection. */
  public void q4_0F32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    f32BatchedMatmul(
        "Q4_0",
        FORMAT_Q4_0,
        32,
        18L,
        NativeKernelCapability.Q4_0_F32_BATCHED_MATMUL,
        weights,
        input,
        batchSize,
        rows,
        cols,
        output);
  }

  /** Computes a batch-major {@code input[batch, cols] * weights[rows, cols]} Q8_0 projection. */
  public void q8_0F32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    f32BatchedMatmul(
        "Q8_0",
        FORMAT_Q8_0,
        32,
        34L,
        NativeKernelCapability.Q8_0_F32_BATCHED_MATMUL,
        weights,
        input,
        batchSize,
        rows,
        cols,
        output);
  }

  /** Computes a batch-major {@code input[batch, cols] * weights[rows, cols]} Q5_0 projection. */
  public void q5_0F32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    f32BatchedMatmul(
        "Q5_0",
        FORMAT_Q5_0,
        32,
        22L,
        NativeKernelCapability.Q5_0_F32_BATCHED_MATMUL,
        weights,
        input,
        batchSize,
        rows,
        cols,
        output);
  }

  /** Computes a batch-major {@code input[batch, cols] * weights[rows, cols]} Q4_K projection. */
  public void q4_KF32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    f32BatchedMatmul(
        "Q4_K",
        FORMAT_Q4_K,
        256,
        144L,
        NativeKernelCapability.Q4_K_F32_BATCHED_MATMUL,
        weights,
        input,
        batchSize,
        rows,
        cols,
        output);
  }

  /** Computes a batch-major {@code input[batch, cols] * weights[rows, cols]} Q5_K projection. */
  public void q5_KF32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    f32BatchedMatmul(
        "Q5_K",
        FORMAT_Q5_K,
        256,
        176L,
        NativeKernelCapability.Q5_K_F32_BATCHED_MATMUL,
        weights,
        input,
        batchSize,
        rows,
        cols,
        output);
  }

  /** Computes a batch-major {@code input[batch, cols] * weights[rows, cols]} Q6_K projection. */
  public void q6_KF32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    f32BatchedMatmul(
        "Q6_K",
        FORMAT_Q6_K,
        256,
        210L,
        NativeKernelCapability.Q6_K_F32_BATCHED_MATMUL,
        weights,
        input,
        batchSize,
        rows,
        cols,
        output);
  }

  void q4_0F32BatchedMatmul(
      MemorySegment weights,
      long weightBytes,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int rows,
      int cols) {
    invokeBatched(
        "Q4_0",
        FORMAT_Q4_0,
        NativeKernelCapability.Q4_0_F32_BATCHED_MATMUL,
        weights,
        weightBytes,
        nativeInput,
        inputElements,
        nativeOutput,
        outputElements,
        batchSize,
        rows,
        cols);
  }

  void q8_0F32BatchedMatmul(
      MemorySegment weights,
      long weightBytes,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int rows,
      int cols) {
    invokeBatched(
        "Q8_0",
        FORMAT_Q8_0,
        NativeKernelCapability.Q8_0_F32_BATCHED_MATMUL,
        weights,
        weightBytes,
        nativeInput,
        inputElements,
        nativeOutput,
        outputElements,
        batchSize,
        rows,
        cols);
  }

  void q5_0F32BatchedMatmul(
      MemorySegment weights,
      long weightBytes,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int rows,
      int cols) {
    invokeBatched(
        "Q5_0",
        FORMAT_Q5_0,
        NativeKernelCapability.Q5_0_F32_BATCHED_MATMUL,
        weights,
        weightBytes,
        nativeInput,
        inputElements,
        nativeOutput,
        outputElements,
        batchSize,
        rows,
        cols);
  }

  void q4_KF32BatchedMatmul(
      MemorySegment weights,
      long weightBytes,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int rows,
      int cols) {
    invokeBatched(
        "Q4_K",
        FORMAT_Q4_K,
        NativeKernelCapability.Q4_K_F32_BATCHED_MATMUL,
        weights,
        weightBytes,
        nativeInput,
        inputElements,
        nativeOutput,
        outputElements,
        batchSize,
        rows,
        cols);
  }

  void q5_KF32BatchedMatmul(
      MemorySegment weights,
      long weightBytes,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int rows,
      int cols) {
    invokeBatched(
        "Q5_K",
        FORMAT_Q5_K,
        NativeKernelCapability.Q5_K_F32_BATCHED_MATMUL,
        weights,
        weightBytes,
        nativeInput,
        inputElements,
        nativeOutput,
        outputElements,
        batchSize,
        rows,
        cols);
  }

  void q6_KF32BatchedMatmul(
      MemorySegment weights,
      long weightBytes,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int rows,
      int cols) {
    invokeBatched(
        "Q6_K",
        FORMAT_Q6_K,
        NativeKernelCapability.Q6_K_F32_BATCHED_MATMUL,
        weights,
        weightBytes,
        nativeInput,
        inputElements,
        nativeOutput,
        outputElements,
        batchSize,
        rows,
        cols);
  }

  void quantizedF32GroupedBatchedMatmul(
      String type,
      NativeKernelCapability capability,
      MemorySegment formats,
      MemorySegment weightPointers,
      MemorySegment weightBytes,
      MemorySegment rows,
      int matrixCount,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int cols) {
    invokeGrouped(
        type,
        capability,
        formats,
        weightPointers,
        weightBytes,
        rows,
        matrixCount,
        nativeInput,
        inputElements,
        nativeOutput,
        outputElements,
        batchSize,
        cols);
  }

  void quantizedF32IndependentBatchedMatmul(
      MemorySegment formats,
      MemorySegment weightPointers,
      MemorySegment weightBytes,
      MemorySegment rows,
      MemorySegment batchSizes,
      int matrixCount,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int cols) {
    if (!supports(NativeKernelCapability.INDEPENDENT_BATCHED_MATMUL)) {
      throw new UnsupportedOperationException(
          "loaded native library has no independent batched kernel");
    }
    // Independent batches always carry more than one token in total; use the whole pool.
    selectWorkers(Integer.MAX_VALUE);
    try {
      int status =
          (int)
              quantizedIndependentBatchedHandle.invokeExact(
                  context,
                  formats,
                  weightPointers,
                  weightBytes,
                  rows,
                  batchSizes,
                  matrixCount,
                  nativeInput,
                  inputElements,
                  nativeOutput,
                  outputElements,
                  cols);
      if (status != STATUS_OK) {
        throw new IllegalStateException(
            "native independent batched kernel failed: " + statusName(status));
      }
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw bridgeFailure("independent batched matmul", failure);
    }
  }

  private void f32BatchedMatmul(
      String type,
      int format,
      int blockElements,
      long blockBytes,
      NativeKernelCapability capability,
      MemorySegment weights,
      float[] input,
      int batchSize,
      int rows,
      int cols,
      float[] output) {
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (!supports(capability)) {
      throw new UnsupportedOperationException(
          "loaded native library has no " + type + " batched kernel");
    }
    if (batchSize < 1 || rows < 1 || cols < 1) {
      throw new IllegalArgumentException("batchSize, rows, and cols must be positive");
    }
    if (cols % blockElements != 0) {
      throw new IllegalArgumentException(
          type + " column count must be a multiple of " + blockElements + ": " + cols);
    }

    int inputElements = Math.multiplyExact(batchSize, cols);
    int outputElements = Math.multiplyExact(batchSize, rows);
    long weightBytes =
        Math.multiplyExact(Math.multiplyExact((long) rows, cols / blockElements), blockBytes);
    if (input.length < inputElements) {
      throw new IllegalArgumentException(
          "input requires " + inputElements + " elements but has " + input.length);
    }
    if (output.length < outputElements) {
      throw new IllegalArgumentException(
          "output requires " + outputElements + " elements but has " + output.length);
    }
    if (weights.byteSize() < weightBytes) {
      throw new IllegalArgumentException(
          "weights require " + weightBytes + " bytes but have " + weights.byteSize());
    }
    if (!weights.isNative()) {
      throw new IllegalArgumentException("weights must be backed by native or mapped memory");
    }

    try (Arena callArena = Arena.ofConfined()) {
      MemorySegment nativeInput = callArena.allocate(ValueLayout.JAVA_FLOAT, inputElements);
      MemorySegment nativeOutput = callArena.allocate(ValueLayout.JAVA_FLOAT, outputElements);
      MemorySegment.copy(input, 0, nativeInput, ValueLayout.JAVA_FLOAT, 0, inputElements);
      invokeBatched(
          type,
          format,
          capability,
          weights,
          weightBytes,
          nativeInput,
          inputElements,
          nativeOutput,
          outputElements,
          batchSize,
          rows,
          cols);
      MemorySegment.copy(nativeOutput, ValueLayout.JAVA_FLOAT, 0, output, 0, outputElements);
    }
  }

  private void invokeBatched(
      String type,
      int format,
      NativeKernelCapability capability,
      MemorySegment weights,
      long weightBytes,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int rows,
      int cols) {
    if (!supports(capability)) {
      throw new UnsupportedOperationException(
          "loaded native library has no " + type + " batched kernel");
    }
    selectWorkers(batchSize);
    try {
      int status =
          (int)
              quantizedBatchedHandle.invokeExact(
                  context,
                  format,
                  weights,
                  weightBytes,
                  nativeInput,
                  inputElements,
                  nativeOutput,
                  outputElements,
                  batchSize,
                  rows,
                  cols);
      if (status != STATUS_OK) {
        throw new IllegalStateException(
            "native " + type + " batched kernel failed: " + statusName(status));
      }
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw bridgeFailure(type + " batched matmul", failure);
    }
  }

  private void invokeGrouped(
      String type,
      NativeKernelCapability capability,
      MemorySegment formats,
      MemorySegment weightPointers,
      MemorySegment weightBytes,
      MemorySegment rows,
      int matrixCount,
      MemorySegment nativeInput,
      long inputElements,
      MemorySegment nativeOutput,
      long outputElements,
      int batchSize,
      int cols) {
    if (!supports(capability)) {
      throw new UnsupportedOperationException(
          "loaded native library has no grouped " + type + " batched kernel");
    }
    selectWorkers(batchSize);
    try {
      int status =
          (int)
              quantizedGroupedBatchedHandle.invokeExact(
                  context,
                  formats,
                  weightPointers,
                  weightBytes,
                  rows,
                  matrixCount,
                  nativeInput,
                  inputElements,
                  nativeOutput,
                  outputElements,
                  batchSize,
                  cols);
      if (status != STATUS_OK) {
        throw new IllegalStateException(
            "native grouped " + type + " batched kernel failed: " + statusName(status));
      }
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw bridgeFailure("grouped " + type + " batched matmul", failure);
    }
  }

  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    closed = true;
    RuntimeException failure = null;
    try {
      int status = (int) contextDestroyHandle.invokeExact(context);
      if (status != STATUS_OK) {
        failure =
            new IllegalStateException(
                "native worker context shutdown failed: " + statusName(status));
      }
    } catch (RuntimeException closeFailure) {
      failure = closeFailure;
    } catch (Throwable closeFailure) {
      failure = bridgeFailure("destroy worker context", closeFailure);
    } finally {
      libraryArena.close();
    }
    if (failure != null) {
      throw failure;
    }
  }

  private static MethodHandle downcall(
      SymbolLookup lookup, String symbolName, FunctionDescriptor descriptor) {
    MemorySegment symbol =
        lookup
            .find(symbolName)
            .orElseThrow(
                () -> new IllegalArgumentException("missing native kernel symbol: " + symbolName));
    return LINKER.downcallHandle(symbol, descriptor);
  }

  private static MethodHandle downcallCritical(
      SymbolLookup lookup, String symbolName, FunctionDescriptor descriptor) {
    MemorySegment symbol =
        lookup
            .find(symbolName)
            .orElseThrow(
                () -> new IllegalArgumentException("missing native kernel symbol: " + symbolName));
    return LINKER.downcallHandle(symbol, descriptor, Linker.Option.critical(true));
  }

  private static void requirePositive(int value, String name) {
    if (value < 1) {
      throw new IllegalArgumentException(name + " must be positive: " + value);
    }
  }

  private static void requireCapacity(float[] values, int required, String name) {
    if (values.length < required) {
      throw new IllegalArgumentException(
          name + " requires " + required + " elements but has " + values.length);
    }
  }

  private static int invokeInt(MethodHandle handle, String operation) {
    try {
      return (int) handle.invokeExact();
    } catch (Throwable failure) {
      throw bridgeFailure(operation, failure);
    }
  }

  private static long invokeLong(MethodHandle handle, String operation) {
    try {
      return (long) handle.invokeExact();
    } catch (Throwable failure) {
      throw bridgeFailure(operation, failure);
    }
  }

  private static MemorySegment invokeAddress(MethodHandle handle, int argument, String operation) {
    try {
      return (MemorySegment) handle.invokeExact(argument);
    } catch (Throwable failure) {
      throw bridgeFailure(operation, failure);
    }
  }

  /** Milliseconds the native workers poll before parking, as configured for this context. */
  long pollMillis() {
    return pollMillis;
  }

  /** Whether the loaded library lets the poll budget be set per context. */
  boolean supportsPollBudget() {
    return (capabilities & NativeKernelCapability.POLL_BUDGET.mask()) != 0;
  }

  /**
   * Rows-taking workers on single-token projections, as configured; equals the pool size by
   * default.
   */
  int decodeThreadCount() {
    return decodeThreadCount;
  }

  /**
   * Points the native pool at the worker count this batch size wants: the configured decode count
   * for a single token, the whole pool otherwise. No-op when unchanged or unsupported.
   */
  private void selectWorkers(int batchSize) {
    if (setActiveThreadsHandle == null) {
      return;
    }
    int desired = batchSize == 1 ? decodeThreadCount : threadCount;
    if (desired == activeThreadCount) {
      return;
    }
    try {
      int inEffect = (int) setActiveThreadsHandle.invokeExact(context, desired);
      if (inEffect < 0) {
        throw new IllegalStateException(
            "native kernel refused active thread count " + desired + ": status " + (-inEffect));
      }
      activeThreadCount = inEffect;
    } catch (Throwable failure) {
      throw bridgeFailure("set active worker count", failure);
    }
  }

  /**
   * Grouped-query attention for one query row over up to two cached key/value spans, computed by
   * the native kernel on its worker pool and written into {@code output} (overwritten, not
   * accumulated). Heap arrays cross the boundary without copying under the critical linker option.
   * {@code scores} is scratch of at least {@code numHeads * (positionsA + positionsB)}.
   */
  public void groupedAttentionF32(
      float[] query,
      int queryOffset,
      float[] keysA,
      int keysAOffset,
      float[] valuesA,
      int valuesAOffset,
      int positionsA,
      float[] keysB,
      int keysBOffset,
      float[] valuesB,
      int valuesBOffset,
      int positionsB,
      float[] output,
      int outputOffset,
      float[] scores,
      int keyDim,
      int valueDim,
      int keyLength,
      int valueLength,
      int numHeads,
      int numKvHeads,
      float scale) {
    if (groupedAttentionHandle == null) {
      throw new UnsupportedOperationException(
          "loaded native library has no grouped attention kernel");
    }
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(keysA, "keysA");
    Objects.requireNonNull(valuesA, "valuesA");
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(scores, "scores");
    float[] safeKeysB = keysB == null ? EMPTY : keysB;
    float[] safeValuesB = valuesB == null ? EMPTY : valuesB;
    selectWorkers(1);
    try {
      int status =
          (int)
              groupedAttentionHandle.invokeExact(
                  context,
                  MemorySegment.ofArray(query),
                  (long) queryOffset,
                  (long) query.length,
                  MemorySegment.ofArray(keysA),
                  (long) keysAOffset,
                  (long) keysA.length,
                  MemorySegment.ofArray(valuesA),
                  (long) valuesAOffset,
                  (long) valuesA.length,
                  positionsA,
                  MemorySegment.ofArray(safeKeysB),
                  (long) keysBOffset,
                  (long) safeKeysB.length,
                  MemorySegment.ofArray(safeValuesB),
                  (long) valuesBOffset,
                  (long) safeValuesB.length,
                  positionsB,
                  MemorySegment.ofArray(output),
                  (long) outputOffset,
                  (long) output.length,
                  MemorySegment.ofArray(scores),
                  (long) scores.length,
                  keyDim,
                  valueDim,
                  keyLength,
                  valueLength,
                  numHeads,
                  numKvHeads,
                  scale);
      if (status != 0) {
        throw new IllegalStateException("native grouped attention failed: " + statusName(status));
      }
    } catch (Throwable failure) {
      throw bridgeFailure("grouped attention", failure);
    }
  }

  private static final float[] EMPTY = new float[0];

  /**
   * Applies the configured worker poll budget, or reports {@link
   * PerformanceCliff#NATIVE_POLL_BUDGET_UNSUPPORTED} when the loaded library has no poll-budget
   * entry point ({@code setPollNanos == null}) and its workers keep their built-in behaviour.
   */
  static void applyPollBudget(MethodHandle setPollNanos, MemorySegment context, long pollMillis) {
    if (setPollNanos == null) {
      PerformanceCliffs.report(
          PerformanceCliff.NATIVE_POLL_BUDGET_UNSUPPORTED,
          "abi=" + ABI_VERSION + ", poll-millis=" + pollMillis);
      return;
    }
    try {
      long applied = (long) setPollNanos.invokeExact(context, pollMillis * 1_000_000L);
      if (applied <= 0) {
        throw new IllegalStateException("native kernel poll budget was not applied");
      }
    } catch (Throwable failure) {
      throw new IllegalStateException("native kernel poll budget could not be set", failure);
    }
  }

  static long configuredPollMillis() {
    String configured = System.getProperty(POLL_MILLIS_PROPERTY);
    if (configured == null || configured.isBlank()) {
      return DEFAULT_POLL_MILLIS;
    }
    try {
      long millis = Long.parseLong(configured.trim());
      if (millis < 1 || millis > 60_000) {
        throw new IllegalArgumentException(
            POLL_MILLIS_PROPERTY + " must be between 1 and 60000 milliseconds: " + configured);
      }
      return millis;
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException(
          POLL_MILLIS_PROPERTY + " must be an integer: " + configured, failure);
    }
  }

  private static int configuredDecodeThreadCount(int threadCount) {
    String configured = System.getProperty(DECODE_THREAD_COUNT_PROPERTY);
    if (configured == null || configured.isBlank()) {
      return threadCount;
    }
    try {
      int decodeThreads = Integer.parseInt(configured);
      if (decodeThreads < 1 || decodeThreads > threadCount) {
        throw new IllegalArgumentException(
            DECODE_THREAD_COUNT_PROPERTY
                + " must be between 1 and the pool size "
                + threadCount
                + ": "
                + configured);
      }
      return decodeThreads;
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException(
          DECODE_THREAD_COUNT_PROPERTY + " must be an integer: " + configured, failure);
    }
  }

  private static int configuredThreadCount() {
    String configured = System.getProperty(THREAD_COUNT_PROPERTY);
    if (configured == null || configured.isBlank()) {
      return Runtime.getRuntime().availableProcessors();
    }
    try {
      int threadCount = Integer.parseInt(configured);
      validateThreadCount(threadCount, configured);
      return threadCount;
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException(
          THREAD_COUNT_PROPERTY + " must be an integer: " + configured, failure);
    }
  }

  private static void validateThreadCount(int threadCount, String configured) {
    if (threadCount < 1 || threadCount > 256) {
      throw new IllegalArgumentException(
          THREAD_COUNT_PROPERTY + " must be between 1 and 256: " + configured);
    }
  }

  private static String statusName(int status) {
    return switch (status) {
      case 1 -> "null pointer";
      case 2 -> "invalid shape";
      case 3 -> "buffer too small";
      case 4 -> "native panic";
      default -> "unknown status " + status;
    };
  }

  private static IllegalStateException bridgeFailure(String operation, Throwable failure) {
    if (failure instanceof Error error) {
      throw error;
    }
    return new IllegalStateException("native kernel bridge failed during " + operation, failure);
  }
}
