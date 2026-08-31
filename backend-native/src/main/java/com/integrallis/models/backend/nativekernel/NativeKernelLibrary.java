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
  private final int threadCount;
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
      int threadCount) {
    this.libraryArena = libraryArena;
    this.capabilities = capabilities;
    this.context = context;
    this.contextDestroyHandle = contextDestroyHandle;
    this.quantizedBatchedHandle = quantizedBatchedHandle;
    this.quantizedGroupedBatchedHandle = quantizedGroupedBatchedHandle;
    this.quantizedIndependentBatchedHandle = quantizedIndependentBatchedHandle;
    this.gatedDeltaNetHandle = gatedDeltaNetHandle;
    this.threadCount = threadCount;
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
      MemorySegment context = invokeAddress(contextCreate, threadCount, "create worker context");
      if (context.address() == 0) {
        throw new IllegalStateException("native kernel worker context creation failed");
      }
      return new NativeKernelLibrary(
          arena,
          capabilityMask,
          context,
          contextDestroy,
          quantizedBatched,
          quantizedGroupedBatched,
          quantizedIndependentBatched,
          gatedDeltaNet,
          threadCount);
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
