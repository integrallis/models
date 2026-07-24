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
  public static final int ABI_VERSION = 1;
  public static final String THREAD_COUNT_PROPERTY = "models.native.kernels.threads";

  private static final int STATUS_OK = 0;
  private static final Linker LINKER = Linker.nativeLinker();
  private static final FunctionDescriptor ABI_VERSION_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.JAVA_INT);
  private static final FunctionDescriptor CAPABILITIES_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.JAVA_LONG);
  private static final FunctionDescriptor CONTEXT_CREATE_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT);
  private static final FunctionDescriptor CONTEXT_DESTROY_DESCRIPTOR =
      FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS);
  private static final FunctionDescriptor Q4_0_BATCHED_WITH_CONTEXT_DESCRIPTOR =
      FunctionDescriptor.of(
          ValueLayout.JAVA_INT,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_LONG,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT,
          ValueLayout.JAVA_INT);
  private static final FunctionDescriptor Q4_0_GROUPED_BATCHED_WITH_CONTEXT_DESCRIPTOR =
      FunctionDescriptor.of(
          ValueLayout.JAVA_INT,
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

  private final Arena libraryArena;
  private final long capabilities;
  private final MemorySegment context;
  private final MethodHandle contextDestroyHandle;
  private final MethodHandle q4BatchedHandle;
  private final MethodHandle q4GroupedBatchedHandle;
  private boolean closed;

  private NativeKernelLibrary(
      Arena libraryArena,
      long capabilities,
      MemorySegment context,
      MethodHandle contextDestroyHandle,
      MethodHandle q4BatchedHandle,
      MethodHandle q4GroupedBatchedHandle) {
    this.libraryArena = libraryArena;
    this.capabilities = capabilities;
    this.context = context;
    this.contextDestroyHandle = contextDestroyHandle;
    this.q4BatchedHandle = q4BatchedHandle;
    this.q4GroupedBatchedHandle = q4GroupedBatchedHandle;
  }

  /** Opens a platform library and rejects incompatible ABI versions immediately. */
  public static NativeKernelLibrary open(Path libraryPath) {
    Objects.requireNonNull(libraryPath, "libraryPath");
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
      MethodHandle q4Batched =
          downcall(
              lookup,
              "jmodels_q4_0_f32_batched_matmul_with_context",
              Q4_0_BATCHED_WITH_CONTEXT_DESCRIPTOR);
      MethodHandle q4GroupedBatched =
          downcall(
              lookup,
              "jmodels_q4_0_f32_grouped_batched_matmul_with_context",
              Q4_0_GROUPED_BATCHED_WITH_CONTEXT_DESCRIPTOR);
      MemorySegment context =
          invokeAddress(contextCreate, configuredThreadCount(), "create worker context");
      if (context.address() == 0) {
        throw new IllegalStateException("native kernel worker context creation failed");
      }
      return new NativeKernelLibrary(
          arena, capabilityMask, context, contextDestroy, q4Batched, q4GroupedBatched);
    } catch (RuntimeException | LinkageError failure) {
      arena.close();
      throw failure;
    }
  }

  /** Returns the ABI version accepted by this Java binding. */
  public int abiVersion() {
    return ABI_VERSION;
  }

  /** Returns whether the loaded library advertises the given operation. */
  public boolean supports(NativeKernelCapability capability) {
    Objects.requireNonNull(capability, "capability");
    return (capabilities & capability.mask()) != 0;
  }

  /** Computes a batch-major {@code input[batch, cols] * weights[rows, cols]} Q4_0 projection. */
  public void q4_0F32BatchedMatmul(
      MemorySegment weights, float[] input, int batchSize, int rows, int cols, float[] output) {
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (!supports(NativeKernelCapability.Q4_0_F32_BATCHED_MATMUL)) {
      throw new UnsupportedOperationException("loaded native library has no Q4_0 batched kernel");
    }
    if (batchSize < 1 || rows < 1 || cols < 1) {
      throw new IllegalArgumentException("batchSize, rows, and cols must be positive");
    }
    if (cols % 32 != 0) {
      throw new IllegalArgumentException("Q4_0 column count must be a multiple of 32: " + cols);
    }

    int inputElements = Math.multiplyExact(batchSize, cols);
    int outputElements = Math.multiplyExact(batchSize, rows);
    long weightBytes = Math.multiplyExact(Math.multiplyExact((long) rows, cols / 32L), 18L);
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
      q4_0F32BatchedMatmul(
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
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw bridgeFailure("Q4_0 batched matmul", failure);
    }
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
    try {
      int status =
          (int)
              q4BatchedHandle.invokeExact(
                  context,
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
        throw new IllegalStateException("native Q4_0 batched kernel failed: " + statusName(status));
      }
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw bridgeFailure("Q4_0 batched matmul", failure);
    }
  }

  void q4_0F32GroupedBatchedMatmul(
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
    if (!supports(NativeKernelCapability.Q4_0_F32_GROUPED_BATCHED_MATMUL)) {
      throw new UnsupportedOperationException(
          "loaded native library has no grouped Q4_0 batched kernel");
    }
    try {
      int status =
          (int)
              q4GroupedBatchedHandle.invokeExact(
                  context,
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
            "native grouped Q4_0 batched kernel failed: " + statusName(status));
      }
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw bridgeFailure("grouped Q4_0 batched matmul", failure);
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
      if (threadCount < 1 || threadCount > 256) {
        throw new IllegalArgumentException(
            THREAD_COUNT_PROPERTY + " must be between 1 and 256: " + configured);
      }
      return threadCount;
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException(
          THREAD_COUNT_PROPERTY + " must be an integer: " + configured, failure);
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
