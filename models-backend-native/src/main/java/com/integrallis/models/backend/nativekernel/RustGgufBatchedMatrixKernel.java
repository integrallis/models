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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/** Reusable off-heap workspace for Models-owned Rust quantized projection kernels. */
public final class RustGgufBatchedMatrixKernel implements GgufBatchedMatrixKernel {
  public static final String NATIVE_DECODE_PROPERTY = "models.native.quantizedDecode";
  public static final String Q5_0_GROUPED_PROPERTY = "models.native.q5_0.grouped";

  private static final Map<String, String> PLAN_RECOMMENDATIONS =
      Map.of(
          PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY,
          "true",
          PureJavaPlanConfiguration.MIXED_K_PROJECTIONS_PROPERTY,
          "true",
          PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY,
          "false",
          PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
          "false");

  private final NativeKernelLibrary library;
  private final boolean nativeDecode;
  private final boolean q5_0Grouped;
  private Arena scratchArena;
  private MemorySegment nativeInput = MemorySegment.NULL;
  private MemorySegment nativeOutput = MemorySegment.NULL;
  private MemorySegment nativeFormats = MemorySegment.NULL;
  private MemorySegment nativeWeightPointers = MemorySegment.NULL;
  private MemorySegment nativeWeightBytes = MemorySegment.NULL;
  private MemorySegment nativeRows = MemorySegment.NULL;
  private int inputCapacity;
  private int outputCapacity;
  private boolean closed;

  private RustGgufBatchedMatrixKernel(
      NativeKernelLibrary library, boolean nativeDecode, boolean q5_0Grouped) {
    this.library = Objects.requireNonNull(library, "library");
    this.nativeDecode = nativeDecode;
    this.q5_0Grouped = q5_0Grouped;
  }

  /** Opens a Rust kernel provider from an explicit platform library. */
  public static RustGgufBatchedMatrixKernel open(Path libraryPath) {
    return open(
        libraryPath,
        Boolean.getBoolean(NATIVE_DECODE_PROPERTY),
        Boolean.getBoolean(Q5_0_GROUPED_PROPERTY));
  }

  static RustGgufBatchedMatrixKernel open(Path libraryPath, boolean nativeDecode) {
    return open(libraryPath, nativeDecode, false);
  }

  static RustGgufBatchedMatrixKernel open(
      Path libraryPath, boolean nativeDecode, boolean q5_0Grouped) {
    return new RustGgufBatchedMatrixKernel(
        NativeKernelLibrary.open(libraryPath), nativeDecode, q5_0Grouped);
  }

  @Override
  public String implementation() {
    return "rust-ffm-quantized-v8";
  }

  @Override
  public Map<String, String> planRecommendations() {
    return PLAN_RECOMMENDATIONS;
  }

  boolean nativeDecodeEnabled() {
    return nativeDecode;
  }

  boolean q5_0GroupedEnabled() {
    return q5_0Grouped;
  }

  @Override
  public boolean supports(GgufTensorType type) {
    if (type == null) {
      return false;
    }
    return switch (type) {
      case Q4_0 -> library.supports(NativeKernelCapability.Q4_0_F32_BATCHED_MATMUL);
      case Q5_0 -> library.supports(NativeKernelCapability.Q5_0_F32_BATCHED_MATMUL);
      case Q8_0 -> library.supports(NativeKernelCapability.Q8_0_F32_BATCHED_MATMUL);
      case Q4_K -> library.supports(NativeKernelCapability.Q4_K_F32_BATCHED_MATMUL);
      case Q5_K -> library.supports(NativeKernelCapability.Q5_K_F32_BATCHED_MATMUL);
      case Q6_K -> library.supports(NativeKernelCapability.Q6_K_F32_BATCHED_MATMUL);
      default -> false;
    };
  }

  @Override
  public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
    return eligibleBatch(batchSize) && supports(type);
  }

  @Override
  public boolean supportsDual(GgufTensorType firstType, GgufTensorType secondType) {
    return q5_0GroupEligible(firstType, secondType)
        && supportsGrouped(firstType)
        && supportsGrouped(secondType)
        && compatibleGroup(firstType, secondType);
  }

  @Override
  public boolean isDualEligible(
      GgufTensorType firstType,
      int firstRows,
      GgufTensorType secondType,
      int secondRows,
      int batchSize,
      int cols) {
    return eligibleBatch(batchSize) && supportsDual(firstType, secondType);
  }

  @Override
  public synchronized void multiplyDual(
      float[] firstOutput,
      MemorySegment firstWeights,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOutput,
      MemorySegment secondWeights,
      GgufTensorType secondType,
      int secondRows,
      float[] input,
      int batchSize,
      int cols) {
    requireOpen();
    int firstElements =
        validateGroupedProjection(firstOutput, firstWeights, firstType, batchSize, firstRows, cols);
    int secondElements =
        validateGroupedProjection(
            secondOutput, secondWeights, secondType, batchSize, secondRows, cols);
    int totalOutputElements = Math.addExact(firstElements, secondElements);
    int inputElements = prepareGroupedWorkspace(input, batchSize, cols, totalOutputElements);
    configureGroupedProjection(0, firstWeights, firstType, firstRows, cols);
    configureGroupedProjection(1, secondWeights, secondType, secondRows, cols);
    invokeGrouped(
        firstType, secondType, null, inputElements, batchSize, cols, 2, totalOutputElements);
    copyOutput(firstOutput, 0, firstElements);
    copyOutput(secondOutput, firstElements, secondElements);
  }

  @Override
  public boolean supportsTriple(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    return q5_0GroupEligible(firstType, secondType, thirdType)
        && supportsGrouped(firstType)
        && supportsGrouped(secondType)
        && supportsGrouped(thirdType)
        && compatibleGroup(firstType, secondType, thirdType);
  }

  @Override
  public boolean isTripleEligible(
      GgufTensorType firstType,
      int firstRows,
      GgufTensorType secondType,
      int secondRows,
      GgufTensorType thirdType,
      int thirdRows,
      int batchSize,
      int cols) {
    return eligibleBatch(batchSize) && supportsTriple(firstType, secondType, thirdType);
  }

  @Override
  public synchronized void multiplyTriple(
      float[] firstOutput,
      MemorySegment firstWeights,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOutput,
      MemorySegment secondWeights,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOutput,
      MemorySegment thirdWeights,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int batchSize,
      int cols) {
    requireOpen();
    int firstElements =
        validateGroupedProjection(firstOutput, firstWeights, firstType, batchSize, firstRows, cols);
    int secondElements =
        validateGroupedProjection(
            secondOutput, secondWeights, secondType, batchSize, secondRows, cols);
    int thirdElements =
        validateGroupedProjection(thirdOutput, thirdWeights, thirdType, batchSize, thirdRows, cols);
    int firstTwoElements = Math.addExact(firstElements, secondElements);
    int totalOutputElements = Math.addExact(firstTwoElements, thirdElements);
    int inputElements = prepareGroupedWorkspace(input, batchSize, cols, totalOutputElements);
    configureGroupedProjection(0, firstWeights, firstType, firstRows, cols);
    configureGroupedProjection(1, secondWeights, secondType, secondRows, cols);
    configureGroupedProjection(2, thirdWeights, thirdType, thirdRows, cols);
    invokeGrouped(
        firstType, secondType, thirdType, inputElements, batchSize, cols, 3, totalOutputElements);
    copyOutput(firstOutput, 0, firstElements);
    copyOutput(secondOutput, firstElements, secondElements);
    copyOutput(thirdOutput, firstTwoElements, thirdElements);
  }

  @Override
  public synchronized void multiply(
      float[] output,
      float[] input,
      MemorySegment weights,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols) {
    requireOpen();
    Objects.requireNonNull(type, "type");
    if (!supports(type)) {
      throw new UnsupportedOperationException("Rust kernel does not support GGUF " + type);
    }
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(weights, "weights");
    int blockElements = blockElements(type);
    if (batchSize < 1 || rows < 1 || cols < 1 || cols % blockElements != 0) {
      throw new IllegalArgumentException(
          type
              + " batch and rows must be positive and cols must be a multiple of "
              + blockElements);
    }
    int inputElements = Math.multiplyExact(batchSize, cols);
    int outputElements = Math.multiplyExact(batchSize, rows);
    long weightBytes = requiredWeightBytes(type, rows, cols);
    if (input.length < inputElements
        || output.length < outputElements
        || weights.byteSize() < weightBytes) {
      throw new IllegalArgumentException(type + " projection buffers are smaller than its shape");
    }
    if (!weights.isNative()) {
      throw new IllegalArgumentException("weights must be backed by native or mapped memory");
    }

    ensureCapacity(inputElements, outputElements);
    MemorySegment.copy(input, 0, nativeInput, ValueLayout.JAVA_FLOAT, 0, inputElements);
    switch (type) {
      case Q4_0 ->
          library.q4_0F32BatchedMatmul(
              weights,
              weightBytes,
              nativeInput,
              inputElements,
              nativeOutput,
              outputElements,
              batchSize,
              rows,
              cols);
      case Q5_0 ->
          library.q5_0F32BatchedMatmul(
              weights,
              weightBytes,
              nativeInput,
              inputElements,
              nativeOutput,
              outputElements,
              batchSize,
              rows,
              cols);
      case Q8_0 ->
          library.q8_0F32BatchedMatmul(
              weights,
              weightBytes,
              nativeInput,
              inputElements,
              nativeOutput,
              outputElements,
              batchSize,
              rows,
              cols);
      case Q4_K ->
          library.q4_KF32BatchedMatmul(
              weights,
              weightBytes,
              nativeInput,
              inputElements,
              nativeOutput,
              outputElements,
              batchSize,
              rows,
              cols);
      case Q5_K ->
          library.q5_KF32BatchedMatmul(
              weights,
              weightBytes,
              nativeInput,
              inputElements,
              nativeOutput,
              outputElements,
              batchSize,
              rows,
              cols);
      case Q6_K ->
          library.q6_KF32BatchedMatmul(
              weights,
              weightBytes,
              nativeInput,
              inputElements,
              nativeOutput,
              outputElements,
              batchSize,
              rows,
              cols);
      default -> throw new AssertionError("validated unsupported GGUF type " + type);
    }
    MemorySegment.copy(nativeOutput, ValueLayout.JAVA_FLOAT, 0, output, 0, outputElements);
  }

  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    closed = true;
    RuntimeException closeFailure = null;
    if (scratchArena != null) {
      try {
        scratchArena.close();
      } catch (RuntimeException failure) {
        closeFailure = failure;
      }
    }
    try {
      library.close();
    } catch (RuntimeException failure) {
      if (closeFailure == null) {
        closeFailure = failure;
      } else {
        closeFailure.addSuppressed(failure);
      }
    }
    if (closeFailure != null) {
      throw closeFailure;
    }
  }

  private void ensureCapacity(int requiredInput, int requiredOutput) {
    if (requiredInput <= inputCapacity && requiredOutput <= outputCapacity) {
      return;
    }
    int newInputCapacity = Math.max(requiredInput, inputCapacity);
    int newOutputCapacity = Math.max(requiredOutput, outputCapacity);
    if (scratchArena != null) {
      scratchArena.close();
    }
    scratchArena = Arena.ofShared();
    nativeInput = scratchArena.allocate(ValueLayout.JAVA_FLOAT, newInputCapacity);
    nativeOutput = scratchArena.allocate(ValueLayout.JAVA_FLOAT, newOutputCapacity);
    nativeFormats = scratchArena.allocate(ValueLayout.JAVA_INT, 3);
    nativeWeightPointers = scratchArena.allocate(ValueLayout.ADDRESS, 3);
    nativeWeightBytes = scratchArena.allocate(ValueLayout.JAVA_LONG, 3);
    nativeRows = scratchArena.allocate(ValueLayout.JAVA_INT, 3);
    inputCapacity = newInputCapacity;
    outputCapacity = newOutputCapacity;
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("Rust GGUF kernel is closed");
    }
  }

  private boolean supportsGrouped(GgufTensorType type) {
    if (type == null) {
      return false;
    }
    return switch (type) {
      case Q4_0 -> library.supports(NativeKernelCapability.Q4_0_F32_GROUPED_BATCHED_MATMUL);
      case Q5_0 -> library.supports(NativeKernelCapability.Q5_0_F32_GROUPED_BATCHED_MATMUL);
      case Q8_0 -> library.supports(NativeKernelCapability.Q8_0_F32_GROUPED_BATCHED_MATMUL);
      case Q4_K -> library.supports(NativeKernelCapability.Q4_K_F32_GROUPED_BATCHED_MATMUL);
      case Q5_K -> library.supports(NativeKernelCapability.Q5_K_F32_GROUPED_BATCHED_MATMUL);
      case Q6_K -> library.supports(NativeKernelCapability.Q6_K_F32_GROUPED_BATCHED_MATMUL);
      default -> false;
    };
  }

  private boolean eligibleBatch(int batchSize) {
    return batchSize > 1 || (nativeDecode && batchSize == 1);
  }

  private int validateGroupedProjection(
      float[] output,
      MemorySegment weights,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(type, "type");
    if (!supportsGrouped(type)) {
      throw new UnsupportedOperationException("Rust grouped kernel does not support GGUF " + type);
    }
    int blockElements = blockElements(type);
    if (batchSize < 1 || rows < 1 || cols < 1 || cols % blockElements != 0) {
      throw new IllegalArgumentException(
          type
              + " batch and rows must be positive and cols must be a multiple of "
              + blockElements);
    }
    int outputElements = Math.multiplyExact(batchSize, rows);
    long requiredWeightBytes = requiredWeightBytes(type, rows, cols);
    if (output.length < outputElements || weights.byteSize() < requiredWeightBytes) {
      throw new IllegalArgumentException(type + " projection buffers are smaller than its shape");
    }
    if (!weights.isNative()) {
      throw new IllegalArgumentException("weights must be backed by native or mapped memory");
    }
    return outputElements;
  }

  private void configureGroupedProjection(
      int index, MemorySegment weights, GgufTensorType type, int rows, int cols) {
    long requiredWeightBytes = requiredWeightBytes(type, rows, cols);
    nativeFormats.setAtIndex(ValueLayout.JAVA_INT, index, formatCode(type));
    nativeWeightPointers.setAtIndex(ValueLayout.ADDRESS, index, weights);
    nativeWeightBytes.setAtIndex(ValueLayout.JAVA_LONG, index, requiredWeightBytes);
    nativeRows.setAtIndex(ValueLayout.JAVA_INT, index, rows);
  }

  private int prepareGroupedWorkspace(float[] input, int batchSize, int cols, int outputElements) {
    Objects.requireNonNull(input, "input");
    int inputElements = Math.multiplyExact(batchSize, cols);
    if (input.length < inputElements) {
      throw new IllegalArgumentException("quantized projection input is smaller than its shape");
    }
    ensureCapacity(inputElements, outputElements);
    MemorySegment.copy(input, 0, nativeInput, ValueLayout.JAVA_FLOAT, 0, inputElements);
    return inputElements;
  }

  private void invokeGrouped(
      GgufTensorType firstType,
      GgufTensorType secondType,
      GgufTensorType thirdType,
      int inputElements,
      int batchSize,
      int cols,
      int matrixCount,
      int outputElements) {
    boolean mixed = firstType != secondType || (thirdType != null && firstType != thirdType);
    NativeKernelCapability capability =
        mixed
            ? NativeKernelCapability.MIXED_K_F32_GROUPED_BATCHED_MATMUL
            : groupedCapability(firstType);
    String type = mixed ? "mixed K-quant" : firstType.toString();
    library.quantizedF32GroupedBatchedMatmul(
        type,
        capability,
        nativeFormats,
        nativeWeightPointers,
        nativeWeightBytes,
        nativeRows,
        matrixCount,
        nativeInput,
        inputElements,
        nativeOutput,
        outputElements,
        batchSize,
        cols);
  }

  private static long requiredWeightBytes(GgufTensorType type, int rows, int cols) {
    long blockBytes =
        switch (type) {
          case Q4_0 -> 18L;
          case Q5_0 -> 22L;
          case Q8_0 -> 34L;
          case Q4_K -> 144L;
          case Q5_K -> 176L;
          case Q6_K -> 210L;
          default -> throw new UnsupportedOperationException("unsupported GGUF type " + type);
        };
    return Math.multiplyExact(
        Math.multiplyExact((long) rows, cols / blockElements(type)), blockBytes);
  }

  private boolean compatibleGroup(GgufTensorType firstType, GgufTensorType secondType) {
    return firstType == secondType
        || (isKQuant(firstType)
            && isKQuant(secondType)
            && library.supports(NativeKernelCapability.MIXED_K_F32_GROUPED_BATCHED_MATMUL));
  }

  private boolean q5_0GroupEligible(GgufTensorType firstType, GgufTensorType secondType) {
    return q5_0Grouped || (firstType != GgufTensorType.Q5_0 && secondType != GgufTensorType.Q5_0);
  }

  private boolean q5_0GroupEligible(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    return q5_0Grouped
        || (firstType != GgufTensorType.Q5_0
            && secondType != GgufTensorType.Q5_0
            && thirdType != GgufTensorType.Q5_0);
  }

  private boolean compatibleGroup(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    return (firstType == secondType && firstType == thirdType)
        || (isKQuant(firstType)
            && isKQuant(secondType)
            && isKQuant(thirdType)
            && library.supports(NativeKernelCapability.MIXED_K_F32_GROUPED_BATCHED_MATMUL));
  }

  private static boolean isKQuant(GgufTensorType type) {
    return type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K
        || type == GgufTensorType.Q6_K;
  }

  private static NativeKernelCapability groupedCapability(GgufTensorType type) {
    return switch (type) {
      case Q4_0 -> NativeKernelCapability.Q4_0_F32_GROUPED_BATCHED_MATMUL;
      case Q5_0 -> NativeKernelCapability.Q5_0_F32_GROUPED_BATCHED_MATMUL;
      case Q8_0 -> NativeKernelCapability.Q8_0_F32_GROUPED_BATCHED_MATMUL;
      case Q4_K -> NativeKernelCapability.Q4_K_F32_GROUPED_BATCHED_MATMUL;
      case Q5_K -> NativeKernelCapability.Q5_K_F32_GROUPED_BATCHED_MATMUL;
      case Q6_K -> NativeKernelCapability.Q6_K_F32_GROUPED_BATCHED_MATMUL;
      default -> throw new UnsupportedOperationException("unsupported GGUF type " + type);
    };
  }

  private static int formatCode(GgufTensorType type) {
    return switch (type) {
      case Q4_0 -> 0;
      case Q8_0 -> 1;
      case Q4_K -> 2;
      case Q6_K -> 3;
      case Q5_K -> 4;
      case Q5_0 -> 5;
      default -> throw new UnsupportedOperationException("unsupported GGUF type " + type);
    };
  }

  private static int blockElements(GgufTensorType type) {
    return switch (type) {
      case Q4_0, Q5_0, Q8_0 -> 32;
      case Q4_K, Q5_K, Q6_K -> 256;
      default -> throw new UnsupportedOperationException("unsupported GGUF type " + type);
    };
  }

  private void copyOutput(float[] output, int nativeOffset, int outputElements) {
    MemorySegment.copy(
        nativeOutput,
        ValueLayout.JAVA_FLOAT,
        Math.multiplyExact((long) nativeOffset, Float.BYTES),
        output,
        0,
        outputElements);
  }
}
