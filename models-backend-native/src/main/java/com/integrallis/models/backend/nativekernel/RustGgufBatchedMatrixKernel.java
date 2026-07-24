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

/** Reusable off-heap workspace for the Models Rust Q4_0 batched projection kernel. */
public final class RustGgufBatchedMatrixKernel implements GgufBatchedMatrixKernel {
  private static final Map<String, String> PLAN_RECOMMENDATIONS =
      Map.of(
          PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY,
          "true",
          PureJavaPlanConfiguration.MIXED_K_PROJECTIONS_PROPERTY,
          "false",
          PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY,
          "false",
          PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
          "false");

  private final NativeKernelLibrary library;
  private Arena scratchArena;
  private MemorySegment nativeInput = MemorySegment.NULL;
  private MemorySegment nativeOutput = MemorySegment.NULL;
  private MemorySegment nativeWeightPointers = MemorySegment.NULL;
  private MemorySegment nativeWeightBytes = MemorySegment.NULL;
  private MemorySegment nativeRows = MemorySegment.NULL;
  private int inputCapacity;
  private int outputCapacity;
  private boolean closed;

  private RustGgufBatchedMatrixKernel(NativeKernelLibrary library) {
    this.library = Objects.requireNonNull(library, "library");
  }

  /** Opens a Rust kernel provider from an explicit platform library. */
  public static RustGgufBatchedMatrixKernel open(Path libraryPath) {
    return new RustGgufBatchedMatrixKernel(NativeKernelLibrary.open(libraryPath));
  }

  @Override
  public String implementation() {
    return "rust-ffm-q4_0-v1";
  }

  @Override
  public Map<String, String> planRecommendations() {
    return PLAN_RECOMMENDATIONS;
  }

  @Override
  public boolean supports(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        && library.supports(NativeKernelCapability.Q4_0_F32_BATCHED_MATMUL);
  }

  @Override
  public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
    return batchSize > 1 && supports(type);
  }

  @Override
  public boolean isDualEligible(
      GgufTensorType firstType,
      int firstRows,
      GgufTensorType secondType,
      int secondRows,
      int batchSize,
      int cols) {
    return batchSize > 1 && supportsGrouped(firstType) && supportsGrouped(secondType);
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
    configureGroupedProjection(0, firstWeights, firstRows, cols);
    configureGroupedProjection(1, secondWeights, secondRows, cols);
    invokeGrouped(inputElements, batchSize, cols, 2, totalOutputElements);
    copyOutput(firstOutput, 0, firstElements);
    copyOutput(secondOutput, firstElements, secondElements);
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
    return batchSize > 1
        && supportsGrouped(firstType)
        && supportsGrouped(secondType)
        && supportsGrouped(thirdType);
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
    configureGroupedProjection(0, firstWeights, firstRows, cols);
    configureGroupedProjection(1, secondWeights, secondRows, cols);
    configureGroupedProjection(2, thirdWeights, thirdRows, cols);
    invokeGrouped(inputElements, batchSize, cols, 3, totalOutputElements);
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
    if (batchSize < 1 || rows < 1 || cols < 1 || cols % 32 != 0) {
      throw new IllegalArgumentException(
          "Q4_0 batch and rows must be positive and cols must be a multiple of 32");
    }
    int inputElements = Math.multiplyExact(batchSize, cols);
    int outputElements = Math.multiplyExact(batchSize, rows);
    long weightBytes = Math.multiplyExact(Math.multiplyExact((long) rows, cols / 32L), 18L);
    if (input.length < inputElements
        || output.length < outputElements
        || weights.byteSize() < weightBytes) {
      throw new IllegalArgumentException("Q4_0 projection buffers are smaller than its shape");
    }
    if (!weights.isNative()) {
      throw new IllegalArgumentException("weights must be backed by native or mapped memory");
    }

    ensureCapacity(inputElements, outputElements);
    MemorySegment.copy(input, 0, nativeInput, ValueLayout.JAVA_FLOAT, 0, inputElements);
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
    return type == GgufTensorType.Q4_0
        && library.supports(NativeKernelCapability.Q4_0_F32_GROUPED_BATCHED_MATMUL);
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
    if (batchSize < 1 || rows < 1 || cols < 1 || cols % 32 != 0) {
      throw new IllegalArgumentException(
          "Q4_0 batch and rows must be positive and cols must be a multiple of 32");
    }
    int outputElements = Math.multiplyExact(batchSize, rows);
    long requiredWeightBytes = Math.multiplyExact(Math.multiplyExact((long) rows, cols / 32L), 18L);
    if (output.length < outputElements || weights.byteSize() < requiredWeightBytes) {
      throw new IllegalArgumentException("Q4_0 projection buffers are smaller than its shape");
    }
    if (!weights.isNative()) {
      throw new IllegalArgumentException("weights must be backed by native or mapped memory");
    }
    return outputElements;
  }

  private void configureGroupedProjection(int index, MemorySegment weights, int rows, int cols) {
    long requiredWeightBytes = Math.multiplyExact(Math.multiplyExact((long) rows, cols / 32L), 18L);
    nativeWeightPointers.setAtIndex(ValueLayout.ADDRESS, index, weights);
    nativeWeightBytes.setAtIndex(ValueLayout.JAVA_LONG, index, requiredWeightBytes);
    nativeRows.setAtIndex(ValueLayout.JAVA_INT, index, rows);
  }

  private int prepareGroupedWorkspace(float[] input, int batchSize, int cols, int outputElements) {
    Objects.requireNonNull(input, "input");
    int inputElements = Math.multiplyExact(batchSize, cols);
    if (input.length < inputElements) {
      throw new IllegalArgumentException("Q4_0 projection input is smaller than its shape");
    }
    ensureCapacity(inputElements, outputElements);
    MemorySegment.copy(input, 0, nativeInput, ValueLayout.JAVA_FLOAT, 0, inputElements);
    return inputElements;
  }

  private void invokeGrouped(
      int inputElements, int batchSize, int cols, int matrixCount, int outputElements) {
    library.q4_0F32GroupedBatchedMatmul(
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
