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
package com.integrallis.models.backend.tornado;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Q4_0 prefill and optional decode projections backed by Java-authored TornadoVM kernels. */
public final class TornadoGgufBatchedMatrixKernel implements GgufBatchedMatrixKernel {
  private static final int MINIMUM_BATCH = 4;
  private static final int DEFAULT_EXECUTION_BATCH_SIZE = 32;
  private static final long MINIMUM_MATRIX_VALUES = 1_048_576L;
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

  private final Map<ProjectionKey, ProjectionPlan> plans = new LinkedHashMap<>();
  private final Map<DualProjectionKey, DualProjectionPlan> dualPlans = new LinkedHashMap<>();
  private final Map<TripleProjectionKey, TripleProjectionPlan> triplePlans = new LinkedHashMap<>();
  private final int executionBatchSize;
  private final boolean accelerateDecode;
  private int planSequence;
  private long calls;
  private long totalNanos;
  private boolean closed;

  public TornadoGgufBatchedMatrixKernel() {
    this(DEFAULT_EXECUTION_BATCH_SIZE, false);
  }

  public TornadoGgufBatchedMatrixKernel(int executionBatchSize) {
    this(executionBatchSize, false);
  }

  public TornadoGgufBatchedMatrixKernel(int executionBatchSize, boolean accelerateDecode) {
    if (executionBatchSize < MINIMUM_BATCH) {
      throw new IllegalArgumentException("executionBatchSize must be at least " + MINIMUM_BATCH);
    }
    this.executionBatchSize = executionBatchSize;
    this.accelerateDecode = accelerateDecode;
  }

  /** Fixed device batch shape used to make compiled plans reusable across prompt lengths. */
  public int executionBatchSize() {
    return executionBatchSize;
  }

  /** Whether this provider creates separate single-token projection plans for decode. */
  public boolean acceleratesDecode() {
    return accelerateDecode;
  }

  int executionBatchSizeFor(int actualBatchSize) {
    return accelerateDecode && actualBatchSize == 1 ? 1 : executionBatchSize;
  }

  @Override
  public String implementation() {
    return accelerateDecode ? "tornadovm-java-q4-prefill-decode" : "tornadovm-java-q4-prefill";
  }

  @Override
  public Map<String, String> planRecommendations() {
    return PLAN_RECOMMENDATIONS;
  }

  @Override
  public boolean supports(GgufTensorType type) {
    return type == GgufTensorType.Q4_0;
  }

  @Override
  public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
    return supports(type)
        && eligibleBatchSize(batchSize)
        && batchSize <= executionBatchSize
        && rows > 0
        && cols > 0
        && (long) rows * cols >= MINIMUM_MATRIX_VALUES;
  }

  @Override
  public boolean supportsDual(GgufTensorType firstType, GgufTensorType secondType) {
    return supports(firstType) && supports(secondType);
  }

  @Override
  public boolean isDualEligible(
      GgufTensorType firstType,
      int firstRows,
      GgufTensorType secondType,
      int secondRows,
      int batchSize,
      int cols) {
    return supportsDual(firstType, secondType)
        && eligibleCombined(batchSize, cols, firstRows, secondRows);
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
    if (!isDualEligible(firstType, firstRows, secondType, secondRows, batchSize, cols)) {
      throw new UnsupportedOperationException(
          "dual projection is not eligible for the Tornado backend");
    }
    validateProjectionStorage(firstOutput, firstWeights, input, batchSize, firstRows, cols);
    validateProjectionStorage(secondOutput, secondWeights, input, batchSize, secondRows, cols);
    int planBatchSize = executionBatchSizeFor(batchSize);
    DualProjectionKey key =
        new DualProjectionKey(
            firstWeights.address(),
            firstWeights.byteSize(),
            firstRows,
            secondWeights.address(),
            secondWeights.byteSize(),
            secondRows,
            planBatchSize,
            cols);
    long started = System.nanoTime();
    DualProjectionPlan plan =
        dualPlans.computeIfAbsent(
            key,
            ignored ->
                new DualProjectionPlan(
                    nextPlanName(),
                    firstWeights,
                    firstRows,
                    secondWeights,
                    secondRows,
                    planBatchSize,
                    cols));
    plan.execute(input, firstOutput, secondOutput, batchSize);
    recordCall(started);
  }

  @Override
  public boolean supportsTriple(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    return supports(firstType) && supports(secondType) && supports(thirdType);
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
    return supportsTriple(firstType, secondType, thirdType)
        && eligibleCombined(batchSize, cols, firstRows, secondRows, thirdRows);
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
    if (!isTripleEligible(
        firstType, firstRows, secondType, secondRows, thirdType, thirdRows, batchSize, cols)) {
      throw new UnsupportedOperationException(
          "triple projection is not eligible for the Tornado backend");
    }
    validateProjectionStorage(firstOutput, firstWeights, input, batchSize, firstRows, cols);
    validateProjectionStorage(secondOutput, secondWeights, input, batchSize, secondRows, cols);
    validateProjectionStorage(thirdOutput, thirdWeights, input, batchSize, thirdRows, cols);
    int planBatchSize = executionBatchSizeFor(batchSize);
    TripleProjectionKey key =
        new TripleProjectionKey(
            firstWeights.address(),
            firstWeights.byteSize(),
            firstRows,
            secondWeights.address(),
            secondWeights.byteSize(),
            secondRows,
            thirdWeights.address(),
            thirdWeights.byteSize(),
            thirdRows,
            planBatchSize,
            cols);
    long started = System.nanoTime();
    TripleProjectionPlan plan =
        triplePlans.computeIfAbsent(
            key,
            ignored ->
                new TripleProjectionPlan(
                    nextPlanName(),
                    firstWeights,
                    firstRows,
                    secondWeights,
                    secondRows,
                    thirdWeights,
                    thirdRows,
                    planBatchSize,
                    cols));
    plan.execute(input, firstOutput, secondOutput, thirdOutput, batchSize);
    recordCall(started);
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
    if (!isEligible(type, batchSize, rows, cols)) {
      throw new UnsupportedOperationException("projection is not eligible for the Tornado backend");
    }
    validateProjectionStorage(output, weights, input, batchSize, rows, cols);
    int planBatchSize = executionBatchSizeFor(batchSize);
    ProjectionKey key =
        new ProjectionKey(weights.address(), weights.byteSize(), planBatchSize, rows, cols);
    long started = System.nanoTime();
    ProjectionPlan plan =
        plans.computeIfAbsent(
            key, ignored -> new ProjectionPlan(nextPlanName(), weights, planBatchSize, rows, cols));
    plan.execute(input, output, batchSize);
    recordCall(started);
  }

  /** Number of distinct tensor/shape execution plans compiled or awaiting first compilation. */
  public synchronized int projectionPlanCount() {
    return plans.size() + dualPlans.size() + triplePlans.size();
  }

  /** Number of model projection calls routed through this provider. */
  public synchronized long calls() {
    return calls;
  }

  /** Wall-clock time spent preparing, compiling, executing, and copying accelerated calls. */
  public synchronized double totalMillis() {
    return totalNanos / 1_000_000.0;
  }

  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    RuntimeException failure = null;
    for (ProjectionPlan plan : plans.values()) {
      try {
        plan.close();
      } catch (RuntimeException exception) {
        if (failure == null) {
          failure = exception;
        } else {
          failure.addSuppressed(exception);
        }
      }
    }
    for (DualProjectionPlan plan : dualPlans.values()) {
      try {
        plan.close();
      } catch (RuntimeException exception) {
        if (failure == null) {
          failure = exception;
        } else {
          failure.addSuppressed(exception);
        }
      }
    }
    for (TripleProjectionPlan plan : triplePlans.values()) {
      try {
        plan.close();
      } catch (RuntimeException exception) {
        if (failure == null) {
          failure = exception;
        } else {
          failure.addSuppressed(exception);
        }
      }
    }
    plans.clear();
    dualPlans.clear();
    triplePlans.clear();
    closed = true;
    if (failure != null) {
      throw failure;
    }
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("Tornado projection kernel is closed");
    }
  }

  private boolean eligibleCombined(int batchSize, int cols, int... rows) {
    if (!eligibleBatchSize(batchSize) || batchSize > executionBatchSize || cols <= 0) {
      return false;
    }
    long combinedRows = 0;
    for (int rowCount : rows) {
      if (rowCount <= 0) {
        return false;
      }
      combinedRows += rowCount;
    }
    return combinedRows * cols >= MINIMUM_MATRIX_VALUES;
  }

  private boolean eligibleBatchSize(int batchSize) {
    return batchSize >= MINIMUM_BATCH || (accelerateDecode && batchSize == 1);
  }

  private static void validateProjectionStorage(
      float[] output, MemorySegment weights, float[] input, int batchSize, int rows, int cols) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(weights, "weights");
    int expectedInput = Math.multiplyExact(batchSize, cols);
    int expectedOutput = Math.multiplyExact(batchSize, rows);
    if (input.length < expectedInput || output.length < expectedOutput) {
      throw new IllegalArgumentException("input or output storage does not match projection shape");
    }
  }

  private String nextPlanName() {
    return "q4-model-" + planSequence++;
  }

  private void recordCall(long started) {
    totalNanos += System.nanoTime() - started;
    calls++;
  }

  private record ProjectionKey(long address, long weightBytes, int batchSize, int rows, int cols) {}

  private record DualProjectionKey(
      long firstAddress,
      long firstWeightBytes,
      int firstRows,
      long secondAddress,
      long secondWeightBytes,
      int secondRows,
      int batchSize,
      int cols) {}

  private record TripleProjectionKey(
      long firstAddress,
      long firstWeightBytes,
      int firstRows,
      long secondAddress,
      long secondWeightBytes,
      int secondRows,
      long thirdAddress,
      long thirdWeightBytes,
      int thirdRows,
      int batchSize,
      int cols) {}

  private static final class ProjectionPlan implements AutoCloseable {
    private final int batchSize;
    private final int rows;
    private final int cols;
    private final float[] paddedInput;
    private final byte[] preparedActivations;
    private final float[] preparedScales;
    private final ByteArray deviceActivations;
    private final FloatArray deviceScales;
    private final FloatArray deviceOutput;
    private final TornadoExecutionPlan plan;

    private ProjectionPlan(String name, MemorySegment weights, int batchSize, int rows, int cols) {
      this.batchSize = batchSize;
      this.rows = rows;
      this.cols = cols;
      int activationEntries = Math.multiplyExact(batchSize, cols);
      this.paddedInput = new float[activationEntries];
      this.preparedActivations = new byte[activationEntries];
      this.preparedScales = new float[activationEntries / 32];
      ByteArray deviceWeights = ByteArray.fromSegment(weights);
      this.deviceActivations = new ByteArray(activationEntries);
      this.deviceScales = new FloatArray(preparedScales.length);
      this.deviceOutput = new FloatArray(Math.multiplyExact(batchSize, rows));
      Q4ProjectionKernel.validate(
          deviceWeights, deviceActivations, deviceScales, deviceOutput, batchSize, rows, cols);
      TaskGraph graph =
          new TaskGraph(name)
              .transferToDevice(DataTransferMode.FIRST_EXECUTION, deviceWeights)
              .transferToDevice(DataTransferMode.EVERY_EXECUTION, deviceActivations, deviceScales)
              .task(
                  "multiply",
                  Q4ProjectionKernel::multiply,
                  deviceWeights,
                  deviceActivations,
                  deviceScales,
                  deviceOutput,
                  batchSize,
                  rows,
                  cols)
              .transferToHost(DataTransferMode.EVERY_EXECUTION, deviceOutput);
      this.plan = new TornadoExecutionPlan(graph.snapshot());
    }

    private void execute(float[] input, float[] output, int actualBatchSize) {
      float[] executionInput =
          prepareExecutionInput(input, paddedInput, actualBatchSize, batchSize, cols);
      Q4ProjectionKernel.quantize(
          executionInput, preparedActivations, preparedScales, batchSize, cols);
      deviceActivations.getSegment().copyFrom(MemorySegment.ofArray(preparedActivations));
      deviceScales.getSegment().copyFrom(MemorySegment.ofArray(preparedScales));
      plan.execute();
      copyOutput(deviceOutput, output, actualBatchSize, rows);
    }

    @Override
    public void close() {
      try {
        plan.close();
      } catch (Exception exception) {
        throw new IllegalStateException("could not close Tornado projection plan", exception);
      }
    }
  }

  private static final class DualProjectionPlan implements AutoCloseable {
    private final int batchSize;
    private final int firstRows;
    private final int secondRows;
    private final int cols;
    private final float[] paddedInput;
    private final byte[] preparedActivations;
    private final float[] preparedScales;
    private final ByteArray deviceActivations;
    private final FloatArray deviceScales;
    private final FloatArray firstDeviceOutput;
    private final FloatArray secondDeviceOutput;
    private final TornadoExecutionPlan plan;

    private DualProjectionPlan(
        String name,
        MemorySegment firstWeights,
        int firstRows,
        MemorySegment secondWeights,
        int secondRows,
        int batchSize,
        int cols) {
      this.batchSize = batchSize;
      this.firstRows = firstRows;
      this.secondRows = secondRows;
      this.cols = cols;
      int activationEntries = Math.multiplyExact(batchSize, cols);
      this.paddedInput = new float[activationEntries];
      this.preparedActivations = new byte[activationEntries];
      this.preparedScales = new float[activationEntries / 32];
      ByteArray firstDeviceWeights = ByteArray.fromSegment(firstWeights);
      ByteArray secondDeviceWeights = ByteArray.fromSegment(secondWeights);
      this.deviceActivations = new ByteArray(activationEntries);
      this.deviceScales = new FloatArray(preparedScales.length);
      this.firstDeviceOutput = new FloatArray(Math.multiplyExact(batchSize, firstRows));
      this.secondDeviceOutput = new FloatArray(Math.multiplyExact(batchSize, secondRows));
      Q4ProjectionKernel.validate(
          firstDeviceWeights,
          deviceActivations,
          deviceScales,
          firstDeviceOutput,
          batchSize,
          firstRows,
          cols);
      Q4ProjectionKernel.validate(
          secondDeviceWeights,
          deviceActivations,
          deviceScales,
          secondDeviceOutput,
          batchSize,
          secondRows,
          cols);
      TaskGraph graph =
          new TaskGraph(name)
              .transferToDevice(
                  DataTransferMode.FIRST_EXECUTION, firstDeviceWeights, secondDeviceWeights)
              .transferToDevice(DataTransferMode.EVERY_EXECUTION, deviceActivations, deviceScales)
              .task(
                  "multiply-dual",
                  Q4ProjectionKernel::multiplyDual,
                  firstDeviceWeights,
                  firstRows,
                  secondDeviceWeights,
                  secondRows,
                  deviceActivations,
                  deviceScales,
                  firstDeviceOutput,
                  secondDeviceOutput,
                  batchSize,
                  cols)
              .transferToHost(
                  DataTransferMode.EVERY_EXECUTION, firstDeviceOutput, secondDeviceOutput);
      this.plan = new TornadoExecutionPlan(graph.snapshot());
    }

    private void execute(
        float[] input, float[] firstOutput, float[] secondOutput, int actualBatchSize) {
      float[] executionInput =
          prepareExecutionInput(input, paddedInput, actualBatchSize, batchSize, cols);
      prepareAndStage(
          executionInput,
          preparedActivations,
          preparedScales,
          deviceActivations,
          deviceScales,
          batchSize,
          cols);
      plan.execute();
      copyOutput(firstDeviceOutput, firstOutput, actualBatchSize, firstRows);
      copyOutput(secondDeviceOutput, secondOutput, actualBatchSize, secondRows);
    }

    @Override
    public void close() {
      closePlan(plan);
    }
  }

  private static final class TripleProjectionPlan implements AutoCloseable {
    private final int batchSize;
    private final int firstRows;
    private final int secondRows;
    private final int thirdRows;
    private final int cols;
    private final float[] paddedInput;
    private final byte[] preparedActivations;
    private final float[] preparedScales;
    private final ByteArray deviceActivations;
    private final FloatArray deviceScales;
    private final FloatArray firstDeviceOutput;
    private final FloatArray secondDeviceOutput;
    private final FloatArray thirdDeviceOutput;
    private final TornadoExecutionPlan plan;

    private TripleProjectionPlan(
        String name,
        MemorySegment firstWeights,
        int firstRows,
        MemorySegment secondWeights,
        int secondRows,
        MemorySegment thirdWeights,
        int thirdRows,
        int batchSize,
        int cols) {
      this.batchSize = batchSize;
      this.firstRows = firstRows;
      this.secondRows = secondRows;
      this.thirdRows = thirdRows;
      this.cols = cols;
      int activationEntries = Math.multiplyExact(batchSize, cols);
      this.paddedInput = new float[activationEntries];
      this.preparedActivations = new byte[activationEntries];
      this.preparedScales = new float[activationEntries / 32];
      ByteArray firstDeviceWeights = ByteArray.fromSegment(firstWeights);
      ByteArray secondDeviceWeights = ByteArray.fromSegment(secondWeights);
      ByteArray thirdDeviceWeights = ByteArray.fromSegment(thirdWeights);
      this.deviceActivations = new ByteArray(activationEntries);
      this.deviceScales = new FloatArray(preparedScales.length);
      this.firstDeviceOutput = new FloatArray(Math.multiplyExact(batchSize, firstRows));
      this.secondDeviceOutput = new FloatArray(Math.multiplyExact(batchSize, secondRows));
      this.thirdDeviceOutput = new FloatArray(Math.multiplyExact(batchSize, thirdRows));
      Q4ProjectionKernel.validate(
          firstDeviceWeights,
          deviceActivations,
          deviceScales,
          firstDeviceOutput,
          batchSize,
          firstRows,
          cols);
      Q4ProjectionKernel.validate(
          secondDeviceWeights,
          deviceActivations,
          deviceScales,
          secondDeviceOutput,
          batchSize,
          secondRows,
          cols);
      Q4ProjectionKernel.validate(
          thirdDeviceWeights,
          deviceActivations,
          deviceScales,
          thirdDeviceOutput,
          batchSize,
          thirdRows,
          cols);
      TaskGraph graph =
          new TaskGraph(name)
              .transferToDevice(
                  DataTransferMode.FIRST_EXECUTION,
                  firstDeviceWeights,
                  secondDeviceWeights,
                  thirdDeviceWeights)
              .transferToDevice(DataTransferMode.EVERY_EXECUTION, deviceActivations, deviceScales)
              .task(
                  "multiply-triple",
                  Q4ProjectionKernel::multiplyTriple,
                  firstDeviceWeights,
                  firstRows,
                  secondDeviceWeights,
                  secondRows,
                  thirdDeviceWeights,
                  thirdRows,
                  deviceActivations,
                  deviceScales,
                  firstDeviceOutput,
                  secondDeviceOutput,
                  thirdDeviceOutput,
                  batchSize,
                  cols)
              .transferToHost(
                  DataTransferMode.EVERY_EXECUTION,
                  firstDeviceOutput,
                  secondDeviceOutput,
                  thirdDeviceOutput);
      this.plan = new TornadoExecutionPlan(graph.snapshot());
    }

    private void execute(
        float[] input,
        float[] firstOutput,
        float[] secondOutput,
        float[] thirdOutput,
        int actualBatchSize) {
      float[] executionInput =
          prepareExecutionInput(input, paddedInput, actualBatchSize, batchSize, cols);
      prepareAndStage(
          executionInput,
          preparedActivations,
          preparedScales,
          deviceActivations,
          deviceScales,
          batchSize,
          cols);
      plan.execute();
      copyOutput(firstDeviceOutput, firstOutput, actualBatchSize, firstRows);
      copyOutput(secondDeviceOutput, secondOutput, actualBatchSize, secondRows);
      copyOutput(thirdDeviceOutput, thirdOutput, actualBatchSize, thirdRows);
    }

    @Override
    public void close() {
      closePlan(plan);
    }
  }

  private static void prepareAndStage(
      float[] input,
      byte[] preparedActivations,
      float[] preparedScales,
      ByteArray deviceActivations,
      FloatArray deviceScales,
      int batchSize,
      int cols) {
    Q4ProjectionKernel.quantize(input, preparedActivations, preparedScales, batchSize, cols);
    deviceActivations.getSegment().copyFrom(MemorySegment.ofArray(preparedActivations));
    deviceScales.getSegment().copyFrom(MemorySegment.ofArray(preparedScales));
  }

  static float[] prepareExecutionInput(
      float[] input, float[] paddedInput, int actualBatchSize, int executionBatchSize, int cols) {
    int actualEntries = Math.multiplyExact(actualBatchSize, cols);
    int executionEntries = Math.multiplyExact(executionBatchSize, cols);
    if (input.length < actualEntries || paddedInput.length != executionEntries) {
      throw new IllegalArgumentException("input storage does not match execution batch shape");
    }
    System.arraycopy(input, 0, paddedInput, 0, actualEntries);
    Arrays.fill(paddedInput, actualEntries, executionEntries, 0.0f);
    return paddedInput;
  }

  private static void copyOutput(FloatArray deviceOutput, float[] output, int batchSize, int rows) {
    long byteSize = Math.multiplyExact((long) batchSize * rows, Float.BYTES);
    MemorySegment.ofArray(output)
        .asSlice(0, byteSize)
        .copyFrom(deviceOutput.getSegment().asSlice(0, byteSize));
  }

  private static void closePlan(TornadoExecutionPlan plan) {
    try {
      plan.close();
    } catch (Exception exception) {
      throw new IllegalStateException("could not close Tornado projection plan", exception);
    }
  }
}
