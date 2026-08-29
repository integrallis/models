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
package com.integrallis.models.accelerator;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.MemorySegment;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Experimental prefill-only Q4_0 projection provider backed by Java-authored TornadoVM kernels. */
public final class TornadoGgufBatchedMatrixKernel implements GgufBatchedMatrixKernel {
  private static final int MINIMUM_BATCH = 4;
  private static final long MINIMUM_MATRIX_VALUES = 1_048_576L;
  private static final Map<String, String> PLAN_RECOMMENDATIONS =
      Map.of(
          PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY,
          "false",
          PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY,
          "false",
          PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
          "false");

  private final Map<ProjectionKey, ProjectionPlan> plans = new LinkedHashMap<>();
  private long calls;
  private long totalNanos;
  private boolean closed;

  @Override
  public String implementation() {
    return "tornadovm-java-q4-prefill-experiment";
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
        && batchSize >= MINIMUM_BATCH
        && rows > 0
        && cols > 0
        && (long) rows * cols >= MINIMUM_MATRIX_VALUES;
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
      throw new UnsupportedOperationException(
          "projection is not eligible for the Tornado experiment");
    }
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(weights, "weights");
    int expectedInput = Math.multiplyExact(batchSize, cols);
    int expectedOutput = Math.multiplyExact(batchSize, rows);
    if (input.length < expectedInput || output.length < expectedOutput) {
      throw new IllegalArgumentException("input or output storage does not match projection shape");
    }
    ProjectionKey key =
        new ProjectionKey(weights.address(), weights.byteSize(), batchSize, rows, cols);
    long started = System.nanoTime();
    ProjectionPlan plan =
        plans.computeIfAbsent(
            key,
            ignored ->
                new ProjectionPlan("q4-model-" + plans.size(), weights, batchSize, rows, cols));
    plan.execute(input, output);
    totalNanos += System.nanoTime() - started;
    calls++;
  }

  /** Number of distinct tensor/shape execution plans compiled or awaiting first compilation. */
  public synchronized int projectionPlanCount() {
    return plans.size();
  }

  /** Number of model projection calls routed through this experiment. */
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
    plans.clear();
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

  private record ProjectionKey(long address, long weightBytes, int batchSize, int rows, int cols) {}

  private static final class ProjectionPlan implements AutoCloseable {
    private final int batchSize;
    private final int rows;
    private final int cols;
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

    private void execute(float[] input, float[] output) {
      Q4ProjectionKernel.quantize(input, preparedActivations, preparedScales, batchSize, cols);
      deviceActivations.getSegment().copyFrom(MemorySegment.ofArray(preparedActivations));
      deviceScales.getSegment().copyFrom(MemorySegment.ofArray(preparedScales));
      plan.execute();
      MemorySegment.ofArray(output)
          .asSlice(0, Math.multiplyExact((long) batchSize * rows, Float.BYTES))
          .copyFrom(deviceOutput.getSegment());
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
}
