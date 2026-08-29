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

import java.util.Locale;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Device correctness gate for the grouped Q4_0 projection dispatches. */
public final class Q4GroupedProjectionExperiment {
  private static final double MAX_RELATIVE_L2 = 2.0e-5;
  private static final int BATCH_SIZE = 3;
  private static final int COLS = 96;
  private static final int FIRST_ROWS = 11;
  private static final int SECOND_ROWS = 7;
  private static final int THIRD_ROWS = 5;

  private Q4GroupedProjectionExperiment() {}

  public static void main(String[] args) throws TornadoExecutionPlanException {
    byte[] firstWeights = Q4ProjectionExperiment.randomQ4Matrix(FIRST_ROWS, COLS, 41L);
    byte[] secondWeights = Q4ProjectionExperiment.randomQ4Matrix(SECOND_ROWS, COLS, 43L);
    byte[] thirdWeights = Q4ProjectionExperiment.randomQ4Matrix(THIRD_ROWS, COLS, 47L);
    float[] input = Q4ProjectionExperiment.randomFloats(BATCH_SIZE * COLS, 53L);
    byte[] activations = new byte[BATCH_SIZE * COLS];
    float[] scales = new float[BATCH_SIZE * COLS / 32];
    Q4ProjectionKernel.quantize(input, activations, scales, BATCH_SIZE, COLS);

    float[] expectedFirst =
        Q4ProjectionExperiment.vectorApiProjection(
            firstWeights, input, BATCH_SIZE, FIRST_ROWS, COLS);
    float[] expectedSecond =
        Q4ProjectionExperiment.vectorApiProjection(
            secondWeights, input, BATCH_SIZE, SECOND_ROWS, COLS);
    float[] expectedThird =
        Q4ProjectionExperiment.vectorApiProjection(
            thirdWeights, input, BATCH_SIZE, THIRD_ROWS, COLS);

    double[] dualErrors =
        runDual(firstWeights, secondWeights, activations, scales, expectedFirst, expectedSecond);
    double[] tripleErrors =
        runTriple(
            firstWeights,
            secondWeights,
            thirdWeights,
            activations,
            scales,
            expectedFirst,
            expectedSecond,
            expectedThird);
    System.out.printf(
        Locale.ROOT, "dual relative L2   first=%.8g second=%.8g%n", dualErrors[0], dualErrors[1]);
    System.out.printf(
        Locale.ROOT,
        "triple relative L2 first=%.8g second=%.8g third=%.8g%n",
        tripleErrors[0],
        tripleErrors[1],
        tripleErrors[2]);
  }

  private static double[] runDual(
      byte[] firstWeights,
      byte[] secondWeights,
      byte[] activations,
      float[] scales,
      float[] expectedFirst,
      float[] expectedSecond)
      throws TornadoExecutionPlanException {
    ByteArray deviceFirstWeights = ByteArray.fromArray(firstWeights);
    ByteArray deviceSecondWeights = ByteArray.fromArray(secondWeights);
    ByteArray deviceActivations = ByteArray.fromArray(activations);
    FloatArray deviceScales = FloatArray.fromArray(scales);
    FloatArray firstOutput = new FloatArray(BATCH_SIZE * FIRST_ROWS);
    FloatArray secondOutput = new FloatArray(BATCH_SIZE * SECOND_ROWS);
    TaskGraph graph =
        new TaskGraph("q4-dual-projection")
            .transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                deviceFirstWeights,
                deviceSecondWeights,
                deviceActivations,
                deviceScales)
            .task(
                "multiply-dual",
                Q4ProjectionKernel::multiplyDual,
                deviceFirstWeights,
                FIRST_ROWS,
                deviceSecondWeights,
                SECOND_ROWS,
                deviceActivations,
                deviceScales,
                firstOutput,
                secondOutput,
                BATCH_SIZE,
                COLS)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, firstOutput, secondOutput);
    try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
      plan.execute();
    }
    return checkedErrors(
        new float[][] {expectedFirst, expectedSecond},
        new FloatArray[] {firstOutput, secondOutput},
        "dual");
  }

  private static double[] runTriple(
      byte[] firstWeights,
      byte[] secondWeights,
      byte[] thirdWeights,
      byte[] activations,
      float[] scales,
      float[] expectedFirst,
      float[] expectedSecond,
      float[] expectedThird)
      throws TornadoExecutionPlanException {
    ByteArray deviceFirstWeights = ByteArray.fromArray(firstWeights);
    ByteArray deviceSecondWeights = ByteArray.fromArray(secondWeights);
    ByteArray deviceThirdWeights = ByteArray.fromArray(thirdWeights);
    ByteArray deviceActivations = ByteArray.fromArray(activations);
    FloatArray deviceScales = FloatArray.fromArray(scales);
    FloatArray firstOutput = new FloatArray(BATCH_SIZE * FIRST_ROWS);
    FloatArray secondOutput = new FloatArray(BATCH_SIZE * SECOND_ROWS);
    FloatArray thirdOutput = new FloatArray(BATCH_SIZE * THIRD_ROWS);
    TaskGraph graph =
        new TaskGraph("q4-triple-projection")
            .transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                deviceFirstWeights,
                deviceSecondWeights,
                deviceThirdWeights,
                deviceActivations,
                deviceScales)
            .task(
                "multiply-triple",
                Q4ProjectionKernel::multiplyTriple,
                deviceFirstWeights,
                FIRST_ROWS,
                deviceSecondWeights,
                SECOND_ROWS,
                deviceThirdWeights,
                THIRD_ROWS,
                deviceActivations,
                deviceScales,
                firstOutput,
                secondOutput,
                thirdOutput,
                BATCH_SIZE,
                COLS)
            .transferToHost(
                DataTransferMode.EVERY_EXECUTION, firstOutput, secondOutput, thirdOutput);
    try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
      plan.execute();
    }
    return checkedErrors(
        new float[][] {expectedFirst, expectedSecond, expectedThird},
        new FloatArray[] {firstOutput, secondOutput, thirdOutput},
        "triple");
  }

  private static double[] checkedErrors(
      float[][] expected, FloatArray[] actual, String projectionKind) {
    double[] errors = new double[expected.length];
    for (int projection = 0; projection < expected.length; projection++) {
      errors[projection] =
          Q4ProjectionExperiment.relativeL2(expected[projection], actual[projection].toHeapArray());
      if (errors[projection] > MAX_RELATIVE_L2) {
        throw new IllegalStateException(
            projectionKind
                + " projection "
                + projection
                + " failed relative-L2 gate: "
                + errors[projection]);
      }
    }
    return errors;
  }
}
