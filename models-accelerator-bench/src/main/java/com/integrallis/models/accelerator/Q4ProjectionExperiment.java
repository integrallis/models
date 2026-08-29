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

import com.integrallis.models.backend.tornado.Q4ProjectionKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Locale;
import java.util.SplittableRandom;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Measures one isolated, production-compatible Java-authored Q4_0 by Q8_0 projection. */
public final class Q4ProjectionExperiment {
  private static final int BLOCK_VALUES = 32;
  private static final int BLOCK_BYTES = 18;

  private Q4ProjectionExperiment() {}

  public static void main(String[] args) throws TornadoExecutionPlanException {
    Configuration configuration = Configuration.parse(args);
    int activationEntries = Math.multiplyExact(configuration.batchSize(), configuration.cols());
    int scaleEntries = activationEntries / BLOCK_VALUES;
    byte[] weights = randomQ4Matrix(configuration.rows(), configuration.cols(), 31L);
    float[] input = randomFloats(activationEntries, 37L);
    float[] expected =
        vectorApiProjection(
            weights, input, configuration.batchSize(), configuration.rows(), configuration.cols());
    byte[] preparedActivations = new byte[activationEntries];
    float[] preparedScales = new float[scaleEntries];
    Q4ProjectionKernel.quantize(
        input,
        preparedActivations,
        preparedScales,
        configuration.batchSize(),
        configuration.cols());

    ByteArray deviceWeights = ByteArray.fromArray(weights);
    ByteArray deviceActivations = ByteArray.fromArray(preparedActivations);
    FloatArray deviceScales = FloatArray.fromArray(preparedScales);
    FloatArray deviceOutput =
        new FloatArray(Math.multiplyExact(configuration.batchSize(), configuration.rows()));
    Q4ProjectionKernel.validate(
        deviceWeights,
        deviceActivations,
        deviceScales,
        deviceOutput,
        configuration.batchSize(),
        configuration.rows(),
        configuration.cols());

    TaskGraph graph =
        new TaskGraph("q4-projection")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, deviceWeights)
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, deviceActivations, deviceScales)
            .task(
                "multiply",
                Q4ProjectionKernel::multiply,
                deviceWeights,
                deviceActivations,
                deviceScales,
                deviceOutput,
                configuration.batchSize(),
                configuration.rows(),
                configuration.cols())
            .transferToHost(DataTransferMode.EVERY_EXECUTION, deviceOutput);

    long coldNanos;
    long[] warmNanos = new long[configuration.iterations()];
    long[] preparationNanos = new long[configuration.iterations()];
    long[] endToEndNanos = new long[configuration.iterations()];
    TornadoExecutionResult lastResult;
    try (TornadoExecutionPlan plan =
        new TornadoExecutionPlan(graph.snapshot()).withProfiler(ProfilerMode.SILENT)) {
      long started = System.nanoTime();
      lastResult = plan.execute();
      coldNanos = System.nanoTime() - started;
      for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
        prepareAndStage(
            input,
            preparedActivations,
            preparedScales,
            deviceActivations,
            deviceScales,
            configuration.batchSize(),
            configuration.cols());
        plan.execute();
      }
      for (int iteration = 0; iteration < warmNanos.length; iteration++) {
        long endToEndStarted = System.nanoTime();
        started = System.nanoTime();
        prepareAndStage(
            input,
            preparedActivations,
            preparedScales,
            deviceActivations,
            deviceScales,
            configuration.batchSize(),
            configuration.cols());
        preparationNanos[iteration] = System.nanoTime() - started;
        started = System.nanoTime();
        lastResult = plan.execute();
        warmNanos[iteration] = System.nanoTime() - started;
        endToEndNanos[iteration] = System.nanoTime() - endToEndStarted;
      }
    }

    float[] actual = deviceOutput.toHeapArray();
    double relativeL2 = relativeL2(expected, actual);
    if (relativeL2 > 2.0e-5) {
      throw new IllegalStateException(
          "accelerator result failed relative-L2 gate: "
              + relativeL2
              + ", expected[0]="
              + expected[0]
              + ", actual[0]="
              + actual[0]);
    }

    long[] cpuNanos =
        timeCpu(
            weights,
            input,
            configuration.batchSize(),
            configuration.rows(),
            configuration.cols(),
            configuration.warmups(),
            configuration.iterations());
    long deviceMedian = median(warmNanos);
    long preparationMedian = median(preparationNanos);
    long endToEndMedian = median(endToEndNanos);
    long cpuMedian = median(cpuNanos);

    System.out.printf(
        Locale.ROOT,
        "shape             batch=%d rows=%d cols=%d%n",
        configuration.batchSize(),
        configuration.rows(),
        configuration.cols());
    System.out.printf(
        Locale.ROOT,
        "weights           %.2f MiB, persistent after first execution%n",
        weights.length / (1024.0 * 1024.0));
    System.out.printf(Locale.ROOT, "correctness       relative L2 %.8g%n", relativeL2);
    System.out.printf(Locale.ROOT, "cold device       %.3f ms%n", coldNanos / 1_000_000.0);
    System.out.printf(Locale.ROOT, "host Q8 staging   %.3f ms%n", preparationMedian / 1_000_000.0);
    System.out.printf(Locale.ROOT, "warm device p50   %.3f ms%n", deviceMedian / 1_000_000.0);
    System.out.printf(Locale.ROOT, "end-to-end p50    %.3f ms%n", endToEndMedian / 1_000_000.0);
    System.out.printf(Locale.ROOT, "Vector API p50    %.3f ms%n", cpuMedian / 1_000_000.0);
    System.out.printf(
        Locale.ROOT, "speedup           %.2fx%n", (double) cpuMedian / endToEndMedian);
    System.out.printf(
        Locale.ROOT,
        "last kernel       %.3f ms%n",
        lastResult.getProfilerResult().getDeviceKernelTime() / 1_000_000.0);
    System.out.printf(
        Locale.ROOT,
        "last transfers    %.3f ms%n",
        lastResult.getProfilerResult().getDataTransfersTime() / 1_000_000.0);
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

  private static long[] timeCpu(
      byte[] weights,
      float[] input,
      int batchSize,
      int rows,
      int cols,
      int warmups,
      int iterations) {
    MemorySegment weightSegment = MemorySegment.ofArray(weights);
    float[] output = new float[Math.multiplyExact(batchSize, rows)];
    byte[] quants = new byte[Math.multiplyExact(batchSize, cols)];
    float[] scales = new float[Math.multiplyExact(batchSize, cols / BLOCK_VALUES)];
    int[] corrections = new int[Math.multiplyExact(batchSize, (cols + 3) / 4)];
    float[] lanes = new float[Math.multiplyExact(output.length, 8)];
    for (int warmup = 0; warmup < warmups; warmup++) {
      VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
          input,
          weightSegment,
          batchSize,
          rows,
          cols,
          output,
          quants,
          scales,
          corrections,
          lanes,
          GgufQ4Kernel.WIDENED);
    }
    long[] nanos = new long[iterations];
    for (int iteration = 0; iteration < iterations; iteration++) {
      long started = System.nanoTime();
      VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
          input,
          weightSegment,
          batchSize,
          rows,
          cols,
          output,
          quants,
          scales,
          corrections,
          lanes,
          GgufQ4Kernel.WIDENED);
      nanos[iteration] = System.nanoTime() - started;
    }
    return nanos;
  }

  static float[] vectorApiProjection(
      byte[] weights, float[] input, int batchSize, int rows, int cols) {
    float[] output = new float[Math.multiplyExact(batchSize, rows)];
    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        input,
        MemorySegment.ofArray(weights),
        batchSize,
        rows,
        cols,
        output,
        new byte[Math.multiplyExact(batchSize, cols)],
        new float[Math.multiplyExact(batchSize, cols / BLOCK_VALUES)],
        new int[Math.multiplyExact(batchSize, (cols + 3) / 4)],
        new float[Math.multiplyExact(output.length, 8)],
        GgufQ4Kernel.WIDENED);
    return output;
  }

  static double relativeL2(float[] expected, float[] actual) {
    double squaredError = 0.0;
    double squaredReference = 0.0;
    for (int index = 0; index < expected.length; index++) {
      double difference = actual[index] - expected[index];
      squaredError += difference * difference;
      squaredReference += expected[index] * expected[index];
    }
    return Math.sqrt(squaredError / squaredReference);
  }

  private static long median(long[] samples) {
    long[] sorted = samples.clone();
    Arrays.sort(sorted);
    return sorted[sorted.length / 2];
  }

  static byte[] randomQ4Matrix(int rows, int cols, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    int blocks = Math.multiplyExact(rows, cols / BLOCK_VALUES);
    byte[] weights = new byte[Math.multiplyExact(blocks, BLOCK_BYTES)];
    for (int block = 0; block < blocks; block++) {
      short scale = Float.floatToFloat16(0.001f + random.nextFloat() * 0.05f);
      int offset = block * BLOCK_BYTES;
      weights[offset] = (byte) scale;
      weights[offset + 1] = (byte) (scale >>> 8);
      for (int quant = 0; quant < 16; quant++) {
        weights[offset + 2 + quant] = (byte) random.nextInt(256);
      }
    }
    return weights;
  }

  static float[] randomFloats(int length, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = random.nextFloat(-2.0f, 2.0f);
    }
    return values;
  }

  private record Configuration(int batchSize, int rows, int cols, int warmups, int iterations) {
    private static Configuration parse(String[] args) {
      if (args.length != 0 && args.length != 5) {
        throw new IllegalArgumentException(
            "usage: Q4ProjectionExperiment [batch rows cols warmups iterations]");
      }
      Configuration configuration =
          args.length == 0
              ? new Configuration(32, 3072, 1024, 3, 10)
              : new Configuration(
                  Integer.parseInt(args[0]),
                  Integer.parseInt(args[1]),
                  Integer.parseInt(args[2]),
                  Integer.parseInt(args[3]),
                  Integer.parseInt(args[4]));
      if (configuration.batchSize < 1
          || configuration.rows < 1
          || configuration.cols < BLOCK_VALUES
          || configuration.cols % BLOCK_VALUES != 0
          || configuration.warmups < 0
          || configuration.iterations < 1) {
        throw new IllegalArgumentException("invalid projection experiment configuration");
      }
      return configuration;
    }
  }
}
