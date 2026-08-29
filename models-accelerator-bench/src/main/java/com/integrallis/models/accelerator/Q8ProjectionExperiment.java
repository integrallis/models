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

/** Measures one isolated Java-authored Q8_0 projection on a TornadoVM device. */
public final class Q8ProjectionExperiment {
  private static final int BLOCK_VALUES = 32;
  private static final int BLOCK_BYTES = 34;

  private Q8ProjectionExperiment() {}

  public static void main(String[] args) throws TornadoExecutionPlanException {
    Configuration configuration = Configuration.parse(args);
    byte[] weights = randomQ8Matrix(configuration.rows(), configuration.cols(), 23L);
    float[] input =
        randomFloats(Math.multiplyExact(configuration.batchSize(), configuration.cols()), 29L);
    float[] expected =
        vectorApiProjection(
            weights, input, configuration.batchSize(), configuration.rows(), configuration.cols());

    ByteArray deviceWeights = ByteArray.fromArray(weights);
    FloatArray deviceInput = FloatArray.fromArray(input);
    FloatArray deviceOutput =
        new FloatArray(Math.multiplyExact(configuration.batchSize(), configuration.rows()));
    Q8ProjectionKernel.validate(
        deviceWeights,
        deviceInput,
        deviceOutput,
        configuration.batchSize(),
        configuration.rows(),
        configuration.cols());

    TaskGraph graph =
        new TaskGraph("q8-projection")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, deviceWeights)
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, deviceInput)
            .task(
                "multiply",
                Q8ProjectionKernel::multiply,
                deviceWeights,
                deviceInput,
                deviceOutput,
                configuration.batchSize(),
                configuration.rows(),
                configuration.cols())
            .transferToHost(DataTransferMode.EVERY_EXECUTION, deviceOutput);

    long coldNanos;
    long[] warmNanos = new long[configuration.iterations()];
    TornadoExecutionResult lastResult;
    try (TornadoExecutionPlan plan =
        new TornadoExecutionPlan(graph.snapshot()).withProfiler(ProfilerMode.SILENT)) {
      long started = System.nanoTime();
      lastResult = plan.execute();
      coldNanos = System.nanoTime() - started;
      for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
        plan.execute();
      }
      for (int iteration = 0; iteration < warmNanos.length; iteration++) {
        started = System.nanoTime();
        lastResult = plan.execute();
        warmNanos[iteration] = System.nanoTime() - started;
      }
    }

    float[] actual = deviceOutput.toHeapArray();
    double relativeL2 = relativeL2(expected, actual);
    if (relativeL2 > 2.0e-5) {
      throw new IllegalStateException("accelerator result failed relative-L2 gate: " + relativeL2);
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
    long cpuMedian = median(cpuNanos);
    double weightGiB = weights.length / (1024.0 * 1024.0 * 1024.0);
    double effectiveGiBPerSecond =
        weightGiB * configuration.batchSize() / (deviceMedian / 1_000_000_000.0);

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
    System.out.printf(Locale.ROOT, "warm device p50   %.3f ms%n", deviceMedian / 1_000_000.0);
    System.out.printf(Locale.ROOT, "Vector API p50    %.3f ms%n", cpuMedian / 1_000_000.0);
    System.out.printf(Locale.ROOT, "speedup           %.2fx%n", (double) cpuMedian / deviceMedian);
    System.out.printf(Locale.ROOT, "effective traffic %.2f GiB/s%n", effectiveGiBPerSecond);
    System.out.printf(
        Locale.ROOT,
        "last kernel       %.3f ms%n",
        lastResult.getProfilerResult().getDeviceKernelTime() / 1_000_000.0);
    System.out.printf(
        Locale.ROOT,
        "last transfers    %.3f ms%n",
        lastResult.getProfilerResult().getDataTransfersTime() / 1_000_000.0);
  }

  private static long[] timeCpu(
      byte[] weights,
      float[] input,
      int batchSize,
      int rows,
      int cols,
      int warmups,
      int iterations) {
    for (int warmup = 0; warmup < warmups; warmup++) {
      vectorApiProjection(weights, input, batchSize, rows, cols);
    }
    long[] nanos = new long[iterations];
    for (int iteration = 0; iteration < iterations; iteration++) {
      long started = System.nanoTime();
      vectorApiProjection(weights, input, batchSize, rows, cols);
      nanos[iteration] = System.nanoTime() - started;
    }
    return nanos;
  }

  private static float[] vectorApiProjection(
      byte[] weights, float[] input, int batchSize, int rows, int cols) {
    float[] output = new float[Math.multiplyExact(batchSize, rows)];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);
    for (int batch = 0; batch < batchSize; batch++) {
      float[] query = Arrays.copyOfRange(input, batch * cols, (batch + 1) * cols);
      float[] projected = new float[rows];
      VectorUtil.ggufQ8_0BatchDotProduct(query, weightSegment, rows, cols, projected);
      System.arraycopy(projected, 0, output, batch * rows, rows);
    }
    return output;
  }

  private static double relativeL2(float[] expected, float[] actual) {
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

  private static byte[] randomQ8Matrix(int rows, int cols, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    int blocks = Math.multiplyExact(rows, cols / BLOCK_VALUES);
    byte[] weights = new byte[Math.multiplyExact(blocks, BLOCK_BYTES)];
    for (int block = 0; block < blocks; block++) {
      short scale = Float.floatToFloat16(0.001f + random.nextFloat() * 0.05f);
      int offset = block * BLOCK_BYTES;
      weights[offset] = (byte) scale;
      weights[offset + 1] = (byte) (scale >>> 8);
      for (int quant = 0; quant < BLOCK_VALUES; quant++) {
        weights[offset + 2 + quant] = (byte) (random.nextInt(255) - 127);
      }
    }
    return weights;
  }

  private static float[] randomFloats(int length, long seed) {
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
            "usage: Q8ProjectionExperiment [batch rows cols warmups iterations]");
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
