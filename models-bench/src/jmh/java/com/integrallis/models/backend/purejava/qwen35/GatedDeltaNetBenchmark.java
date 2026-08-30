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
package com.integrallis.models.backend.purejava.qwen35;

import com.integrallis.models.backend.nativekernel.NativeKernelLibrary;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Compares the Java and optional native Gated DeltaNet kernels at a Qwen 3.5 0.8B shape. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class GatedDeltaNetBenchmark {

  private static final int HEADS = 16;
  private static final int DIMENSION = 128;

  @State(Scope.Thread)
  public static class Inputs {
    @Param({"1", "32"})
    int tokenCount;

    float[] query;
    float[] key;
    float[] value;
    float[] logDecay;
    float[] beta;
    float[] javaState;
    float[] nativeState;
    float[] javaOutput;
    float[] nativeOutput;
    float[] normalizedQuery;
    float[] normalizedKey;
    float[] memory;
    float[] delta;
    NativeKernelLibrary nativeLibrary;

    @Setup(Level.Trial)
    public void setUp() {
      int tokenElements = Math.multiplyExact(Math.multiplyExact(tokenCount, HEADS), DIMENSION);
      int gateElements = Math.multiplyExact(tokenCount, HEADS);
      int stateElements = Math.multiplyExact(Math.multiplyExact(HEADS, DIMENSION), DIMENSION);
      query = new float[tokenElements];
      key = new float[tokenElements];
      value = new float[tokenElements];
      logDecay = new float[gateElements];
      beta = new float[gateElements];
      javaState = new float[stateElements];
      nativeState = new float[stateElements];
      javaOutput = new float[tokenElements];
      nativeOutput = new float[tokenElements];
      normalizedQuery = new float[DIMENSION];
      normalizedKey = new float[DIMENSION];
      memory = new float[DIMENSION];
      delta = new float[DIMENSION];
      Random random = new Random(42L);
      fill(random, query, 0.25f);
      fill(random, key, 0.25f);
      fill(random, value, 0.25f);
      fill(random, javaState, 0.01f);
      System.arraycopy(javaState, 0, nativeState, 0, stateElements);
      for (int index = 0; index < gateElements; index++) {
        logDecay[index] = -0.05f - random.nextFloat() * 0.2f;
        beta[index] = 0.2f + random.nextFloat() * 0.7f;
      }
      String libraryPath = System.getProperty("models.native.kernels.library");
      if (libraryPath == null || libraryPath.isBlank()) {
        throw new IllegalStateException(
            "set -Dmodels.native.kernels.library to benchmark the native GDN kernel");
      }
      nativeLibrary = NativeKernelLibrary.open(Path.of(libraryPath));
    }

    @TearDown(Level.Trial)
    public void close() {
      nativeLibrary.close();
    }

    private static void fill(Random random, float[] values, float scale) {
      for (int index = 0; index < values.length; index++) {
        values[index] = (random.nextFloat() - 0.5f) * scale;
      }
    }
  }

  @Benchmark
  public void javaVectorApi(Inputs inputs, Blackhole blackhole) {
    GatedDeltaNetRecurrence.forwardPrefixInPlace(
        inputs.query,
        inputs.key,
        inputs.value,
        inputs.logDecay,
        inputs.beta,
        inputs.javaState,
        inputs.javaOutput,
        inputs.normalizedQuery,
        inputs.normalizedKey,
        inputs.memory,
        inputs.delta,
        inputs.tokenCount,
        HEADS,
        HEADS,
        DIMENSION,
        DIMENSION);
    blackhole.consume(inputs.javaOutput);
  }

  @Benchmark
  public void nativeFfm(Inputs inputs, Blackhole blackhole) {
    inputs.nativeLibrary.gatedDeltaNetF32(
        inputs.query,
        inputs.key,
        inputs.value,
        inputs.logDecay,
        inputs.beta,
        inputs.nativeState,
        inputs.nativeOutput,
        inputs.tokenCount,
        HEADS,
        HEADS,
        DIMENSION,
        DIMENSION);
    blackhole.consume(inputs.nativeOutput);
  }
}
