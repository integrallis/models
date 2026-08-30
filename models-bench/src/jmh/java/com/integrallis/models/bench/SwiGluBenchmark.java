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
package com.integrallis.models.bench;

import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Compares the production Vector API SwiGLU with the scalar transformer reference. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
public class SwiGluBenchmark {

  @State(Scope.Thread)
  public static class Inputs {
    @Param({"3072", "11008"})
    int size;

    float[] output;
    float[] gate;
    float[] up;

    @Setup
    public void setUp() {
      output = new float[size];
      gate = new float[size];
      up = new float[size];
      Random random = new Random(42L);
      for (int index = 0; index < size; index++) {
        gate[index] = (float) (random.nextGaussian() * 2.5);
        up[index] = (float) random.nextGaussian();
      }
    }
  }

  @Benchmark
  public void vectorApi(Inputs inputs, Blackhole blackhole) {
    TensorOps.swiGlu(inputs.output, inputs.gate, inputs.up, inputs.size);
    blackhole.consume(inputs.output);
  }

  @Benchmark
  public void scalarReference(Inputs inputs, Blackhole blackhole) {
    for (int index = 0; index < inputs.size; index++) {
      float value = inputs.gate[index];
      inputs.output[index] = value / (1.0f + (float) Math.exp(-value)) * inputs.up[index];
    }
    blackhole.consume(inputs.output);
  }
}
