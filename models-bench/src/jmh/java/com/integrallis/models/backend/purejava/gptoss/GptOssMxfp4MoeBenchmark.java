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
package com.integrallis.models.backend.purejava.gptoss;

import com.integrallis.vectors.core.Mxfp4Matrix;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/** Measures one official-shape GPT-OSS 20B token across its four selected experts. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 5, time = 1)
public class GptOssMxfp4MoeBenchmark {

  private static final int HIDDEN = 2880;
  private static final int INTERMEDIATE = 2880;
  private static final int TOP_K = 4;

  private GptOssMxfp4Moe moe;
  private float[] hidden;
  private int[] selectedExperts;
  private float[] routingWeights;
  private float[] output;

  @Setup
  public void setUp() {
    GptOssMxfp4ExpertWeights.Expert[] experts = new GptOssMxfp4ExpertWeights.Expert[TOP_K];
    for (int expert = 0; expert < experts.length; expert++) {
      experts[expert] =
          new GptOssMxfp4ExpertWeights.Expert(
              matrix(2 * INTERMEDIATE, HIDDEN, expert),
              new float[2 * INTERMEDIATE],
              matrix(HIDDEN, INTERMEDIATE, expert + TOP_K),
              new float[HIDDEN]);
    }
    moe = new GptOssMxfp4Moe(GptOssMxfp4ExpertWeights.of(experts), 1.702f, 7.0f);
    hidden = new float[HIDDEN];
    for (int index = 0; index < hidden.length; index++) {
      hidden[index] = (index % 29 - 14) * 0.003f;
    }
    selectedExperts = new int[] {0, 1, 2, 3};
    routingWeights = new float[] {0.4f, 0.3f, 0.2f, 0.1f};
    output = new float[HIDDEN];
  }

  @Benchmark
  public float exactF32Activation() {
    moe.forwardExact(hidden, selectedExperts, routingWeights, output);
    return output[HIDDEN - 1];
  }

  @Benchmark
  public float w4a8SharedActivation() {
    moe.forwardQ8(hidden, selectedExperts, routingWeights, output);
    return output[HIDDEN - 1];
  }

  private static Mxfp4Matrix matrix(int rows, int columns, int seed) {
    byte[] blocks = new byte[Math.multiplyExact(rows, columns / 2)];
    byte[] scales = new byte[Math.multiplyExact(rows, columns / 32)];
    Arrays.fill(blocks, (byte) (((seed + 2) & 0x07) << 4 | ((seed + 1) & 0x07)));
    Arrays.fill(scales, (byte) (119 + seed % 3));
    return Mxfp4Matrix.of(
        MemorySegment.ofArray(blocks), MemorySegment.ofArray(scales), rows, columns);
  }
}
