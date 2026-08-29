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

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

/** Device correctness and latency gate for Qwen3-shaped causal grouped-query attention. */
public final class CausalAttentionExperiment {
  private static final int EXECUTION_BATCH = 32;
  private static final int HEADS = 16;
  private static final int KV_HEADS = 8;
  private static final int HEAD_LENGTH = 64;
  private static final int QUERY_DIM = HEADS * HEAD_LENGTH;
  private static final int KV_DIM = KV_HEADS * HEAD_LENGTH;
  private static final int MAX_SEQUENCE = 512;
  private static final int CHUNKS = 8;
  private static final int MEASUREMENTS = 7;
  private static final double MAX_RELATIVE_L2 = 2.0e-5;

  private CausalAttentionExperiment() {}

  public static void main(String[] args) {
    if (args.length != 0) {
      throw new IllegalArgumentException("usage: CausalAttentionExperiment");
    }
    Chunk[] chunks = chunks();
    float[][] expected = referenceOutputs(chunks);
    float[][] actual = new float[CHUNKS][EXECUTION_BATCH * QUERY_DIM];

    try (TornadoCausalAttentionPlan plan =
        new TornadoCausalAttentionPlan(
            "qwen3-causal-attention",
            EXECUTION_BATCH,
            HEADS,
            KV_HEADS,
            HEAD_LENGTH,
            HEAD_LENGTH,
            MAX_SEQUENCE,
            0)) {
      long coldStarted = System.nanoTime();
      executeSequence(plan, chunks, actual);
      double coldMillis = elapsedMillis(coldStarted);
      double maximumError = 0.0;
      for (int chunk = 0; chunk < CHUNKS; chunk++) {
        maximumError =
            Math.max(
                maximumError, Q4ProjectionExperiment.relativeL2(expected[chunk], actual[chunk]));
      }
      if (maximumError > MAX_RELATIVE_L2) {
        throw new IllegalStateException(
            "attention result failed relative-L2 gate: " + maximumError);
      }

      double[] gpuMillis = new double[MEASUREMENTS];
      for (int measurement = 0; measurement < MEASUREMENTS; measurement++) {
        long started = System.nanoTime();
        executeSequence(plan, chunks, actual);
        gpuMillis[measurement] = elapsedMillis(started);
      }
      double[] cpuMillis = new double[MEASUREMENTS];
      for (int measurement = 0; measurement < MEASUREMENTS; measurement++) {
        long started = System.nanoTime();
        referenceOutputs(chunks);
        cpuMillis[measurement] = elapsedMillis(started);
      }
      System.out.printf(
          Locale.ROOT,
          "Qwen3 attention chunks=%d tokens=%d relativeL2=%.8g cold=%.3f ms GPU-p50=%.3f ms CPU-p50=%.3f ms speedup=%.2fx calls=%d%n",
          CHUNKS,
          CHUNKS * EXECUTION_BATCH,
          maximumError,
          coldMillis,
          median(gpuMillis),
          median(cpuMillis),
          median(cpuMillis) / median(gpuMillis),
          plan.calls());
    }
  }

  private static void executeSequence(
      TornadoCausalAttentionPlan plan, Chunk[] chunks, float[][] outputs) {
    for (int chunk = 0; chunk < chunks.length; chunk++) {
      Chunk inputs = chunks[chunk];
      plan.execute(
          inputs.query(),
          inputs.key(),
          inputs.value(),
          outputs[chunk],
          EXECUTION_BATCH,
          chunk * EXECUTION_BATCH);
    }
  }

  private static Chunk[] chunks() {
    Chunk[] chunks = new Chunk[CHUNKS];
    for (int chunk = 0; chunk < chunks.length; chunk++) {
      long seed = 101L + chunk * 10L;
      chunks[chunk] =
          new Chunk(
              randomFloats(EXECUTION_BATCH * QUERY_DIM, seed),
              randomFloats(EXECUTION_BATCH * KV_DIM, seed + 1),
              randomFloats(EXECUTION_BATCH * KV_DIM, seed + 2));
    }
    return chunks;
  }

  private static float[] randomFloats(int length, long seed) {
    Random random = new Random(seed);
    float[] values = new float[length];
    for (int index = 0; index < values.length; index++) {
      values[index] = random.nextFloat(-1.0f, 1.0f);
    }
    return values;
  }

  private static float[][] referenceOutputs(Chunk[] chunks) {
    float[] keyCache = new float[MAX_SEQUENCE * KV_DIM];
    float[] valueCache = new float[MAX_SEQUENCE * KV_DIM];
    float[][] outputs = new float[chunks.length][EXECUTION_BATCH * QUERY_DIM];
    for (int chunk = 0; chunk < chunks.length; chunk++) {
      int startPosition = chunk * EXECUTION_BATCH;
      Chunk inputs = chunks[chunk];
      for (int batch = 0; batch < EXECUTION_BATCH; batch++) {
        System.arraycopy(
            inputs.key(), batch * KV_DIM, keyCache, (startPosition + batch) * KV_DIM, KV_DIM);
        System.arraycopy(
            inputs.value(), batch * KV_DIM, valueCache, (startPosition + batch) * KV_DIM, KV_DIM);
      }
      referenceChunk(inputs.query(), keyCache, valueCache, outputs[chunk], startPosition);
    }
    return outputs;
  }

  private static void referenceChunk(
      float[] query, float[] keyCache, float[] valueCache, float[] output, int startPosition) {
    float scale = (float) (1.0 / Math.sqrt(HEAD_LENGTH));
    int groupSize = HEADS / KV_HEADS;
    float[] scores = new float[MAX_SEQUENCE];
    for (int batch = 0; batch < EXECUTION_BATCH; batch++) {
      int position = startPosition + batch;
      for (int head = 0; head < HEADS; head++) {
        int kvHead = head / groupSize;
        float maximum = Float.NEGATIVE_INFINITY;
        for (int cached = 0; cached <= position; cached++) {
          float dot = 0.0f;
          for (int column = 0; column < HEAD_LENGTH; column++) {
            dot +=
                query[batch * QUERY_DIM + head * HEAD_LENGTH + column]
                    * keyCache[cached * KV_DIM + kvHead * HEAD_LENGTH + column];
          }
          scores[cached] = dot * scale;
          maximum = Math.max(maximum, scores[cached]);
        }
        float denominator = 0.0f;
        for (int cached = 0; cached <= position; cached++) {
          scores[cached] = (float) Math.exp(scores[cached] - maximum);
          denominator += scores[cached];
        }
        int outputOffset = batch * QUERY_DIM + head * HEAD_LENGTH;
        for (int column = 0; column < HEAD_LENGTH; column++) {
          float weighted = 0.0f;
          for (int cached = 0; cached <= position; cached++) {
            weighted +=
                scores[cached] * valueCache[cached * KV_DIM + kvHead * HEAD_LENGTH + column];
          }
          output[outputOffset + column] = weighted / denominator;
        }
      }
    }
  }

  private static double median(double[] values) {
    double[] sorted = values.clone();
    Arrays.sort(sorted);
    return sorted[sorted.length / 2];
  }

  private static double elapsedMillis(long started) {
    return (System.nanoTime() - started) / 1_000_000.0;
  }

  private record Chunk(float[] query, float[] key, float[] value) {}
}
