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

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel.AttentionScope;
import com.integrallis.models.backend.tornado.TornadoAttentionRouting;
import com.integrallis.models.backend.tornado.TornadoAttentionShape;
import com.integrallis.models.backend.tornado.TornadoCausalAttentionKernel;
import java.util.Arrays;
import java.util.Locale;
import java.util.SplittableRandom;

/**
 * Device gate for the productised single-token attention kernel.
 *
 * <p>Off-device CI already proves the kernel's arithmetic, its grouped-query mapping, its causal
 * bound, and the routing and mirror bookkeeping around it. What it cannot prove is that TornadoVM
 * compiles this kernel to PTX at all, that the {@code FIRST_EXECUTION} mirror really stays resident
 * across executions, or that a decode step is faster on the device than on the Vector API. That is
 * what this runs.
 *
 * <p>It is a gate, not a benchmark of the whole runtime: it drives the attention kernel directly at
 * a fixed geometry, the way {@code LlamaForwardPass} drives it, and reports parity against the
 * production Java attention path together with per-step latency at several context lengths. A
 * full-model decode gate is the separate, later step, and it is the one that decides whether this
 * path ships — the standing negative result for device attention on this codebase (A16, 2026-08-29:
 * a tiled prefill kernel at relative L2 3.47e-7 and an isolated 1.31x that regressed warm prefill
 * from 4.781 s to 10.266 s) was an isolated win that lost in the model.
 */
public final class TornadoAttentionDecodeExperiment {
  private static final int HEADS = 32;
  private static final int KV_HEADS = 8;
  private static final int HEAD_LENGTH = 128;
  private static final int MAX_SEQUENCE = 4096;
  private static final int LAYERS = 4;
  private static final float SCALE = (float) (1.0 / Math.sqrt(HEAD_LENGTH));
  private static final int[] CONTEXTS = {128, 512, 1024, 2048, 4095};
  private static final int MEASUREMENTS = 9;
  private static final double MAX_RELATIVE_L2 = 2.0e-5;

  private TornadoAttentionDecodeExperiment() {}

  public static void main(String[] args) {
    if (args.length != 0) {
      throw new IllegalArgumentException("usage: TornadoAttentionDecodeExperiment");
    }
    int keyDim = KV_HEADS * HEAD_LENGTH;
    int queryDim = HEADS * HEAD_LENGTH;
    float[] keys = randomFloats(MAX_SEQUENCE * keyDim, 3L);
    float[] values = randomFloats(MAX_SEQUENCE * keyDim, 5L);
    float[] queries = randomFloats(MAX_SEQUENCE * queryDim, 7L);

    TornadoAttentionShape shape =
        new TornadoAttentionShape(
            1, 64, HEADS, KV_HEADS, HEAD_LENGTH, HEAD_LENGTH, MAX_SEQUENCE, 0, SCALE);
    System.out.printf(
        Locale.ROOT,
        "geometry heads=%d kvHeads=%d headLength=%d context=%d layers=%d%n"
            + "  device bytes for the mirror and scratch: %.1f MiB%n"
            + "  KV uploaded per decode step, mirrored: %.1f KiB;"
            + " host-resident at full context: %.1f MiB%n",
        HEADS,
        KV_HEADS,
        HEAD_LENGTH,
        MAX_SEQUENCE,
        LAYERS,
        shape.deviceBytes(LAYERS) / (1024.0 * 1024.0),
        shape.mirrorBytesPerPosition() * LAYERS / 1024.0,
        shape.hostResidentUploadBytesPerDecodeStep(LAYERS, MAX_SEQUENCE - 1) / (1024.0 * 1024.0));

    try (TornadoCausalAttentionKernel kernel = new TornadoCausalAttentionKernel(Long.MAX_VALUE)) {
      AttentionScope scope = new AttentionScope(1L, 0, SCALE, false);
      float[] output = new float[queryDim];
      double worstRelativeL2 = 0.0;

      for (int context : CONTEXTS) {
        kernel.reset();
        kernel.selectScope(scope);
        for (int layer = 0; layer < LAYERS; layer++) {
          if (!eligible(kernel, layer, context)) {
            throw new IllegalStateException(
                "layer " + layer + " was refused: " + kernel.lastRefusal());
          }
          kernel.mirrorSpan(layer, 0, context, keys, 0, keyDim, values, 0, keyDim);
        }

        double[] deviceMillis = new double[MEASUREMENTS];
        for (int measurement = 0; measurement < MEASUREMENTS; measurement++) {
          long started = System.nanoTime();
          for (int layer = 0; layer < LAYERS; layer++) {
            kernel.selectScope(scope);
            if (!eligible(kernel, layer, context)) {
              throw new IllegalStateException(
                  "layer " + layer + " was refused mid-run: " + kernel.lastRefusal());
            }
            kernel.attend(
                output,
                Arrays.copyOfRange(queries, context * queryDim, (context + 1) * queryDim),
                Arrays.copyOfRange(keys, context * keyDim, (context + 1) * keyDim),
                Arrays.copyOfRange(values, context * keyDim, (context + 1) * keyDim),
                layer,
                context,
                1,
                HEADS,
                KV_HEADS,
                HEAD_LENGTH,
                HEAD_LENGTH,
                MAX_SEQUENCE,
                0);
            kernel.rewind(context);
          }
          deviceMillis[measurement] = (System.nanoTime() - started) / 1_000_000.0;
        }

        float[] reference = javaAttention(keys, values, queries, context);
        double relativeL2 = relativeL2(output, reference);
        worstRelativeL2 = Math.max(worstRelativeL2, relativeL2);

        double[] javaMillis = new double[MEASUREMENTS];
        for (int measurement = 0; measurement < MEASUREMENTS; measurement++) {
          long started = System.nanoTime();
          for (int layer = 0; layer < LAYERS; layer++) {
            javaAttention(keys, values, queries, context);
          }
          javaMillis[measurement] = (System.nanoTime() - started) / 1_000_000.0;
        }

        System.out.printf(
            Locale.ROOT,
            "context=%5d relativeL2=%.4g device-p50=%8.3f ms vector-api-p50=%8.3f ms"
                + " speedup=%.2fx%n",
            context,
            relativeL2,
            median(deviceMillis),
            median(javaMillis),
            median(javaMillis) / median(deviceMillis));
      }

      TornadoAttentionRouting routing = kernel.routing();
      System.out.println(routing.summary());
      if (!routing.ranOnDevice()) {
        throw new IllegalStateException(
            "no attention step reached the device: " + kernel.lastRefusal());
      }
      if (worstRelativeL2 > MAX_RELATIVE_L2) {
        throw new IllegalStateException(
            "attention failed the stated numeric contract: " + worstRelativeL2);
      }
      System.out.printf(
          Locale.ROOT,
          "PASS worst relativeL2=%.4g against the production Java attention path (contract %.1g)%n",
          worstRelativeL2,
          MAX_RELATIVE_L2);
    }
  }

  private static boolean eligible(TornadoCausalAttentionKernel kernel, int layer, int position) {
    return kernel.isEligible(
        layer, position, 1, HEADS, KV_HEADS, HEAD_LENGTH, HEAD_LENGTH, MAX_SEQUENCE, 0);
  }

  /** The production Java attention path, head by head, as {@code LlamaForwardPass} runs it. */
  private static float[] javaAttention(
      float[] keys, float[] values, float[] queries, int position) {
    int keyDim = KV_HEADS * HEAD_LENGTH;
    int queryDim = HEADS * HEAD_LENGTH;
    KvCache cache = new KvCache(1, MAX_SEQUENCE, keyDim, keyDim);
    for (int stored = 0; stored <= position; stored++) {
      cache.store(0, stored, keys, stored * keyDim, values, stored * keyDim);
    }
    float[] output = new float[queryDim];
    float[] scores = new float[MAX_SEQUENCE];
    KvCache.AttentionView view = cache.attentionView(0, 0, position + 1);
    int groupSize = HEADS / KV_HEADS;
    for (int head = 0; head < HEADS; head++) {
      int kvHead = head / groupSize;
      cache.writeAttentionScores(
          view,
          0,
          position + 1,
          kvHead * HEAD_LENGTH,
          queries,
          position * queryDim + head * HEAD_LENGTH,
          HEAD_LENGTH,
          SCALE,
          scores,
          0,
          false);
      TensorOps.softmax(scores, 0, position + 1);
      cache.addAttentionValues(
          view,
          0,
          position + 1,
          kvHead * HEAD_LENGTH,
          output,
          head * HEAD_LENGTH,
          HEAD_LENGTH,
          scores,
          0,
          false);
    }
    return output;
  }

  private static double relativeL2(float[] actual, float[] expected) {
    double difference = 0.0;
    double magnitude = 0.0;
    for (int index = 0; index < expected.length; index++) {
      double delta = (double) actual[index] - expected[index];
      difference += delta * delta;
      magnitude += (double) expected[index] * expected[index];
    }
    return magnitude == 0.0 ? Math.sqrt(difference) : Math.sqrt(difference / magnitude);
  }

  private static float[] randomFloats(int length, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[] values = new float[length];
    for (int index = 0; index < values.length; index++) {
      values[index] = (float) random.nextDouble(-1.0, 1.0);
    }
    return values;
  }

  private static double median(double[] samples) {
    double[] sorted = samples.clone();
    Arrays.sort(sorted);
    return sorted[sorted.length / 2];
  }
}
