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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Steady-state single-token decode rate for any GGUF, through the route the RAG bench uses.
 *
 * <p>Exists to answer one question that an EdgeRock-only measurement cannot: when our decode rate
 * sits below a llama.cpp arm, is that this model or is it this decoder? Comparing EdgeRock against
 * llama.cpp gives a ratio; comparing a second, unrelated, already-qualified model against the same
 * llama.cpp build on the same box says whether that ratio is a property of EdgeRock's memory layers
 * or the baseline for {@code pure-java}. Those two readings lead to very different decisions, so
 * the control is worth its own probe.
 *
 * <p>Deliberately on {@code PureJavaBackend.load(Path)}: that is what {@code
 * InProcessGenerationClient} and {@code FrameworkExerciseCli} call for the {@code pure-java} arm,
 * so it is the route qualification measures. It passes {@code GgufBatchedMatrixKernel.none()},
 * which is a property of the measured route and not an oversight here.
 *
 * <p>Prints rather than asserts a threshold. A throughput number is hardware-specific, and a test
 * that failed on a slower box would be measuring the box.
 */
@Tag("integration")
@EnabledIfSystemProperty(named = PureJavaDecodeRateProbe.MODEL_PROPERTY, matches = ".+")
class PureJavaDecodeRateProbe {

  static final String MODEL_PROPERTY = "models.fixtures.decodeRateModel";

  @Test
  void reportsSteadyStateDecodeRate() {
    Path model = Path.of(System.getProperty(MODEL_PROPERTY));
    int warmup = Integer.getInteger("models.fixtures.decodeRateWarmup", 12);
    int measured = Integer.getInteger("models.fixtures.decodeRateTokens", 120);
    // Required, not optional. Left unset, a model is loaded at its declared context_length, and
    // EdgeRock declares 262144: its eight attention layers would ask for roughly 17 GB of key and
    // value cache and the run dies on heap space rather than reporting a rate. Every arm of a
    // comparison must also be at the same context for its numbers to mean anything together.
    String context = System.getProperty("models.fixtures.decodeRateContext", "16384");
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, context);

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      int[] prompt = backend.tokenizer().encode("The capital of France is");
      float[] logits = null;
      int position = 0;
      for (int token : prompt) {
        logits = backend.forward(token, position++);
      }

      // Warm up away from JIT and first-touch page faults before timing anything.
      for (int step = 0; step < warmup; step++) {
        logits = backend.forward(argmax(logits), position++);
      }

      long start = System.nanoTime();
      for (int step = 0; step < measured; step++) {
        logits = backend.forward(argmax(logits), position++);
      }
      double seconds = (System.nanoTime() - start) / 1e9;

      assertThat(logits).isNotNull();
      System.out.printf(
          "[decode-rate] %s: %d tokens in %.1fs = %.2f tok/s (vocab %d, warmup %d)%n",
          model.getFileName(), measured, seconds, measured / seconds, logits.length, warmup);
      System.out.printf("[decode-rate] context %s%n", context);
    } finally {
      if (previous == null) {
        System.clearProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      } else {
        System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }
}
