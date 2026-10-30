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
 * Real-weight smoke for EdgeRock-4B served through the Qwen3.5 hybrid decoder.
 *
 * <p>This asserts that the model loads, that its memory anchors are wired, and that decoding
 * produces finite logits and in-vocabulary tokens. It deliberately does <b>not</b> assert that the
 * output is correct: nothing here is pinned against the reference, so a coherent-looking
 * continuation would be evidence of nothing. Correctness is settled by pinning greedy token IDs
 * against the patched llama.cpp the publisher ships with the GGUF, which is the same discipline
 * {@code HammerYarnModelIntegrationTest} applies, and that pin does not exist yet.
 */
@Tag("integration")
@EnabledIfSystemProperty(named = EdgeRockModelIntegrationTest.FIXTURE_PROPERTY, matches = ".+")
class EdgeRockModelIntegrationTest {

  static final String FIXTURE_PROPERTY = "models.fixtures.edgeRock4bGguf";

  @Test
  void loadsAndDecodesWithFiniteLogits() {
    Path model = Path.of(System.getProperty(FIXTURE_PROPERTY));
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "512");

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      int[] promptTokens = backend.tokenizer().encode("The capital of France is");
      assertThat(promptTokens).isNotEmpty();

      int vocabulary = -1;
      StringBuilder decoded = new StringBuilder();
      int[] generated = new int[16];
      int position = 0;
      float[] logits = null;
      for (int token : promptTokens) {
        logits = backend.forward(token, position++);
      }
      for (int step = 0; step < generated.length; step++) {
        assertThat(logits).isNotNull();
        vocabulary = logits.length;
        int best = argmax(logits);
        assertThat(best).as("greedy token must be in vocabulary").isBetween(0, vocabulary - 1);
        for (float logit : logits) {
          assertThat(Float.isFinite(logit))
              .as("every logit must be finite at step %d", step)
              .isTrue();
        }
        generated[step] = best;
        decoded.append(backend.tokenizer().decode(new int[] {best}));
        logits = backend.forward(best, position++);
      }

      assertThat(vocabulary).isEqualTo(248_320);
      // Printed rather than asserted: this is an observation to pin against the reference, not a
      // result. Asserting whatever our own decoder happens to emit would be a corpus we wrote
      // scoring our own assumptions back to us.
      System.out.println("[edgerock] greedy tokens: " + java.util.Arrays.toString(generated));
      System.out.println("[edgerock] greedy text: " + decoded);
    } finally {
      if (previous == null) {
        System.clearProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      } else {
        System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }
  }

  @Test
  void theMemoryAnchorsActuallyContribute() throws java.io.IOException {
    // Coherent text proves the backbone runs; it does not prove the memory ran. A memory module
    // that returned zeros would leave a plain Qwen3.5 hybrid, which also produces fluent output,
    // so "it reads well" cannot distinguish the two. This drives one real memory layer with real
    // weights and requires its contribution to be finite and non-zero, and requires the bank to
    // have taken the token.
    Path model = Path.of(System.getProperty(FIXTURE_PROPERTY));
    try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofShared()) {
      com.integrallis.models.backend.purejava.gguf.GgufFile file =
          com.integrallis.models.backend.purejava.gguf.GgufParser.parse(model, arena);
      com.integrallis.models.backend.purejava.edgerock.EdgeRockMemory memory =
          com.integrallis.models.backend.purejava.edgerock.EdgeRockMemory.load(file);
      com.integrallis.models.backend.purejava.edgerock.MemoryBank[] banks = memory.newBanks();

      int embedding = memory.config().embeddingLength();
      int layer = memory.config().fullAttentionLayers()[0];
      float[] hidden = new float[embedding];
      java.util.Random random = new java.util.Random(20_261_030L);
      for (int i = 0; i < embedding; i++) {
        hidden[i] = (float) random.nextGaussian() * 0.1f;
      }
      float[] out = new float[embedding];
      int rotatedPairs = memory.config().ropeDimensionCount() / 2;
      float[] cos = new float[rotatedPairs];
      float[] sin = new float[rotatedPairs];
      java.util.Arrays.fill(cos, 1.0f);
      var rotation =
          com.integrallis.models.backend.purejava.edgerock.EdgeRockMemoryLayer.rotation(cos, sin);

      // Nothing is readable until a chunk commits. The read happens before the write, the read
      // sees only the committed selection, and commit() fires when the pending chunk reaches
      // commit_chunk -- 256 on this model. So a contribution of exactly zero is the correct
      // answer for every token before the first commit, and a test that drove one token would
      // have called the layer dead when it was merely early.
      int chunk = memory.config().memoryCommitChunk();
      assertThat(chunk).isEqualTo(256);

      double beforeFirstCommit = 0;
      for (int position = 0; position < chunk; position++) {
        randomise(hidden, random);
        memory.layer(layer).forward(hidden, position, rotation, banks[layer], out);
        beforeFirstCommit += l1(out);
      }
      assertThat(beforeFirstCommit)
          .as("no anchor is readable before the first chunk commits")
          .isZero();
      assertThat(banks[layer].total()).isEqualTo(chunk);

      // The chunk committed on the last staged token, so the next read has anchors to attend to.
      randomise(hidden, random);
      memory.layer(layer).forward(hidden, chunk, rotation, banks[layer], out);

      for (float value : out) {
        assertThat(Float.isFinite(value)).isTrue();
      }
      double magnitude = l1(out);
      assertThat(magnitude)
          .as("memory layer %d must contribute to the residual once a chunk has committed", layer)
          .isGreaterThan(0.0);
      System.out.println(
          "[edgerock] layer " + layer + " memory L1 after one commit = " + magnitude);
    }
  }

  private static void randomise(float[] values, java.util.Random random) {
    for (int i = 0; i < values.length; i++) {
      values[i] = (float) random.nextGaussian() * 0.1f;
    }
  }

  private static double l1(float[] values) {
    double total = 0;
    for (float value : values) {
      total += Math.abs(value);
    }
    return total;
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
