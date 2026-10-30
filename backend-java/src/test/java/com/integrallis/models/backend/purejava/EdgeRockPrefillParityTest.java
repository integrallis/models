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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.qwen35.Qwen35ForwardPass;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Batched prefill must give EdgeRock the same logits as one token at a time.
 *
 * <p>EdgeRock was clamped to one token per prefill step until the memory contribution was wired
 * into the batched path. The risk that clamp was guarding against is specific: a token reads the
 * bank before writing to it, and a write can trigger the commit that changes what later tokens
 * read, so the rows of a batch are ordered with respect to one another. Getting that order wrong
 * does not throw -- it shifts the logits.
 *
 * <p>The prompt length is the point of this test. Below 256 tokens the bank never commits, the read
 * sees nothing and the memory contributes exactly zero, so a parity test on a short prompt would
 * pass with the memory removed entirely. This runs past the commit chunk so that at least one
 * commit lands inside a batch rather than on a boundary.
 */
@Tag("integration")
@EnabledIfSystemProperty(named = EdgeRockPrefillParityTest.FIXTURE_PROPERTY, matches = ".+")
class EdgeRockPrefillParityTest {

  static final String FIXTURE_PROPERTY = "models.fixtures.edgeRock4bGguf";

  /** Past the 256-token commit chunk, so a commit falls inside a batch. */
  private static final int TOKENS = 300;

  @Test
  void batchedPrefillAgreesWithOneTokenAtATimeAcrossACommit() throws IOException {
    Path model = Path.of(System.getProperty(FIXTURE_PROPERTY));
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "2048");

    try (Arena arena = Arena.ofShared()) {
      GgufFile file = GgufParser.parse(model, arena);
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(file);
      assertThat(graph.config().usesMemoryAnchors()).isTrue();
      assertThat(graph.config().attentionWindow()).isEqualTo(1_024);

      int[] prompt = new int[TOKENS];
      // Deterministic, in-vocabulary, and not a repeated token: a constant prompt would give every
      // staged anchor the same admission score and hide an ordering error behind a tie.
      for (int index = 0; index < TOKENS; index++) {
        prompt[index] = 1_000 + (index * 7_919) % 100_000;
      }

      Qwen35ForwardPass.Session stepped = graph.openSession(TOKENS + 1);
      float[] expected = null;
      for (int index = 0; index < prompt.length; index++) {
        expected = graph.forward(stepped, prompt[index], index);
      }
      float[] oneAtATime = expected.clone();

      for (int width : new int[] {32, 64, 128}) {
        Qwen35ForwardPass.Session session = graph.openSession(TOKENS + 1);
        int position = 0;
        while (position < prompt.length - 1) {
          int span = Math.min(width, prompt.length - 1 - position);
          graph.prefill(session, Arrays.copyOfRange(prompt, position, position + span), position);
          position += span;
        }
        float[] batched = graph.forward(session, prompt[prompt.length - 1], prompt.length - 1);

        int argmaxBatched = argmax(batched);
        assertThat(argmaxBatched)
            .as("prefill width %d must choose the same token as one row at a time", width)
            .isEqualTo(argmax(oneAtATime));
        // The batched kernels reduce in a different order, so this is a tolerance rather than
        // equality. It is tight enough that a dropped or misordered memory contribution -- which
        // measured L1 5.94 on one layer alone -- could not hide inside it.
        for (int index = 0; index < batched.length; index++) {
          assertThat(batched[index])
              .as("prefill width %d, logit %d", width, index)
              .isCloseTo(oneAtATime[index], org.assertj.core.api.Assertions.within(0.05f));
        }
      }
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
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }
}
