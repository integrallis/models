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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class RustFfmBackendIntegrationTest {
  private static final Path Q4_MODEL_PATH =
      Path.of(System.getProperty("user.home"), ".jvllm", "models", "Qwen3-0.6B-Q4_0.gguf");
  private static final Path Q8_MODEL_PATH =
      Path.of(
          System.getProperty("user.home"), ".jvllm", "models", "smollm2-360m-instruct-q8_0.gguf");
  private static final int[] EXPECTED_PROMPT_TOKENS = {785, 3974, 13876, 38835};
  private static final int[] EXPECTED_GENERATED_TOKENS = {34208, 916, 279, 15678};

  @Test
  void matchesPinnedQwen3GreedyTokenOracle() {
    assertThat(Q4_MODEL_PATH)
        .as("download the pinned Qwen3 fixture before running native integration tests")
        .isRegularFile();
    Path library = Path.of(System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY));
    assertThat(library).isRegularFile();
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "128");

    try (RustFfmBackend backend = RustFfmBackend.load(Q4_MODEL_PATH, library)) {
      assertThat(backend.name()).isEqualTo("rust-ffm");
      assertThat(backend.diagnostics().backend()).isEqualTo("rust-ffm");
      assertThat(backend.diagnostics().planVersion()).isEqualTo(RustFfmBackend.PLAN_VERSION);
      assertThat(backend.diagnostics().optimization("rust-q4-0-batched-matmul")).isPresent();
      assertThat(backend.executionPlan().groupedProjections()).isTrue();

      int[] promptTokens = backend.tokenizer().encode("The quick brown fox");
      assertThat(promptTokens).containsExactly(EXPECTED_PROMPT_TOKENS);
      assertThat(greedyTokens(backend, promptTokens, EXPECTED_GENERATED_TOKENS.length))
          .containsExactly(EXPECTED_GENERATED_TOKENS);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
    }
  }

  @Test
  void q8KernelMatchesEstablishedPureJavaGreedyOracle() {
    assertThat(Q8_MODEL_PATH)
        .as("download the pinned SmolLM2 fixture before running native integration tests")
        .isRegularFile();
    Path library = Path.of(System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY));
    assertThat(library).isRegularFile();
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "256");

    try {
      String prompt = "Explain why Java runs anywhere in one concise sentence.";
      int[] promptTokens;
      int[] expected;
      try (PureJavaBackend backend = PureJavaBackend.load(Q8_MODEL_PATH)) {
        promptTokens = backend.tokenizer().encode(prompt);
        expected = greedyTokens(backend, promptTokens, 8);
      }

      try (RustFfmBackend backend = RustFfmBackend.load(Q8_MODEL_PATH, library)) {
        assertThat(backend.diagnostics().optimization("rust-q8-0-batched-matmul")).isPresent();
        assertThat(backend.tokenizer().encode(prompt)).containsExactly(promptTokens);
        assertThat(greedyTokens(backend, promptTokens, expected.length)).containsExactly(expected);
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
    }
  }

  private static int[] greedyTokens(
      InferenceBackend backend, int[] promptTokens, int generatedTokenCount) {
    float[] logits = backend.prefill(promptTokens, 0);
    int[] generated = new int[generatedTokenCount];
    int position = promptTokens.length;
    for (int index = 0; index < generated.length; index++) {
      int token = argmax(logits);
      generated[index] = token;
      logits = backend.forward(token, position++);
    }
    return generated;
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

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
