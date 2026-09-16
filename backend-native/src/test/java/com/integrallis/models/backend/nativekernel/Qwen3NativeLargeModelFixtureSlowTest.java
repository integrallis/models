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

import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Verifies the owned Rust Q4_K path against the pinned Qwen3 8B llama.cpp oracle. */
@Tag("slow")
class Qwen3NativeLargeModelFixtureSlowTest {

  private static final String FILE_NAME = "Qwen3-8B-Q4_K_M.gguf";
  private static final long FILE_SIZE = 5_027_783_488L;
  private static final int[] EXPECTED_PROMPT_TOKENS = {785, 3974, 13876, 38835};
  private static final int[] EXPECTED_GENERATED_TOKENS = {34208, 916, 279, 15678};

  @Test
  void pinnedArtifactMatchesTheLlamaCppGreedyOracleThroughRustFfm() throws Exception {
    Path model = fixturePath();
    assertThat(model).isRegularFile();
    assertThat(Files.size(model)).isEqualTo(FILE_SIZE);

    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    String previousNativeDecode =
        System.getProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "128");
    System.setProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, "true");

    try (RustFfmBackend backend = RustFfmBackend.load(model)) {
      assertThat(backend.diagnostics().environment())
          .containsEntry("kernel-runtime", "rust-ffm")
          .containsEntry("native-quantized-decode", "true");
      assertThat(backend.diagnostics().optimization("rust-q4-k-batched-matmul")).isPresent();
      assertThat(backend.tokenizer().encode("The quick brown fox"))
          .containsExactly(EXPECTED_PROMPT_TOKENS);
      assertThat(greedyTokens(backend, EXPECTED_PROMPT_TOKENS, EXPECTED_GENERATED_TOKENS.length))
          .as("native greedy token IDs must match llama.cpp b9960 for the pinned Qwen3 8B GGUF")
          .containsExactly(EXPECTED_GENERATED_TOKENS);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
      restoreSystemProperty(
          RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, previousNativeDecode);
    }
  }

  private static Path fixturePath() {
    String configured = System.getProperty("models.fixtures.directory");
    Path directory =
        configured == null || configured.isBlank()
            ? Path.of(System.getProperty("user.home"), ".jvllm", "models")
            : Path.of(configured);
    return directory.resolve(FILE_NAME);
  }

  private static int[] greedyTokens(RustFfmBackend backend, int[] promptTokens, int count) {
    float[] logits = backend.prefill(promptTokens, 0);
    int[] generated = new int[count];
    int position = promptTokens.length;
    for (int index = 0; index < count; index++) {
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
