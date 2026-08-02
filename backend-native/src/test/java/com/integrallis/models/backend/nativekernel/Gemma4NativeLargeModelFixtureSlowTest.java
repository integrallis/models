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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
class Gemma4NativeLargeModelFixtureSlowTest {

  private static final String FILE_NAME = "gemma-4-26B-A4B-it-Q4_K_M.gguf";
  private static final long FILE_SIZE = 16_796_015_136L;
  private static final int[] EXPECTED_TOKENS = {
    9259, 236888, 2088, 740, 564, 1601, 611, 3124, 236881, 106
  };

  @Test
  void pinnedArtifactMatchesTheGreedyOracleThroughRustFfm() throws Exception {
    Path model = fixturePath();
    assertThat(model).isRegularFile();
    assertThat(Files.size(model)).isEqualTo(FILE_SIZE);

    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    String previousNativeDecode =
        System.getProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY);
    String previousLoadWarmup = System.getProperty(RustFfmBackend.LOAD_WARMUP_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "128");
    System.setProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, "true");
    System.setProperty(RustFfmBackend.LOAD_WARMUP_PROPERTY, "true");

    long loadStarted = System.nanoTime();
    try (RustFfmBackend backend = RustFfmBackend.load(model)) {
      double loadMillis = elapsedMillis(loadStarted);
      assertThat(backend.diagnostics().environment())
          .containsEntry("kernel-runtime", "rust-ffm")
          .containsEntry("native-quantized-decode", "true")
          .containsEntry("native-load-warmup", "true");

      ModelPrompt prompt = ChatTemplate.GEMMA4.render(List.of(ChatMessage.user("Hello")));
      int[] promptTokens = backend.tokenizer().encode(prompt);
      long prefillStarted = System.nanoTime();
      float[] logits = backend.prefill(promptTokens, 0);
      double prefillMillis = elapsedMillis(prefillStarted);
      double[] decodeMillis = new double[EXPECTED_TOKENS.length];
      int[] actual = new int[EXPECTED_TOKENS.length];
      for (int index = 0; index < EXPECTED_TOKENS.length; index++) {
        actual[index] = argmax(logits);
        long decodeStarted = System.nanoTime();
        logits = backend.forward(EXPECTED_TOKENS[index], promptTokens.length + index);
        decodeMillis[index] = elapsedMillis(decodeStarted);
      }

      assertThat(actual)
          .as("native greedy token IDs must match llama.cpp a582222 for the pinned Gemma 4 GGUF")
          .containsExactly(EXPECTED_TOKENS);
      backend.reset();
      long warmPrefillStarted = System.nanoTime();
      float[] warmLogits = backend.prefill(promptTokens, 0);
      double warmPrefillMillis = elapsedMillis(warmPrefillStarted);
      assertThat(argmax(warmLogits)).isEqualTo(EXPECTED_TOKENS[0]);
      System.out.printf(
          Locale.ROOT,
          "GEMMA4_NATIVE_METRICS load_ms=%.3f prefill_tokens=%d prefill_ms=%.3f "
              + "warm_prefill_ms=%.3f decode_ms=%s%n",
          loadMillis,
          promptTokens.length,
          prefillMillis,
          warmPrefillMillis,
          format(decodeMillis));
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
      restoreSystemProperty(
          RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, previousNativeDecode);
      restoreSystemProperty(RustFfmBackend.LOAD_WARMUP_PROPERTY, previousLoadWarmup);
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

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static double elapsedMillis(long startNanos) {
    return (System.nanoTime() - startNanos) / 1_000_000.0;
  }

  private static String format(double[] values) {
    StringBuilder result = new StringBuilder();
    for (int index = 0; index < values.length; index++) {
      if (index > 0) {
        result.append(',');
      }
      result.append(String.format(Locale.ROOT, "%.3f", values[index]));
    }
    return result.toString();
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
