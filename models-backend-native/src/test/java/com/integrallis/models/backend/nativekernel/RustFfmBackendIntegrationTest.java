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
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
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
  private static final Path MINICPM5_MODEL_PATH =
      Path.of(System.getProperty("user.home"), ".jvllm", "models", "MiniCPM5-1B-Q4_K_M.gguf");
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

  @Test
  void kQuantizedKernelMatchesPinnedMiniCpm5GreedyOracle() {
    assertThat(MINICPM5_MODEL_PATH)
        .as("download the pinned MiniCPM5 fixture before running native integration tests")
        .isRegularFile();
    Path library = Path.of(System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY));
    assertThat(library).isRegularFile();
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    String previousNativeDecode =
        System.getProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "128");
    System.setProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, "true");

    try (RustFfmBackend backend = RustFfmBackend.load(MINICPM5_MODEL_PATH, library)) {
      assertThat(backend.diagnostics().optimization("rust-q4-k-batched-matmul")).isPresent();
      assertThat(backend.diagnostics().optimization("rust-q5-k-batched-matmul")).isPresent();
      assertThat(backend.diagnostics().optimization("rust-q6-k-batched-matmul")).isPresent();
      assertThat(backend.diagnostics().optimization("rust-mixed-k-grouped-matmul")).isPresent();

      int[] promptTokens = backend.tokenizer().encode("public static void main(String[] args) {");
      assertThat(promptTokens)
          .containsExactly(12243, 10254, 9249, 1903, 37559, 21099, 36758, 30, 319);
      assertThat(greedyTokens(backend, promptTokens, 4))
          .as("native greedy token IDs must match the pinned llama.cpp MiniCPM5 oracle")
          .containsExactly(5028, 6706, 5018, 1735);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
      restoreSystemProperty(
          RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, previousNativeDecode);
    }
  }

  @Test
  void kQuantizedKernelMatchesVectorProjectionOracleOnPinnedMiniCpm5Weights() throws Exception {
    assertThat(MINICPM5_MODEL_PATH).isRegularFile();
    Path library = Path.of(System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY));
    assertThat(library).isRegularFile();

    try (Arena arena = Arena.ofShared();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(library, true)) {
      var file = GgufParser.parse(MINICPM5_MODEL_PATH, arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaWeights.LayerWeights layer = weights.layer(0);
      float[] input = new float[config.embeddingDim()];
      float[] normalized = new float[config.embeddingDim()];
      weights.embedToken(12243, input);
      TensorOps.rmsNorm(
          normalized, input, layer.attentionNorm(), config.embeddingDim(), config.rmsNormEps());

      assertProjectionMatches(
          kernel, normalized, layer.wq(), layer.wqType(), config.queryDim(), config.embeddingDim());
      assertProjectionMatches(
          kernel, normalized, layer.wv(), layer.wvType(), config.valueDim(), config.embeddingDim());
    }
  }

  private static void assertProjectionMatches(
      RustGgufBatchedMatrixKernel kernel,
      float[] input,
      MemorySegment weights,
      GgufTensorType type,
      int rows,
      int cols) {
    float[] expected = new float[rows];
    float[] actual = new float[rows];
    TensorOps.ggufMatmul(expected, input, weights, type, rows, cols);
    kernel.multiply(actual, input, weights, type, 1, rows, cols);

    float maximumDifference = 0.0f;
    int maximumIndex = 0;
    for (int index = 0; index < rows; index++) {
      float difference = Math.abs(expected[index] - actual[index]);
      if (difference > maximumDifference) {
        maximumDifference = difference;
        maximumIndex = index;
      }
    }
    assertThat(maximumDifference)
        .as(
            "%s maximum projection difference at row %s: expected=%s actual=%s",
            type, maximumIndex, expected[maximumIndex], actual[maximumIndex])
        .isLessThanOrEqualTo(1e-4f);
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
