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

import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("integration")
class MiniCpm5ModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;
  private static final String MIXED_K_PROJECTIONS_PROPERTY = "models.purejava.mixedKProjections";
  private static final String PREFILL_BATCH_SIZE_PROPERTY = "models.purejava.prefillBatchSize";

  private static final ModelJarRequirement MINICPM5_1B_Q4_K_M =
      ModelJarRequirement.forSource("hf://openbmb/MiniCPM5-1B-GGUF")
          .versionRange("[5.0.0,6.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("text-generation")
          .build();

  @Test
  void pinnedArtifactContainsSupportedLlamaTensorLayout() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);

      assertThat(file.metadata().getString("general.architecture")).contains("llama");
      assertThat(file.metadata().getString("tokenizer.ggml.model")).contains("gpt2");
      assertThat(file.metadata().getString("tokenizer.ggml.pre")).contains("llama-bpe");
      assertThat(file.metadata().getUint32("llama.attention.key_length")).contains(128);
      assertThat(file.metadata().getUint32("llama.attention.value_length")).contains(128);

      var tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      assertThat(tokenizer.encode("Hello123 world!")).containsExactly(36417, 5645, 1782, 22);
      assertThat(tokenizer.encode("I'm testing MiniCPM5."))
          .containsExactly(62, 2077, 7961, 44524, 7739, 66, 42, 35);
      assertThat(tokenizer.encode("camelCase42HTTP"))
          .containsExactly(88, 30327, 19282, 2189, 83076);
      assertThat(tokenizer.encode("你好，MiniCPM5！"))
          .containsExactly(75828, 74717, 7053, 7739, 66, 42, 1398);
      assertThat(tokenizer.encode("1231d29e25650029\nYou"))
          .containsExactly(5645, 38, 89, 1096, 90, 7373, 3161, 1096, 220, 2311);
      assertThat(tokenizer.encode("168037df5d951dc1\nYou"))
          .containsExactly(7018, 15571, 4865, 42, 89, 18491, 13920, 38, 220, 2311);

      assertThat(file.tensorInfos()).hasSize(219);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.F32)
          .hasSize(49);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.Q4_K)
          .hasSize(145);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.Q6_K)
          .hasSize(25);
    }
  }

  @Test
  void groupedQ4KProjectionsMatchSeparateLayerMatmulsExactly() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofShared()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaWeights.LayerWeights layer = weights.layer(0);

      assertThat(layer.wqType()).isEqualTo(GgufTensorType.Q4_K);
      assertThat(layer.wkType()).isEqualTo(GgufTensorType.Q4_K);
      assertThat(layer.wvType()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(layer.ffnGateType()).isEqualTo(GgufTensorType.Q4_K);
      assertThat(layer.ffnUpType()).isEqualTo(GgufTensorType.Q4_K);

      int cols = config.embeddingDim();
      float[] input = new float[cols];
      float[] normalized = new float[cols];
      weights.embedToken(36417, input);
      TensorOps.rmsNorm(normalized, input, layer.attentionNorm(), cols, config.rmsNormEps());

      assertTripleProjectionMatchesSeparate(
          normalized, cols, layer, config.queryDim(), config.keyDim(), config.valueDim());
      assertDualProjectionMatchesSeparate(normalized, cols, layer, config.hiddenDim());
    }
  }

  @Test
  void matchesLlamaCppGreedyCodeCompletionTokens() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
      assertThat(backend.metadata().contextLength()).isEqualTo(131_072);
      assertThat(backend.executionPlan().mixedKProjections()).isTrue();
      assertThat(backend.diagnostics().optimization("mixed-k-projections"))
          .hasValueSatisfying(
              decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));

      int[] promptTokens = backend.tokenizer().encode("public static void main(String[] args) {");
      assertThat(promptTokens)
          .containsExactly(12243, 10254, 9249, 1903, 37559, 21099, 36758, 30, 319);
      assertThat(greedyTokens(backend, promptTokens, 4))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned MiniCPM5 GGUF")
          .containsExactly(5028, 6706, 5018, 1735);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  @Test
  void mixedKProjectionGroupingMatchesTheTwoDispatchControlExactly() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    String previousMixedK = System.getProperty(MIXED_K_PROJECTIONS_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));
    System.setProperty(MIXED_K_PROJECTIONS_PROPERTY, "false");

    try {
      float[] expected;
      int[] promptTokens;
      try (PureJavaBackend control = PureJavaBackend.load(descriptor)) {
        assertThat(control.executionPlan().mixedKProjections()).isFalse();
        promptTokens = control.tokenizer().encode("public static void main(String[] args) {");
        expected = finalLogits(control, promptTokens, 2);
      }

      System.setProperty(MIXED_K_PROJECTIONS_PROPERTY, "true");
      try (PureJavaBackend grouped = PureJavaBackend.load(descriptor)) {
        assertThat(grouped.executionPlan().mixedKProjections()).isTrue();
        assertThat(finalLogits(grouped, promptTokens, 2)).containsExactly(expected);
      }
    } finally {
      restoreSystemProperty(MIXED_K_PROJECTIONS_PROPERTY, previousMixedK);
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
    }
  }

  @Test
  void batchedPrefillMatchesSequentialInferenceExactly() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    String previousBatchSize = System.getProperty(PREFILL_BATCH_SIZE_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));
    System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, "1");

    try (PureJavaBackend sequential = PureJavaBackend.load(descriptor)) {
      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, "32");
      try (PureJavaBackend batched = PureJavaBackend.load(descriptor)) {
        int[] promptTokens =
            sequential.tokenizer().encode("public static void main(String[] args) {");
        float[] expected = null;
        for (int position = 0; position < promptTokens.length; position++) {
          expected = sequential.forward(promptTokens[position], position);
        }

        float[] actual = batched.prefill(promptTokens, 0);
        assertThat(actual).containsExactly(expected);

        for (int generated = 0; generated < 2; generated++) {
          int token = argmax(expected);
          int position = promptTokens.length + generated;
          expected = sequential.forward(token, position);
          actual = batched.forward(token, position);
          assertThat(actual)
              .as("decode logits at position %d after batched prefill", position)
              .containsExactly(expected);
        }
      }
    } finally {
      restoreSystemProperty(PREFILL_BATCH_SIZE_PROPERTY, previousBatchSize);
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledArtifact() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(MINICPM5_1B_Q4_K_M).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run :models-backend-purejava:integrationTest or the fixture"
                + " download task before running this test.",
            descriptor.localPath().orElseThrow())
        .isTrue();
    return descriptor;
  }

  private static int[] greedyTokens(PureJavaBackend backend, int[] promptTokens, int count) {
    backend.reset();
    float[] logits = null;
    int position = 0;
    for (int token : promptTokens) {
      logits = backend.forward(token, position++);
    }

    int[] generated = new int[count];
    for (int index = 0; index < count; index++) {
      int token = argmax(logits);
      generated[index] = token;
      logits = backend.forward(token, position++);
    }
    return generated;
  }

  private static float[] finalLogits(
      PureJavaBackend backend, int[] promptTokens, int generatedTokens) {
    float[] logits = null;
    int position = 0;
    for (int token : promptTokens) {
      logits = backend.forward(token, position++);
    }
    for (int index = 0; index < generatedTokens; index++) {
      logits = backend.forward(argmax(logits), position++);
    }
    return logits;
  }

  private static void assertTripleProjectionMatchesSeparate(
      float[] input,
      int cols,
      LlamaWeights.LayerWeights layer,
      int queryRows,
      int keyRows,
      int valueRows) {
    float[] expectedQuery = new float[queryRows];
    float[] expectedKey = new float[keyRows];
    float[] expectedValue = new float[valueRows];
    float[] actualQuery = new float[queryRows];
    float[] actualKey = new float[keyRows];
    float[] actualValue = new float[valueRows];
    TensorOps.ggufMatmul(expectedQuery, input, layer.wq(), layer.wqType(), queryRows, cols);
    TensorOps.ggufMatmul(expectedKey, input, layer.wk(), layer.wkType(), keyRows, cols);
    TensorOps.ggufMatmul(expectedValue, input, layer.wv(), layer.wvType(), valueRows, cols);

    TensorOps.ggufTripleMatmul(
        actualQuery,
        layer.wq(),
        layer.wqType(),
        queryRows,
        actualKey,
        layer.wk(),
        layer.wkType(),
        keyRows,
        actualValue,
        layer.wv(),
        layer.wvType(),
        valueRows,
        input,
        cols,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    assertThat(actualQuery).containsExactly(expectedQuery);
    assertThat(actualKey).containsExactly(expectedKey);
    assertThat(actualValue).containsExactly(expectedValue);
  }

  private static void assertDualProjectionMatchesSeparate(
      float[] input, int cols, LlamaWeights.LayerWeights layer, int hiddenRows) {
    float[] expectedGate = new float[hiddenRows];
    float[] expectedUp = new float[hiddenRows];
    float[] actualGate = new float[hiddenRows];
    float[] actualUp = new float[hiddenRows];
    TensorOps.ggufMatmul(
        expectedGate, input, layer.ffnGate(), layer.ffnGateType(), hiddenRows, cols);
    TensorOps.ggufMatmul(expectedUp, input, layer.ffnUp(), layer.ffnUpType(), hiddenRows, cols);

    TensorOps.ggufDualMatmul(
        actualGate,
        layer.ffnGate(),
        layer.ffnGateType(),
        hiddenRows,
        actualUp,
        layer.ffnUp(),
        layer.ffnUpType(),
        hiddenRows,
        input,
        cols,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    assertThat(actualGate).containsExactly(expectedGate);
    assertThat(actualUp).containsExactly(expectedUp);
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
