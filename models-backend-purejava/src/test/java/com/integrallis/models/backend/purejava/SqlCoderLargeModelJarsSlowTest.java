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

import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("slow")
class SqlCoderLargeModelJarsSlowTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;
  private static final String SQL_PROMPT = "SQL:";

  private static final ModelJarRequirement SQLCODER_7B_2_Q5_K_M =
      ModelJarRequirement.forSource("hf://defog/sqlcoder-7b-2")
          .versionRange("[2.0.0,3.0.0)")
          .variant("q5_k_m")
          .backend("pure-java")
          .capability("text-to-sql")
          .build();

  @Test
  void pinnedArtifactContainsExpectedLlamaQ5KTensorLayout() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);

      assertThat(file.metadata().getString("general.architecture")).contains("llama");
      assertThat(file.metadata().getString("tokenizer.ggml.model")).contains("llama");
      assertThat(file.metadata().getUint32("llama.block_count")).contains(32);
      assertThat(file.metadata().getUint32("llama.context_length")).contains(16_384);
      assertThat(file.metadata().getFloat32("llama.rope.freq_base")).contains(1_000_000.0f);
      assertThat(file.tensorInfos()).hasSize(291);
      assertThat(file.getTensor("token_embd.weight").type()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(file.getTensor("output.weight").type()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(file.getTensor("blk.0.attn_q.weight").type()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(file.getTensor("blk.0.attn_k.weight").type()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(file.getTensor("blk.0.attn_v.weight").type()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(file.getTensor("blk.0.ffn_gate.weight").type()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(file.getTensor("blk.0.ffn_up.weight").type()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(file.getTensor("blk.0.ffn_down.weight").type()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(file.tensorInfos())
          .extracting(tensor -> tensor.type())
          .contains(GgufTensorType.F32, GgufTensorType.Q5_K, GgufTensorType.Q6_K);
    }
  }

  @Test
  void groupedQ5KProjectionsMatchSeparateLayerMatmulsExactly() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofShared()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaWeights.LayerWeights layer = weights.layer(0);

      assertThat(layer.wqType()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(layer.wkType()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(layer.wvType()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(layer.ffnGateType()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(layer.ffnUpType()).isEqualTo(GgufTensorType.Q5_K);

      int cols = config.embeddingDim();
      float[] input = new float[cols];
      float[] normalized = new float[cols];
      weights.embedToken(3758, input);
      TensorOps.rmsNorm(normalized, input, layer.attentionNorm(), cols, config.rmsNormEps());

      assertTripleProjectionMatchesSeparate(
          normalized, cols, layer, config.queryDim(), config.keyDim(), config.valueDim());
      assertDualProjectionMatchesSeparate(normalized, cols, layer, config.hiddenDim());
    }
  }

  @Test
  void matchesLlamaCppGreedyTextToSqlTokens() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
      assertThat(backend.metadata().contextLength()).isEqualTo(16_384);

      int[] promptTokens = backend.tokenizer().encode(SQL_PROMPT);
      assertThat(promptTokens).containsExactly(1, 3758, 29901);
      assertThat(greedyTokens(backend, promptTokens, 4))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned SQLCoder GGUF")
          .containsExactly(13, 917, 29901, 274);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledArtifact() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(SQLCODER_7B_2_Q5_K_M).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run the SQLCoder 7B fixture download task before this test.",
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
      if (index + 1 < count) {
        logits = backend.forward(token, position++);
      }
    }
    return generated;
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
