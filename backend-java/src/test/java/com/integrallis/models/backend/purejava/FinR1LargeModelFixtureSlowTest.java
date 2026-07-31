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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.lang.foreign.Arena;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
class FinR1LargeModelFixtureSlowTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;
  private static final String SYSTEM_PROMPT =
      "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
          + "You first think about the reasoning process as an internal monologue and then provide "
          + "the user with the answer. Respond in the following format: <think>\n"
          + "...\n"
          + "</think>\n"
          + "<answer>\n"
          + "...\n"
          + "</answer>";
  private static final ModelPrompt FINANCIAL_PROMPT =
      ChatTemplate.CHATML.render(
          List.of(
              ChatMessage.system(SYSTEM_PROMPT),
              ChatMessage.user("If a $100 investment gains 10%, what is its value?")));

  private static final ModelFixtureRequirement FIN_R1_7B_Q4_K_M =
      ModelFixtureRequirement.of("hf://bartowski/SUFE-AIFLM-Lab_Fin-R1-GGUF")
          .version("[1.0.0,2.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("financial-reasoning");

  @Test
  void pinnedArtifactContainsExpectedQwen2FinancialModelContract() throws Exception {
    ModelFixtureDescriptor descriptor = descriptorWithInstalledArtifact();

    assertThat(descriptor.sizeBytes().orElseThrow()).isEqualTo(4_683_073_600L);
    assertThat(descriptor.sha256().orElseThrow())
        .isEqualTo("d50f16c5149b4dc103c68e249a136ab7c82f7569a7df707a2d6150bff5994c33");
    assertThat(descriptor.features())
        .contains("financial-advice-warning", "license-card-only", "thinking-final-response");

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);

      assertThat(file.metadata().getString("general.architecture")).contains("qwen2");
      assertThat(file.metadata().getString("tokenizer.ggml.model")).contains("gpt2");
      assertThat(file.metadata().getString("tokenizer.ggml.pre")).contains("qwen2");
      assertThat(file.metadata().getUint32("qwen2.block_count")).contains(28);
      assertThat(file.metadata().getUint32("qwen2.context_length")).contains(32_768);
      assertThat(file.metadata().getUint32("qwen2.embedding_length")).contains(3_584);
      assertThat(file.metadata().getUint32("qwen2.feed_forward_length")).contains(18_944);
      assertThat(file.metadata().getUint32("qwen2.attention.head_count")).contains(28);
      assertThat(file.metadata().getUint32("qwen2.attention.head_count_kv")).contains(4);
      assertThat(file.metadata().getFloat32("qwen2.rope.freq_base")).contains(1_000_000.0f);
      assertThat(file.tensorInfos()).hasSize(339);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.F32)
          .hasSize(141);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.Q4_K)
          .hasSize(169);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.Q6_K)
          .hasSize(29);

      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      assertThat(tokenizer.encode(FINANCIAL_PROMPT))
          .containsExactly(
              151644, 8948, 198, 2610, 525, 264, 10950, 15235, 21388, 429, 5707, 1632, 5504, 1497,
              291, 323, 11682, 14507, 13, 1446, 1156, 1744, 911, 279, 32711, 1882, 438, 458, 5306,
              1615, 76728, 323, 1221, 3410, 279, 1196, 448, 279, 4226, 13, 39533, 304, 279, 2701,
              3561, 25, 366, 26865, 397, 9338, 522, 26865, 397, 27, 9217, 397, 9338, 522, 9217, 29,
              151645, 198, 151644, 872, 198, 2679, 264, 400, 16, 15, 15, 9162, 19619, 220, 16, 15,
              13384, 1128, 374, 1181, 897, 30, 151645, 198, 151644, 77091, 198);
    }
  }

  @Test
  void matchesLlamaCppGreedyFinancialReasoningTokens() {
    ModelFixtureDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));
    ModelOracleTestSupport.assertPanamaEnabled();

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor.localPath().orElseThrow())) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen2");
      assertThat(backend.metadata().contextLength()).isEqualTo(32_768);

      int[] promptTokens = backend.tokenizer().encode(FINANCIAL_PROMPT);
      assertThat(promptTokens)
          .containsExactly(
              151644, 8948, 198, 2610, 525, 264, 10950, 15235, 21388, 429, 5707, 1632, 5504, 1497,
              291, 323, 11682, 14507, 13, 1446, 1156, 1744, 911, 279, 32711, 1882, 438, 458, 5306,
              1615, 76728, 323, 1221, 3410, 279, 1196, 448, 279, 4226, 13, 39533, 304, 279, 2701,
              3561, 25, 366, 26865, 397, 9338, 522, 26865, 397, 27, 9217, 397, 9338, 522, 9217, 29,
              151645, 198, 151644, 872, 198, 2679, 264, 400, 16, 15, 15, 9162, 19619, 220, 16, 15,
              13384, 1128, 374, 1181, 897, 30, 151645, 198, 151644, 77091, 198);
      assertThat(ModelOracleTestSupport.greedyTokens(backend, promptTokens, 12))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned Fin-R1 GGUF")
          .containsExactly(13708, 766, 397, 32313, 11, 1077, 594, 21403, 419, 3491, 3019, 553);
    } finally {
      ModelOracleTestSupport.restoreSystemProperty(
          PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static ModelFixtureDescriptor descriptorWithInstalledArtifact() {
    return ModelOracleTestSupport.installedDescriptor(
        FIN_R1_7B_Q4_K_M, ":backend-java:downloadFinR17BQ4KMModel");
  }
}
