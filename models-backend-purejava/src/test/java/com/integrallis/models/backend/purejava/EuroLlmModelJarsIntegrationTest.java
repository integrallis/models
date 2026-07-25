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
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRequirement;

@Tag("integration")
class EuroLlmModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;
  private static final String TRANSLATION_PROMPT =
      "<|im_start|>system\n"
          + "<|im_end|>\n"
          + "<|im_start|>user\n"
          + "Translate the following English source text to Portuguese:\n"
          + "English: The sky is blue.\n"
          + "Portuguese:<|im_end|>\n"
          + "<|im_start|>assistant\n";

  private static final ModelJarRequirement EUROLLM_1_7B_Q4_K_M =
      ModelJarRequirement.forSource("hf://mradermacher/EuroLLM-1.7B-Instruct-GGUF")
          .versionRange("[1.0.0,2.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("translation")
          .build();

  @Test
  void pinnedArtifactContainsExpectedMultilingualLlamaContract() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    assertThat(descriptor.sizeBytes().orElseThrow()).isEqualTo(1_045_157_088L);
    assertThat(descriptor.sha256().orElseThrow())
        .isEqualTo("1cade17f491ea46a686dbee51fbd52442e0f001f102380c3b9d66b4a77f84093");
    assertThat(descriptor.features())
        .contains("multilingual-35", "sentencepiece", "unaligned-output-warning");

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);

      assertThat(file.metadata().getString("general.architecture")).contains("llama");
      assertThat(file.metadata().getString("tokenizer.ggml.model")).contains("llama");
      assertThat(file.metadata().getUint32("llama.block_count")).contains(24);
      assertThat(file.metadata().getUint32("llama.context_length")).contains(8_192);
      assertThat(file.metadata().getUint32("llama.embedding_length")).contains(2_048);
      assertThat(file.metadata().getUint32("llama.feed_forward_length")).contains(5_632);
      assertThat(file.metadata().getUint32("llama.attention.head_count")).contains(16);
      assertThat(file.metadata().getUint32("llama.attention.head_count_kv")).contains(8);
      assertThat(file.metadata().getArraySize("tokenizer.ggml.tokens")).contains(128_000);
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

      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      assertThat(tokenizer.encode(TRANSLATION_PROMPT))
          .containsExactly(
              1, 3, 2205, 271, 4, 119715, 271, 3, 15236, 271, 31702, 31817, 557, 5302, 6771, 7684,
              6001, 591, 53439, 119782, 271, 31601, 119782, 806, 14930, 656, 15388, 119735, 271,
              23392, 19269, 1046, 119782, 4, 119715, 271, 3, 58406, 271);
    }
  }

  @Test
  void matchesLlamaCppGreedyPortugueseCompletionTokens() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));
    ModelOracleTestSupport.assertPanamaEnabled();

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
      assertThat(backend.metadata().contextLength()).isEqualTo(8_192);
      assertThat(backend.executionPlan().finalLayerPrefillPruning()).isTrue();
      assertThat(backend.executionPlan().finalLayerKvOnlyPrefill()).isTrue();

      int[] promptTokens = backend.tokenizer().encode(TRANSLATION_PROMPT);
      assertThat(promptTokens)
          .containsExactly(
              1, 3, 2205, 271, 4, 119715, 271, 3, 15236, 271, 31702, 31817, 557, 5302, 6771, 7684,
              6001, 591, 53439, 119782, 271, 31601, 119782, 806, 14930, 656, 15388, 119735, 271,
              23392, 19269, 1046, 119782, 4, 119715, 271, 3, 58406, 271);
      assertThat(ModelOracleTestSupport.greedyTokens(backend, promptTokens, 6))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned EuroLLM GGUF")
          .containsExactly(119802, 83672, 775, 35784, 119735, 4);
    } finally {
      ModelOracleTestSupport.restoreSystemProperty(
          PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledArtifact() {
    return ModelOracleTestSupport.installedDescriptor(
        EUROLLM_1_7B_Q4_K_M, ":models-backend-purejava:downloadEuroLlm17BQ4KMModel");
  }
}
