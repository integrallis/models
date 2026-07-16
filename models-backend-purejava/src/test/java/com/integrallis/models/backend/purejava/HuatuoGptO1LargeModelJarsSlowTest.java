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
import com.integrallis.vectors.core.VectorizationProvider;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("slow")
class HuatuoGptO1LargeModelJarsSlowTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;
  private static final String MEDICAL_PROMPT =
      "Question: Which organ pumps blood through the human body?\nAnswer:";

  private static final ModelJarRequirement HUATUOGPT_O1_7B_Q4_K_M =
      ModelJarRequirement.forSource("hf://bartowski/HuatuoGPT-o1-7B-GGUF")
          .versionRange("[1.0.0,2.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("medical-reasoning")
          .build();

  @Test
  void pinnedArtifactContainsExpectedQwen2MedicalModelContract() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    assertThat(descriptor.sizeBytes().orElseThrow()).isEqualTo(4_683_074_720L);
    assertThat(descriptor.sha256().orElseThrow())
        .isEqualTo("4643521a184cb26df0f7c57da9aead0c632b286a9aff103c9f9dca4dc059abd7");
    assertThat(descriptor.features()).contains("medical-use-warning");

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);

      assertThat(file.metadata().getString("general.architecture")).contains("qwen2");
      assertThat(file.metadata().getString("tokenizer.ggml.model")).contains("gpt2");
      assertThat(file.metadata().getString("tokenizer.ggml.pre")).contains("qwen2");
      assertThat(file.metadata().getUint32("qwen2.block_count")).contains(28);
      assertThat(file.metadata().getUint32("qwen2.context_length")).contains(32_768);
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
      assertThat(tokenizer.encode(MEDICAL_PROMPT))
          .containsExactly(
              14582, 25, 15920, 2872, 42775, 6543, 1526, 279, 3738, 2487, 5267, 16141, 25);
    }
  }

  @Test
  void matchesLlamaCppGreedyMedicalCompletionTokens() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    assertThat(VectorizationProvider.isPanamaEnabled())
        .as(
            "large-model token oracle requires Panama SIMD; provider=%s, panamaFailure=%s",
            VectorizationProvider.getProviderName(),
            VectorizationProvider.getPanamaFailure().map(Throwable::toString).orElse("none"))
        .isTrue();

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen2");
      assertThat(backend.metadata().contextLength()).isEqualTo(32_768);

      int[] promptTokens = backend.tokenizer().encode(MEDICAL_PROMPT);
      assertThat(promptTokens)
          .containsExactly(
              14582, 25, 15920, 2872, 42775, 6543, 1526, 279, 3738, 2487, 5267, 16141, 25);
      assertThat(greedyTokens(backend, promptTokens, 4))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned HuatuoGPT-o1 GGUF")
          .containsExactly(576, 4746, 374, 279);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledArtifact() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(HUATUOGPT_O1_7B_Q4_K_M).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run :models-backend-purejava:huatuoGptO17BSlowTest.",
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
