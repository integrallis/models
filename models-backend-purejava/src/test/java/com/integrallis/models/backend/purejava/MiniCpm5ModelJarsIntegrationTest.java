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
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("integration")
class MiniCpm5ModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;

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
  void matchesLlamaCppGreedyCodeCompletionTokens() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
      assertThat(backend.metadata().contextLength()).isEqualTo(131_072);

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
