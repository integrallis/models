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

@Tag("slow")
class SmolLm3ModelJarsSlowTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;

  private static final ModelJarRequirement SMOLLM3_3B_Q4_K_M =
      ModelJarRequirement.forSource("hf://ggml-org/SmolLM3-3B-GGUF")
          .versionRange("[3.0.0,4.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("text-generation")
          .build();

  @Test
  void pinnedArtifactContainsSupportedSmolLm3TensorLayout() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);

      assertThat(file.metadata().getString("general.architecture")).contains("smollm3");
      assertThat(file.metadata().getString("tokenizer.ggml.model")).contains("gpt2");
      assertThat(file.metadata().getString("tokenizer.ggml.pre")).contains("smaug-bpe");
      assertThat(file.metadata().getUint32("smollm3.block_count")).contains(36);
      assertThat(file.metadata().getUint32("smollm3.context_length")).contains(65_536);
      assertThat(file.metadata().getFloat32("smollm3.rope.freq_base")).contains(5_000_000.0f);
      assertThat(file.tensorInfos()).hasSize(326);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.F32)
          .hasSize(73);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.Q4_K)
          .hasSize(216);
      assertThat(file.tensorInfos())
          .filteredOn(tensor -> tensor.type() == GgufTensorType.Q6_K)
          .hasSize(37);

      var tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      assertThat(tokenizer.encode("The quick brown fox")).containsExactly(791, 4062, 14198, 39935);
      assertThat(tokenizer.encode("1234567890")).containsExactly(4513, 10961, 16474, 15);
      assertThat(tokenizer.encode("I'm testing SmolLM3."))
          .containsExactly(40, 2846, 7649, 4487, 337, 11237, 18, 13);
      assertThat(tokenizer.encode("camelCase42HTTP")).containsExactly(94421, 4301, 2983, 9412);
      assertThat(tokenizer.encode("你好，SmolLM3！"))
          .containsExactly(57668, 53901, 3922, 10902, 337, 11237, 18, 6447);
      assertThat(tokenizer.encode("Hello   world\nnext"))
          .containsExactly(9906, 256, 1917, 198, 3684);
    }
  }

  @Test
  void matchesLlamaCppGreedyCompletionTokens() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("smollm3");
      assertThat(backend.metadata().contextLength()).isEqualTo(65_536);

      int[] promptTokens = backend.tokenizer().encode("The quick brown fox");
      assertThat(promptTokens).containsExactly(791, 4062, 14198, 39935);
      assertThat(greedyTokens(backend, promptTokens, 4))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned SmolLM3 GGUF")
          .containsExactly(35308, 927, 279, 16053);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledArtifact() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(SMOLLM3_3B_Q4_K_M).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run :models-backend-purejava:smolLm33BSlowTest.",
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
