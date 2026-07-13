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
class DeepSeekR1DistillQwenLargeModelJarsSlowTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;

  private static final ModelJarRequirement DEEPSEEK_R1_DISTILL_QWEN_7B_Q4_K_M =
      ModelJarRequirement.forSource("hf://bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF")
          .versionRange("[1.0.0,2.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("reasoning")
          .build();

  @Test
  void pinnedArtifactContainsExpectedQwen2TensorLayout() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);

      assertThat(file.metadata().getString("general.architecture")).contains("qwen2");
      assertThat(file.metadata().getString("tokenizer.ggml.pre")).contains("deepseek-r1-qwen");
      assertThat(file.metadata().getUint32("qwen2.block_count")).contains(28);
      assertThat(file.metadata().getUint32("qwen2.context_length")).contains(131_072);
      assertThat(file.metadata().getUint32("qwen2.attention.head_count")).contains(28);
      assertThat(file.metadata().getUint32("qwen2.attention.head_count_kv")).contains(4);
      assertThat(file.tensorInfos()).hasSize(339);
      assertThat(file.tensorInfos())
          .extracting(tensor -> tensor.type())
          .contains(GgufTensorType.F32, GgufTensorType.Q4_K, GgufTensorType.Q6_K);
    }
  }

  @Test
  void matchesLlamaCppDeepSeekR1QwenTokenizer() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      var tokenizer = GgufTokenizer.fromMetadata(file.metadata());

      assertThat(tokenizer.encode("The quick brown fox"))
          .containsExactly(151646, 785, 3974, 13876, 38835);
      assertThat(tokenizer.encode("I'm testing DeepSeek-R1."))
          .containsExactly(151646, 40, 2776, 7497, 18183, 39350, 10911, 16, 13);
      assertThat(tokenizer.encode("camelCase42HTTP"))
          .containsExactly(151646, 93321, 4207, 19, 17, 9230);
      assertThat(tokenizer.encode("price=$12.50; rate=7%"))
          .containsExactly(151646, 6555, 3186, 16, 17, 13, 20, 15, 26, 4379, 28, 22, 4);
    }
  }

  @Test
  void matchesLlamaCppGreedyCompletionTokens() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen2");
      assertThat(backend.metadata().contextLength()).isEqualTo(131_072);

      int[] promptTokens = backend.tokenizer().encode("The quick brown fox");
      assertThat(promptTokens).containsExactly(151646, 785, 3974, 13876, 38835);
      assertThat(greedyTokens(backend, promptTokens, 4))
          .as(
              "greedy token IDs must match llama.cpp b9960 for the pinned DeepSeek R1 Distill"
                  + " Qwen 7B GGUF")
          .containsExactly(34208, 916, 279, 15678);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledArtifact() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(DEEPSEEK_R1_DISTILL_QWEN_7B_Q4_K_M).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run the DeepSeek R1 Distill Qwen 7B fixture download task before"
                + " this test.",
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
