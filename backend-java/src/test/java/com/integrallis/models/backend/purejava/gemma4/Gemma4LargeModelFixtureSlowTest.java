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
package com.integrallis.models.backend.purejava.gemma4;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
class Gemma4LargeModelFixtureSlowTest {

  private static final ModelFixtureRequirement GEMMA4_26B_A4B_Q4_K_M =
      ModelFixtureRequirement.of("hf://ggml-org/gemma-4-26B-A4B-it-GGUF")
          .version("[4.0.0,5.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("chat");

  @Test
  void pinnedArtifactContainsTheExactTextDecoderContract() throws Exception {
    ModelFixtureDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      Gemma4Config config = Gemma4Config.fromMetadata(file.metadata());
      Gemma4Weights weights = Gemma4Weights.fromGgufFile(file, config);

      assertThat(file.metadata().getString("general.architecture")).contains("gemma4");
      assertThat(file.tensorInfos()).hasSize(658);
      assertThat(file.tensorInfos())
          .extracting(tensor -> tensor.type())
          .contains(
              GgufTensorType.F32,
              GgufTensorType.Q4_K,
              GgufTensorType.Q5_0,
              GgufTensorType.Q6_K,
              GgufTensorType.Q8_0);
      assertThat(config.embeddingDim()).isEqualTo(2_816);
      assertThat(config.numLayers()).isEqualTo(30);
      assertThat(config.numExperts()).isEqualTo(128);
      assertThat(config.numExpertsUsed()).isEqualTo(8);
      assertThat(config.slidingWindow()).isEqualTo(1_024);
      assertThat(config.contextLength()).isEqualTo(262_144);
      assertThat(config.numKvHeads(0)).isEqualTo(8);
      assertThat(config.numKvHeads(5)).isEqualTo(2);
      assertThat(config.usesSlidingWindow(0)).isTrue();
      assertThat(config.usesSlidingWindow(5)).isFalse();
      assertThat(weights.ropeFrequencyFactors()).hasSize(256);
      assertThat(weights.expertLayout().routedExpertBytes()).isEqualTo(15_130_165_248L);
      assertThat(weights.expertLayout().residentBytes()).isEqualTo(1_650_027_640L);
    }
  }

  @Test
  void tokenizerAndChatEnvelopeMatchLlamaCpp() throws Exception {
    ModelFixtureDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());

      assertThat(tokenizer.encode("Hello")).containsExactly(2, 9259);
      assertThat(tokenizer.encode("2024")).containsExactly(2, 236778, 236771, 236778, 236812);
      assertThat(
              tokenizer.encode(
                  ChatTemplate.GEMMA4.render(
                      List.of(
                          ChatMessage.system("System"),
                          ChatMessage.user("Question"),
                          ChatMessage.assistant("Answer"),
                          ChatMessage.user("Next")))))
          .containsExactly(
              2, 105, 9731, 107, 4521, 106, 107, 105, 2364, 107, 14977, 106, 107, 105, 4368, 107,
              7925, 106, 107, 105, 2364, 107, 9272, 106, 107, 105, 4368, 107, 100, 45518, 107, 101);
    }
  }

  @Test
  void matchesLlamaCppGreedyChatCompletionTokens() {
    ModelFixtureDescriptor descriptor = descriptorWithInstalledArtifact();

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor.localPath().orElseThrow())) {
      ModelPrompt prompt = ChatTemplate.GEMMA4.render(List.of(ChatMessage.user("Hello")));
      int[] promptTokens = backend.tokenizer().encode(prompt);

      backend.reset();
      float[] logits = backend.prefill(promptTokens, 0);
      int[] expected = {9259, 236888, 2088, 740};
      int[] actual = new int[expected.length];
      for (int index = 0; index < expected.length; index++) {
        actual[index] = argmax(logits);
        logits = backend.forward(expected[index], promptTokens.length + index);
      }

      assertThat(actual)
          .as("greedy token IDs must match llama.cpp a582222 for the pinned Gemma 4 GGUF")
          .containsExactly(expected);
    }
  }

  private static ModelFixtureDescriptor descriptorWithInstalledArtifact() {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(GEMMA4_26B_A4B_Q4_K_M).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run :backend-java:downloadGemma426BA4BQ4KMModel first.",
            descriptor.localPath().orElseThrow())
        .isTrue();
    return descriptor;
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
}
