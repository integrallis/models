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

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.SpeculativeGenerationOptions;
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("integration")
class Qwen3ModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;

  private static final ModelJarRequirement QWEN3_0_6B_Q4_0 =
      ModelJarRequirement.forSource("hf://ggml-org/Qwen3-0.6B-GGUF")
          .versionRange("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation")
          .build();

  private static final ModelJarRequirement QWEN3_1_7B_Q8_0 =
      ModelJarRequirement.forSource("hf://Qwen/Qwen3-1.7B-GGUF")
          .versionRange("[3.0.0,4.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("text-generation")
          .build();

  @Test
  void loadsQwen306BQ40ThroughModelJars() {
    assertLoadsQwen3(QWEN3_0_6B_Q4_0);
  }

  @Test
  void loadsQwen317BQ80ThroughModelJars() {
    assertLoadsQwen3(QWEN3_1_7B_Q8_0);
  }

  @Test
  void matchesLlamaCppGreedyTokensForQwen306BQ40() {
    assertGreedyReference(
        QWEN3_0_6B_Q4_0,
        "The quick brown fox",
        new int[] {785, 3974, 13876, 38835},
        new int[] {34208, 916, 279, 15678});
  }

  @Test
  void ngramSpeculationMatchesSequentialQwen306BQ40Generation() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN3_0_6B_Q4_0).orElseThrow();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      SamplingOptions sampling =
          SamplingOptions.builder().temperature(0.0f).repetitionPenalty(1.0f).maxTokens(16).build();
      String prompt =
          "Continue_only_the_exact_sequence_without_explanation:"
              + "_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_";
      String sequential = new GenerationLoop(backend).generate(prompt, sampling);
      GenerationLoop speculative =
          new GenerationLoop(backend, SpeculativeGenerationOptions.builder().build());

      String accelerated = speculative.generate(prompt, sampling);

      assertThat(accelerated).isEqualTo(sequential);
      assertThat(speculative.lastSpeculativeMetrics().active()).isTrue();
      assertThat(speculative.lastSpeculativeMetrics().acceptedTokens()).isPositive();
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static void assertLoadsQwen3(ModelJarRequirement requirement) {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(requirement).orElseThrow();

    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run :models-backend-purejava:integrationTest or the fixture"
                + " download task before running this test.",
            descriptor.localPath().orElseThrow())
        .isTrue();

    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.name()).isEqualTo("pure-java");
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen3");
      assertThat(backend.metadata().contextLength()).isGreaterThanOrEqualTo(40_960);

      int[] tokens = backend.tokenizer().encode("The quick brown fox");
      assertThat(tokens).isNotEmpty();

      float[] logits = backend.forward(tokens[0], 0);
      assertThat(logits).hasSize(backend.metadata().vocabSize());
      for (int i = 0; i < logits.length; i++) {
        assertThat(logits[i]).as("Logit at index %d should be finite", i).isFinite();
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }

  private static void assertGreedyReference(
      ModelJarRequirement requirement,
      String prompt,
      int[] expectedPromptTokens,
      int[] expectedGeneratedTokens) {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(requirement).orElseThrow();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      int[] promptTokens = backend.tokenizer().encode(prompt);
      assertThat(promptTokens).containsExactly(expectedPromptTokens);
      assertThat(greedyTokens(backend, promptTokens, expectedGeneratedTokens.length))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned GGUF")
          .containsExactly(expectedGeneratedTokens);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static int[] greedyTokens(PureJavaBackend backend, int[] promptTokens, int count) {
    backend.reset();
    float[] logits = backend.prefill(promptTokens, 0);
    int position = promptTokens.length;

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
}
