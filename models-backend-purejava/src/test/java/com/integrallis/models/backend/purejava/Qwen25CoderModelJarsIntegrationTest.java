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

import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("integration")
class Qwen25CoderModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;

  private static final ModelJarRequirement QWEN25_CODER_0_5B_Q4_0 =
      ModelJarRequirement.forSource("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
          .versionRange("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion")
          .build();

  private static final ModelJarRequirement QWEN25_CODER_0_5B_Q8_0 =
      ModelJarRequirement.forSource("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
          .versionRange("[2.5.0,3.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("code-completion")
          .build();

  private static final ModelJarRequirement QWEN25_CODER_1_5B_Q4_0 =
      ModelJarRequirement.forSource("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF")
          .versionRange("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion")
          .build();

  private static final ModelJarRequirement QWEN25_CODER_1_5B_Q8_0 =
      ModelJarRequirement.forSource("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF")
          .versionRange("[2.5.0,3.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("code-completion")
          .build();

  private static final ModelJarRequirement QWEN25_CODER_3B_Q4_0 =
      ModelJarRequirement.forSource("hf://Qwen/Qwen2.5-Coder-3B-Instruct-GGUF")
          .versionRange("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion")
          .build();

  @Test
  void loadsQwen25Coder05BQ40ThroughModelJars() {
    assertLoadsQwen25Coder(QWEN25_CODER_0_5B_Q4_0);
  }

  @Test
  void loadsQwen25Coder05BQ80ThroughModelJars() {
    assertLoadsQwen25Coder(QWEN25_CODER_0_5B_Q8_0);
  }

  @Test
  void loadsQwen25Coder15BQ40ThroughModelJars() {
    assertLoadsQwen25Coder(QWEN25_CODER_1_5B_Q4_0);
  }

  @Test
  void loadsQwen25Coder15BQ80ThroughModelJars() {
    assertLoadsQwen25Coder(QWEN25_CODER_1_5B_Q8_0);
  }

  @Test
  void loadsQwen25Coder3BQ40ThroughModelJars() {
    assertLoadsQwen25Coder(QWEN25_CODER_3B_Q4_0);
  }

  @Test
  void matchesLlamaCppGreedyTokensForQwen25Coder05BQ40() {
    assertGreedyReference(
        QWEN25_CODER_0_5B_Q4_0,
        "public static void main",
        new int[] {888, 1099, 737, 1887},
        new int[] {2242, 1294, 2827, 8});
  }

  private static void assertLoadsQwen25Coder(ModelJarRequirement requirement) {
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
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen2");
      assertThat(backend.metadata().contextLength()).isGreaterThanOrEqualTo(32_768);

      int[] tokens = backend.tokenizer().encode("public static void main");
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
}
