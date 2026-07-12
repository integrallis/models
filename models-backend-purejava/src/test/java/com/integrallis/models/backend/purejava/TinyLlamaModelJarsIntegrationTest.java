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
class TinyLlamaModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;

  private static final ModelJarRequirement TINYLLAMA_1_1B_CHAT_V1_0_Q4_0 =
      ModelJarRequirement.forSource("hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
          .versionRange("[1.0.0,2.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("chat")
          .build();

  @Test
  void loadsTinyLlamaThroughModelJars() {
    ModelJarDescriptor descriptor = descriptorWithInstalledModel();
    String previous = useIntegrationContextLength();

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.name()).isEqualTo("pure-java");
      assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
      assertThat(backend.metadata().contextLength()).isEqualTo(2_048);

      int[] tokens = backend.tokenizer().encode("The quick brown fox");
      assertThat(tokens).containsExactly(1, 450, 4996, 17354, 1701, 29916);

      float[] logits = backend.forward(tokens[0], 0);
      assertThat(logits).hasSize(backend.metadata().vocabSize());
      for (int index = 0; index < logits.length; index++) {
        assertThat(logits[index]).as("Logit at index %d should be finite", index).isFinite();
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  @Test
  void matchesLlamaCppGreedyTokens() {
    ModelJarDescriptor descriptor = descriptorWithInstalledModel();
    String previous = useIntegrationContextLength();

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      int[] promptTokens = backend.tokenizer().encode("Hello from Java");
      assertThat(promptTokens).containsExactly(1, 15043, 515, 3355);
      assertThat(greedyTokens(backend, promptTokens, 4))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned GGUF")
          .containsExactly(29991, 13, 13, 8404);
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledModel() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(TINYLLAMA_1_1B_CHAT_V1_0_Q4_0).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run :models-backend-purejava:integrationTest or the fixture"
                + " download task before running this test.",
            descriptor.localPath().orElseThrow())
        .isTrue();
    return descriptor;
  }

  private static String useIntegrationContextLength() {
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));
    return previous;
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
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
