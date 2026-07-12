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
class SmolLm2ModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;

  private static final ModelJarRequirement SMOLLM2_360M_Q8_0 =
      ModelJarRequirement.forSource("hf://HuggingFaceTB/SmolLM2-360M-Instruct-GGUF")
          .versionRange("[2.0.0,3.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("chat")
          .build();

  @Test
  void loadsSmolLm2360MQ80ThroughModelJars() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(SMOLLM2_360M_Q8_0).orElseThrow();

    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run :models-backend-purejava:integrationTest or the"
                + " fixture download task before running this test.",
            descriptor.localPath().orElseThrow())
        .isTrue();

    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.name()).isEqualTo("pure-java");
      assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
      assertThat(backend.metadata().contextLength()).isEqualTo(8_192);

      int[] tokens = backend.tokenizer().encode("Hello from Java");
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
}
