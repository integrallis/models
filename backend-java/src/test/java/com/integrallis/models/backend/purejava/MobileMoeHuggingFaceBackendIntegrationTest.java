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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-checkpoint load gate for the Java-native MobileMoE QAT path. */
@Tag("integration")
class MobileMoeHuggingFaceBackendIntegrationTest {

  @Test
  void loadsTheOfficialQatDirectoryThroughThePublicBackend() {
    Path directory = fixtureDirectory();
    assumeTrue(Files.isRegularFile(directory.resolve("model.safetensors")));
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "16");
    try (PureJavaBackend backend = PureJavaBackend.load(directory)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("mobilemoe");
      assertThat(backend.metadata().vocabSize()).isEqualTo(128_256);
      assertThat(backend.contextCapacity()).isEqualTo(16);
      assertThat(backend.tokenizer().encode("Hello, MobileMoE!"))
          .containsExactly(128000, 9906, 11, 13716, 26694, 36, 0);

      float[] logits = backend.prefill(new int[] {128000, 9906}, 0);
      assertThat(logits).hasSize(128_256);
      assertThat(argmax(logits)).isEqualTo(11);
      assertThat(backend.diagnostics().environment())
          .containsEntry("artifact-format", "safetensors")
          .containsEntry("weight-encoding", "packed-int4-g32")
          .containsEntry("runtime-weight-layout", "q8")
          .containsEntry("architecture-prefill-batch-size", "16");
      assertThat(backend.diagnostics().optimizations())
          .anySatisfy(
              decision -> {
                assertThat(decision.id()).isEqualTo("mobilemoe-runtime-weight-layout");
                assertThat(decision.status().name()).isEqualTo("ENABLED");
                assertThat(decision.settings()).containsEntry("runtime-layout", "q8");
              })
          .anySatisfy(
              decision -> {
                assertThat(decision.id()).isEqualTo("batched-prefill");
                assertThat(decision.status().name()).isEqualTo("ENABLED");
                assertThat(decision.settings()).containsEntry("batch-size", "16");
              });
    } finally {
      if (previous == null) {
        System.clearProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      } else {
        System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }
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

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.mobileMoeQatDirectory", "");
    return configured.isBlank() ? Path.of("missing-mobilemoe-qat-fixture") : Path.of(configured);
  }
}
