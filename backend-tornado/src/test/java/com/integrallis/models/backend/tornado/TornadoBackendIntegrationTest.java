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
package com.integrallis.models.backend.tornado;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class TornadoBackendIntegrationTest {

  @Test
  void loadsAndRunsThePinnedQwenModelThroughAutomaticSelectionWhenProvided() {
    String configured = System.getProperty("models.fixtures.qwen306BQ40", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.qwen306BQ40=<model.gguf>");
    Path model = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isRegularFile(model), "Qwen fixture is not installed");
    boolean required = Boolean.getBoolean("models.accelerator.required");
    TornadoBackendOptions options = new TornadoBackendOptions(true, true, required, 32);

    try (TornadoBackendRuntime runtime = TornadoBackend.open(model, options)) {
      int token = runtime.backend().tokenizer().bosToken();
      if (token < 0 || token >= runtime.backend().metadata().vocabSize()) {
        token = 0;
      }
      float[] logits = runtime.backend().prefill(new int[] {token}, 0);

      assertThat(logits).hasSize(runtime.backend().metadata().vocabSize());
      for (float logit : logits) {
        assertThat(Float.isFinite(logit)).isTrue();
      }
      if (required) {
        assertThat(runtime.status().accelerated()).isTrue();
        assertThat(runtime.status().readinessTime()).isPositive();
      }
      String expected = System.getProperty("models.accelerator.expected", "");
      if (!expected.isBlank()) {
        assertThat(runtime.status().accelerated()).isEqualTo(Boolean.parseBoolean(expected));
      }
    }
  }
}
