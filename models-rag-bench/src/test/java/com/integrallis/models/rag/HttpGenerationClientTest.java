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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.URI;
import org.junit.jupiter.api.Test;

class HttpGenerationClientTest {
  @Test
  void reportsPublishedBackendDiagnosticsAndDeterministicControls() {
    try (HttpGenerationClient client =
        new HttpGenerationClient(
            "llama.cpp", "qwen.gguf", URI.create("http://localhost:8080"), 2_048, 8, 0)) {
      assertThat(client.diagnostics().backend()).isEqualTo("llama.cpp");
      assertThat(client.diagnostics().planVersion()).isEqualTo("local-http-v1");
      assertThat(client.generationControls())
          .containsEntry("temperature", "0")
          .containsEntry("promptCache", "false");
    }
  }
}
