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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.node.ObjectNode;
import java.net.URI;
import org.junit.jupiter.api.Test;

class HttpBenchmarkTargetTest {

  @Test
  void ollamaUsesRawDeterministicPromptAndExplicitResourceControls() {
    try (HttpBenchmarkTarget target =
        new HttpBenchmarkTarget(
            "ollama", "fixture:gguf", URI.create("http://localhost:11434"), 4096, 8, 0, 0)) {
      ObjectNode body = target.requestBody("prompt", 64);

      assertThat(body.path("raw").asBoolean()).isTrue();
      assertThat(body.path("options").path("num_ctx").asInt()).isEqualTo(4096);
      assertThat(body.path("options").path("num_thread").asInt()).isEqualTo(8);
      assertThat(body.path("options").path("repeat_penalty").asDouble()).isEqualTo(1.0);
    }
  }

  @Test
  void llamaCppUsesMatchingDeterministicControls() {
    try (HttpBenchmarkTarget target =
        new HttpBenchmarkTarget(
            "llama.cpp", "fixture", URI.create("http://localhost:8080"), 4096, 8, 0, 0)) {
      ObjectNode body = target.requestBody("prompt", 64);

      assertThat(body.path("n_threads").asInt()).isEqualTo(8);
      assertThat(body.path("repeat_penalty").asDouble()).isEqualTo(1.0);
      assertThat(body.path("cache_prompt").asBoolean()).isFalse();
    }
  }
}
