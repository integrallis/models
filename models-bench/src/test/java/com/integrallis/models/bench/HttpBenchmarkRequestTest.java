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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

class HttpBenchmarkRequestTest {

  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void makesOllamaUseTheRawPromptAndExplicitDeterministicControls() {
    ObjectNode body =
        HttpBenchmarkTarget.requestBody(
            mapper, "ollama", "fixture:latest", "same prompt", 64, 2_048, 3);

    assertThat(body.path("raw").asBoolean()).isTrue();
    assertThat(body.path("options").path("num_ctx").asInt()).isEqualTo(2_048);
    assertThat(body.path("options").path("num_thread").asInt()).isEqualTo(3);
    assertThat(body.path("options").path("repeat_penalty").asDouble()).isEqualTo(1.0);
    assertThat(body.path("options").path("top_k").asInt()).isEqualTo(1);
    assertThat(body.path("options").path("seed").asLong()).isEqualTo(42);
  }

  @Test
  void disablesLlamaCppPromptCachingAndSamplingVariation() {
    ObjectNode body =
        HttpBenchmarkTarget.requestBody(
            mapper, "llama.cpp", "fixture", "same prompt", 64, 2_048, 1);

    assertThat(body.path("cache_prompt").asBoolean()).isFalse();
    assertThat(body.path("repeat_penalty").asDouble()).isEqualTo(1.0);
    assertThat(body.path("top_k").asInt()).isEqualTo(1);
    assertThat(body.path("seed").asLong()).isEqualTo(42);
  }
}
