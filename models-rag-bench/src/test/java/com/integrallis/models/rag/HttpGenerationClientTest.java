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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.net.URI;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class HttpGenerationClientTest {

  @Test
  void ollamaRequestUsesRawDeterministicGeneration() {
    try (HttpGenerationClient client =
        new HttpGenerationClient(
            "ollama", "qwen:test", URI.create("http://localhost:11434"), 2_048, 8, 0)) {
      var body = client.requestBody("prompt", 32);

      assertThat(body.path("raw").asBoolean()).isTrue();
      assertThat(body.path("options").path("temperature").asDouble()).isZero();
      assertThat(body.path("options").path("top_k").asInt()).isEqualTo(1);
      assertThat(body.path("options").path("seed").asInt()).isEqualTo(42);
      assertThat(body.path("options").path("num_predict").asInt()).isEqualTo(32);
    }
  }

  @Test
  void llamaCppRequestDisablesPromptCacheAndUsesSameSampling() {
    try (HttpGenerationClient client =
        new HttpGenerationClient(
            "llama.cpp", "qwen.gguf", URI.create("http://localhost:8080"), 2_048, 8, 0)) {
      var body = client.requestBody("prompt", 32);

      assertThat(body.path("cache_prompt").asBoolean()).isFalse();
      assertThat(body.path("temperature").asDouble()).isZero();
      assertThat(body.path("top_k").asInt()).isEqualTo(1);
      assertThat(body.path("seed").asInt()).isEqualTo(42);
      assertThat(body.path("n_predict").asInt()).isEqualTo(32);
    }
  }

  @Test
  void retriesOneTransportFailureBeforeResponseHeaders() throws Exception {
    AtomicInteger attempts = new AtomicInteger();

    String result =
        HttpGenerationClient.withSingleTransportRetry(
            () -> {
              if (attempts.incrementAndGet() == 1) {
                throw new IOException("stale pooled connection");
              }
              return "response";
            });

    assertThat(result).isEqualTo("response");
    assertThat(attempts).hasValue(2);
  }

  @Test
  void exposesBothFailuresWhenTheTransportRetryFails() {
    AtomicInteger attempts = new AtomicInteger();

    assertThatThrownBy(
            () ->
                HttpGenerationClient.withSingleTransportRetry(
                    () -> {
                      throw new IOException("attempt-" + attempts.incrementAndGet());
                    }))
        .isInstanceOf(IOException.class)
        .hasMessage("attempt-2")
        .satisfies(failure -> assertThat(failure.getSuppressed()).hasSize(1));
  }
}
