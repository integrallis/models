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

class HostedGenerationClientTest {

  @Test
  void createsAnOpenAiStreamingRequestWithoutExposingTheKeyInDiagnostics() {
    try (HostedGenerationClient client =
        new HostedGenerationClient(
            "openai",
            "gpt-5.4-nano-2026-03-17",
            URI.create("https://api.openai.com/v1"),
            "secret-value")) {
      var body = client.requestBody("canonical prompt", 64);

      assertThat(body.path("model").asText()).isEqualTo("gpt-5.4-nano-2026-03-17");
      assertThat(body.path("stream").asBoolean()).isTrue();
      assertThat(body.path("stream_options").path("include_usage").asBoolean()).isTrue();
      assertThat(body.path("max_completion_tokens").asInt()).isEqualTo(64);
      assertThat(body.path("reasoning_effort").asText()).isEqualTo("none");
      assertThat(body.path("messages").path(0).path("role").asText()).isEqualTo("user");
      assertThat(body.path("messages").path(0).path("content").asText())
          .isEqualTo("canonical prompt");
      assertThat(client.diagnostics().environment().values()).doesNotContain("secret-value");
    }
  }

  @Test
  void createsADeepSeekNonThinkingStreamingRequest() {
    try (HostedGenerationClient client =
        new HostedGenerationClient(
            "deepseek",
            "deepseek-v4-flash",
            URI.create("https://api.deepseek.com"),
            "secret-value")) {
      var body = client.requestBody("canonical prompt", 64);

      assertThat(body.path("max_tokens").asInt()).isEqualTo(64);
      assertThat(body.path("thinking").path("type").asText()).isEqualTo("disabled");
      assertThat(body.path("temperature").asDouble()).isZero();
      assertThat(body.has("top_p")).isFalse();
      assertThat(body.path("stream_options").path("include_usage").asBoolean()).isTrue();
    }
  }

  @Test
  void createsAnAnthropicStreamingMessagesRequest() {
    try (HostedGenerationClient client =
        new HostedGenerationClient(
            "anthropic",
            "claude-haiku-4-5-20251001",
            URI.create("https://api.anthropic.com/v1"),
            "secret-value")) {
      var body = client.requestBody("canonical prompt", 64);

      assertThat(body.path("max_tokens").asInt()).isEqualTo(64);
      assertThat(body.path("temperature").asDouble()).isZero();
      assertThat(body.has("top_p")).isFalse();
      assertThat(body.path("stream").asBoolean()).isTrue();
      assertThat(body.path("messages").path(0).path("content").asText())
          .isEqualTo("canonical prompt");
    }
  }
}
