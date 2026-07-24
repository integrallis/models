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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.Objects;

/** Parses Anthropic Messages streaming events. */
final class AnthropicGenerationParser {
  private final ObjectMapper mapper;
  private final StringBuilder text = new StringBuilder();
  private double firstTextTokenMillis = -1;
  private int uncachedInputTokens = -1;
  private int cacheReadInputTokens;
  private int cacheWriteInputTokens;
  private int outputTokens = -1;

  AnthropicGenerationParser(ObjectMapper mapper) {
    this.mapper = Objects.requireNonNull(mapper, "mapper");
  }

  void accept(String line, double elapsedMillis) throws IOException {
    if (line == null || line.isBlank() || line.startsWith("event:")) {
      return;
    }
    String data = line.startsWith("data:") ? line.substring("data:".length()).trim() : line.trim();
    if (data.isEmpty()) {
      return;
    }

    JsonNode event = mapper.readTree(data);
    if ("error".equals(event.path("type").asText())) {
      throw new IOException("Anthropic stream error: " + event.path("error"));
    }

    JsonNode delta = event.path("delta");
    if ("text_delta".equals(delta.path("type").asText())) {
      appendText(delta.path("text"), elapsedMillis);
    }
    JsonNode contentBlock = event.path("content_block");
    if ("text".equals(contentBlock.path("type").asText())) {
      appendText(contentBlock.path("text"), elapsedMillis);
    }

    JsonNode usage =
        "message_start".equals(event.path("type").asText())
            ? event.path("message").path("usage")
            : event.path("usage");
    updateUsage(usage);
  }

  ParsedHostedGeneration result() {
    if (text.isEmpty() || firstTextTokenMillis < 0) {
      throw new IllegalStateException("Anthropic stream did not contain a text delta");
    }
    if (uncachedInputTokens < 0 || outputTokens < 1) {
      throw new IllegalStateException("Anthropic stream did not contain terminal token usage");
    }
    return new ParsedHostedGeneration(
        text.toString(),
        firstTextTokenMillis,
        uncachedInputTokens + cacheReadInputTokens + cacheWriteInputTokens,
        cacheReadInputTokens,
        cacheWriteInputTokens,
        outputTokens);
  }

  private void appendText(JsonNode value, double elapsedMillis) {
    if (value.isTextual() && !value.textValue().isEmpty()) {
      if (firstTextTokenMillis < 0) {
        firstTextTokenMillis = elapsedMillis;
      }
      text.append(value.textValue());
    }
  }

  private void updateUsage(JsonNode usage) {
    if (!usage.isObject()) {
      return;
    }
    if (usage.path("input_tokens").isIntegralNumber()) {
      uncachedInputTokens = usage.path("input_tokens").intValue();
    }
    if (usage.path("cache_read_input_tokens").isIntegralNumber()) {
      cacheReadInputTokens = usage.path("cache_read_input_tokens").intValue();
    }
    if (usage.path("cache_creation_input_tokens").isIntegralNumber()) {
      cacheWriteInputTokens = usage.path("cache_creation_input_tokens").intValue();
    }
    if (usage.path("output_tokens").isIntegralNumber()) {
      outputTokens = usage.path("output_tokens").intValue();
    }
  }
}
