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

/** Parses OpenAI-compatible streaming chat-completion events. */
final class OpenAiChatGenerationParser {
  private final ObjectMapper mapper;
  private final StringBuilder text = new StringBuilder();
  private double firstTextTokenMillis = -1;
  private int inputTokens = -1;
  private int cacheReadInputTokens;
  private int outputTokens = -1;

  OpenAiChatGenerationParser(ObjectMapper mapper) {
    this.mapper = Objects.requireNonNull(mapper, "mapper");
  }

  void accept(String line, double elapsedMillis) throws IOException {
    if (line == null || line.isBlank() || line.startsWith("event:")) {
      return;
    }
    String data = line.startsWith("data:") ? line.substring("data:".length()).trim() : line.trim();
    if (data.isEmpty() || "[DONE]".equals(data)) {
      return;
    }

    JsonNode event = mapper.readTree(data);
    JsonNode error = event.path("error");
    if (!error.isMissingNode() && !error.isNull()) {
      throw new IOException("hosted API stream error: " + error);
    }

    JsonNode content = event.path("choices").path(0).path("delta").path("content");
    if (content.isTextual() && !content.textValue().isEmpty()) {
      if (firstTextTokenMillis < 0) {
        firstTextTokenMillis = elapsedMillis;
      }
      text.append(content.textValue());
    }

    JsonNode usage = event.path("usage");
    if (usage.isObject()) {
      inputTokens = usage.path("prompt_tokens").asInt(-1);
      outputTokens = usage.path("completion_tokens").asInt(-1);
      JsonNode deepSeekCache = usage.path("prompt_cache_hit_tokens");
      cacheReadInputTokens =
          deepSeekCache.isIntegralNumber()
              ? deepSeekCache.intValue()
              : usage.path("prompt_tokens_details").path("cached_tokens").asInt(0);
    }
  }

  ParsedHostedGeneration result() {
    if (text.isEmpty() || firstTextTokenMillis < 0) {
      throw new IllegalStateException("hosted API stream did not contain a text delta");
    }
    if (inputTokens < 0 || outputTokens < 1) {
      throw new IllegalStateException("hosted API stream did not contain terminal token usage");
    }
    return new ParsedHostedGeneration(
        text.toString(), firstTextTokenMillis, inputTokens, cacheReadInputTokens, 0, outputTokens);
  }
}
