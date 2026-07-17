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

final class LlamaCppGenerationParser {
  private static final String DATA_PREFIX = "data:";

  private final ObjectMapper mapper;
  private final StringBuilder text = new StringBuilder();
  private double firstTokenMillis = -1;
  private int inputTokens;
  private int outputTokens;
  private double prefillTokensPerSecond;
  private boolean complete;

  LlamaCppGenerationParser(ObjectMapper mapper) {
    this.mapper = Objects.requireNonNull(mapper, "mapper");
  }

  void accept(String line, double elapsedMillis) throws IOException {
    if (line == null || line.isBlank()) {
      return;
    }
    String payload =
        line.startsWith(DATA_PREFIX) ? line.substring(DATA_PREFIX.length()).trim() : line;
    if (payload.isEmpty() || "[DONE]".equals(payload)) {
      return;
    }
    JsonNode event = mapper.readTree(payload);
    String content = event.path("content").asText("");
    if (!content.isEmpty() && firstTokenMillis < 0) {
      firstTokenMillis = elapsedMillis;
    }
    text.append(content);
    JsonNode timings = event.path("timings");
    if (event.path("stop").asBoolean(false) && !timings.isMissingNode()) {
      complete = true;
      inputTokens = timings.path("prompt_n").asInt(0);
      outputTokens = timings.path("predicted_n").asInt(0);
      double promptMillis = timings.path("prompt_ms").asDouble(0);
      prefillTokensPerSecond =
          inputTokens > 0 && promptMillis > 0 ? inputTokens * 1_000.0 / promptMillis : 0;
    }
  }

  ParsedStreamingGeneration result() {
    if (!complete) {
      throw new IllegalStateException("llama.cpp stream did not contain final timings");
    }
    if (outputTokens < 1 || firstTokenMillis < 0) {
      throw new IllegalStateException("llama.cpp stream did not contain an output token");
    }
    return new ParsedStreamingGeneration(
        text.toString(), firstTokenMillis, inputTokens, outputTokens, prefillTokensPerSecond, 0);
  }
}
