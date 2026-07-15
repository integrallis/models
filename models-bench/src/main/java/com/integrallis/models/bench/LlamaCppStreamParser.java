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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.Objects;

/** Incremental parser for llama.cpp server's streaming completion response. */
final class LlamaCppStreamParser {

  private static final String DATA_PREFIX = "data:";

  private final ObjectMapper mapper;
  private final StringBuilder text = new StringBuilder();
  private double firstTokenMillis = -1;
  private int inputTokens;
  private int outputTokens;
  private double prefillTokensPerSecond;
  private double decodeTokensPerSecond;
  private boolean complete;

  LlamaCppStreamParser(ObjectMapper mapper) {
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
      prefillTokensPerSecond = tokensPerSecond(inputTokens, timings.path("prompt_ms").asDouble(0));
      decodeTokensPerSecond =
          tokensPerSecond(outputTokens, timings.path("predicted_ms").asDouble(0));
    }
  }

  ParsedGeneration result() {
    if (!complete) {
      throw new IllegalStateException("llama.cpp stream did not contain final timings");
    }
    if (outputTokens <= 0 || firstTokenMillis < 0) {
      throw new IllegalStateException("llama.cpp stream did not contain an output token");
    }
    return new ParsedGeneration(
        text.toString(),
        firstTokenMillis,
        inputTokens,
        outputTokens,
        prefillTokensPerSecond,
        decodeTokensPerSecond,
        0);
  }

  private static double tokensPerSecond(int tokens, double durationMillis) {
    return tokens > 0 && durationMillis > 0 ? tokens * 1_000.0 / durationMillis : 0;
  }
}
