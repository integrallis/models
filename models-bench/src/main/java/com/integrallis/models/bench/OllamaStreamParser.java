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

/** Incremental parser for Ollama's streaming NDJSON generation response. */
final class OllamaStreamParser {

  private final ObjectMapper mapper;
  private final StringBuilder text = new StringBuilder();
  private double firstTokenMillis = -1;
  private int inputTokens;
  private int outputTokens;
  private double prefillTokensPerSecond;
  private double decodeTokensPerSecond;
  private double loadMillis;
  private boolean complete;

  OllamaStreamParser(ObjectMapper mapper) {
    this.mapper = Objects.requireNonNull(mapper, "mapper");
  }

  void accept(String line, double elapsedMillis) throws IOException {
    if (line == null || line.isBlank()) {
      return;
    }
    JsonNode event = mapper.readTree(line);
    String content = event.path("response").asText("");
    if (!content.isEmpty() && firstTokenMillis < 0) {
      firstTokenMillis = elapsedMillis;
    }
    text.append(content);
    if (event.path("done").asBoolean(false)) {
      complete = true;
      inputTokens = event.path("prompt_eval_count").asInt(0);
      outputTokens = event.path("eval_count").asInt(0);
      prefillTokensPerSecond =
          tokensPerSecond(inputTokens, event.path("prompt_eval_duration").asLong(0));
      decodeTokensPerSecond = tokensPerSecond(outputTokens, event.path("eval_duration").asLong(0));
      loadMillis = event.path("load_duration").asLong(0) / 1_000_000.0;
    }
  }

  ParsedGeneration result() {
    if (!complete) {
      throw new IllegalStateException("Ollama stream did not contain a final event");
    }
    if (outputTokens <= 0 || firstTokenMillis < 0) {
      throw new IllegalStateException("Ollama stream did not contain an output token");
    }
    return new ParsedGeneration(
        text.toString(),
        firstTokenMillis,
        inputTokens,
        outputTokens,
        prefillTokensPerSecond,
        decodeTokensPerSecond,
        loadMillis);
  }

  private static double tokensPerSecond(int tokens, long durationNanos) {
    return tokens > 0 && durationNanos > 0 ? tokens * 1_000_000_000.0 / durationNanos : 0;
  }
}
