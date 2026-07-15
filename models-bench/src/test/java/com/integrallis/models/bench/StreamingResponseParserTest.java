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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

class StreamingResponseParserTest {

  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void parsesOllamaNdjsonTimingAndTokenCounts() throws Exception {
    OllamaStreamParser parser = new OllamaStreamParser(mapper);
    parser.accept("{\"response\":\"Hello\",\"done\":false}", 100);
    parser.accept(
        "{\"response\":\" world\",\"done\":true,\"prompt_eval_count\":12,"
            + "\"prompt_eval_duration\":600000000,\"eval_count\":8,"
            + "\"eval_duration\":400000000,\"load_duration\":900000000}",
        500);

    ParsedGeneration parsed = parser.result();
    assertThat(parsed.text()).isEqualTo("Hello world");
    assertThat(parsed.firstTokenMillis()).isEqualTo(100);
    assertThat(parsed.inputTokens()).isEqualTo(12);
    assertThat(parsed.outputTokens()).isEqualTo(8);
    assertThat(parsed.prefillTokensPerSecond()).isEqualTo(20);
    assertThat(parsed.decodeTokensPerSecond()).isEqualTo(20);
    assertThat(parsed.loadMillis()).isEqualTo(900);
  }

  @Test
  void parsesLlamaCppServerSseAndFinalTimings() throws Exception {
    LlamaCppStreamParser parser = new LlamaCppStreamParser(mapper);
    parser.accept("data: {\"content\":\"Hello\",\"stop\":false}", 80);
    parser.accept(
        "data: {\"content\":\"!\",\"stop\":true,\"timings\":{"
            + "\"prompt_n\":16,\"prompt_ms\":400.0,\"predicted_n\":10,"
            + "\"predicted_ms\":500.0}}",
        600);

    ParsedGeneration parsed = parser.result();
    assertThat(parsed.text()).isEqualTo("Hello!");
    assertThat(parsed.firstTokenMillis()).isEqualTo(80);
    assertThat(parsed.inputTokens()).isEqualTo(16);
    assertThat(parsed.outputTokens()).isEqualTo(10);
    assertThat(parsed.prefillTokensPerSecond()).isEqualTo(40);
    assertThat(parsed.decodeTokensPerSecond()).isEqualTo(20);
  }

  @Test
  void rejectsCompletedStreamsThatProducedNoOutputToken() throws Exception {
    OllamaStreamParser parser = new OllamaStreamParser(mapper);
    parser.accept(
        "{\"response\":\"\",\"done\":true,\"prompt_eval_count\":12,"
            + "\"prompt_eval_duration\":600000000,\"eval_count\":0,"
            + "\"eval_duration\":0}",
        500);

    assertThatThrownBy(parser::result)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("output token");
  }
}
