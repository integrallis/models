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

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

class StreamingGenerationParserTest {

  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void parsesOllamaNdjsonAndNanosecondTimingMetadata() throws Exception {
    OllamaGenerationParser parser = new OllamaGenerationParser(mapper);

    parser.accept("{\"response\":\"North\",\"done\":false}", 12);
    parser.accept("{\"response\":\"star\",\"done\":false}", 18);
    parser.accept(
        "{\"response\":\"\",\"done\":true,\"prompt_eval_count\":100,"
            + "\"eval_count\":2,\"prompt_eval_duration\":500000000,"
            + "\"load_duration\":7000000}",
        25);

    ParsedStreamingGeneration result = parser.result();
    assertThat(result.text()).isEqualTo("Northstar");
    assertThat(result.firstTokenMillis()).isEqualTo(12);
    assertThat(result.inputTokens()).isEqualTo(100);
    assertThat(result.outputTokens()).isEqualTo(2);
    assertThat(result.prefillTokensPerSecond()).isEqualTo(200);
    assertThat(result.loadMillis()).isEqualTo(7);
  }

  @Test
  void parsesLlamaCppSseAndMillisecondTimingMetadata() throws Exception {
    LlamaCppGenerationParser parser = new LlamaCppGenerationParser(mapper);

    parser.accept("data: {\"content\":\"North\",\"stop\":false}", 10);
    parser.accept("data: {\"content\":\"star\",\"stop\":false}", 16);
    parser.accept(
        "data: {\"content\":\"\",\"stop\":true,\"timings\":{"
            + "\"prompt_n\":100,\"predicted_n\":2,\"prompt_ms\":400.0}}",
        22);

    ParsedStreamingGeneration result = parser.result();
    assertThat(result.text()).isEqualTo("Northstar");
    assertThat(result.firstTokenMillis()).isEqualTo(10);
    assertThat(result.inputTokens()).isEqualTo(100);
    assertThat(result.outputTokens()).isEqualTo(2);
    assertThat(result.prefillTokensPerSecond()).isEqualTo(250);
  }
}
