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

class HostedStreamingGenerationParserTest {

  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void parsesOpenAiTextTimingAndCachedUsage() throws Exception {
    OpenAiChatGenerationParser parser = new OpenAiChatGenerationParser(mapper);

    parser.accept("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}", 8);
    parser.accept("data: {\"choices\":[{\"delta\":{\"content\":\"North\"}}]}", 12);
    parser.accept("data: {\"choices\":[{\"delta\":{\"content\":\"star\"}}]}", 18);
    parser.accept(
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,"
            + "\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":25}}}",
        22);
    parser.accept("data: [DONE]", 24);

    ParsedHostedGeneration result = parser.result();
    assertThat(result.text()).isEqualTo("Northstar");
    assertThat(result.firstTextTokenMillis()).isEqualTo(12);
    assertThat(result.inputTokens()).isEqualTo(100);
    assertThat(result.cacheReadInputTokens()).isEqualTo(25);
    assertThat(result.cacheWriteInputTokens()).isZero();
    assertThat(result.outputTokens()).isEqualTo(2);
  }

  @Test
  void parsesDeepSeekCacheUsageFields() throws Exception {
    OpenAiChatGenerationParser parser = new OpenAiChatGenerationParser(mapper);

    parser.accept("data: {\"choices\":[{\"delta\":{\"content\":\"Answer\"}}]}", 15);
    parser.accept(
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":120,"
            + "\"prompt_cache_hit_tokens\":70,\"prompt_cache_miss_tokens\":50,"
            + "\"completion_tokens\":4}}",
        30);

    ParsedHostedGeneration result = parser.result();
    assertThat(result.inputTokens()).isEqualTo(120);
    assertThat(result.cacheReadInputTokens()).isEqualTo(70);
    assertThat(result.outputTokens()).isEqualTo(4);
  }

  @Test
  void parsesAnthropicEventFlowAndCumulativeUsage() throws Exception {
    AnthropicGenerationParser parser = new AnthropicGenerationParser(mapper);

    parser.accept(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{"
            + "\"input_tokens\":80,\"cache_creation_input_tokens\":10,"
            + "\"cache_read_input_tokens\":20,\"output_tokens\":1}}}",
        9);
    parser.accept(
        "data: {\"type\":\"content_block_delta\",\"delta\":{"
            + "\"type\":\"text_delta\",\"text\":\"North\"}}",
        13);
    parser.accept(
        "data: {\"type\":\"content_block_delta\",\"delta\":{"
            + "\"type\":\"text_delta\",\"text\":\"star\"}}",
        19);
    parser.accept("data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":3}}", 25);
    parser.accept("data: {\"type\":\"message_stop\"}", 27);

    ParsedHostedGeneration result = parser.result();
    assertThat(result.text()).isEqualTo("Northstar");
    assertThat(result.firstTextTokenMillis()).isEqualTo(13);
    assertThat(result.inputTokens()).isEqualTo(110);
    assertThat(result.cacheReadInputTokens()).isEqualTo(20);
    assertThat(result.cacheWriteInputTokens()).isEqualTo(10);
    assertThat(result.outputTokens()).isEqualTo(3);
  }
}
