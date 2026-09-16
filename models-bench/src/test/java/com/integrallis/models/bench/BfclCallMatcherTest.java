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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import java.util.List;
import org.junit.jupiter.api.Test;

class BfclCallMatcherTest {
  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void parsesOnlyCompleteStrictToolCallBlocks() {
    assertThat(
            BfclCallMatcher.parseStrictCompletion(
                    mapper,
                    "<tool_call>\n{\"name\":\"weather\",\"arguments\":{\"zip\":\"88252\"}}\n</tool_call>\n"
                        + "<tool_call>{\"name\":\"clock\",\"arguments\":{}}</tool_call>")
                .orElseThrow())
        .extracting(ToolCall::name)
        .containsExactly("weather", "clock");
    assertThat(
            BfclCallMatcher.parseStrictCompletion(mapper, "<tool_call>[]</tool_call>")
                .orElseThrow())
        .isEmpty();
  }

  @Test
  void rejectsProseTrailingContentAndInvalidCallShapes() {
    assertThat(
            List.of(
                "I'll call it. <tool_call>{}</tool_call>",
                "<tool_call>{\"name\":\"weather\"}</tool_call> trailing",
                "<tool_call>{\"name\":\"weather\",\"arguments\":[]}</tool_call>"))
        .allSatisfy(
            output -> assertThat(BfclCallMatcher.parseStrictCompletion(mapper, output)).isEmpty());
  }

  @Test
  void acceptsOnlyDeclaredAlternativesAndOptionalEmptyArguments() throws Exception {
    var expected =
        mapper.readTree(
            """
            [{"weather":{"zipcode":["88252"],"units":["","fahrenheit"]}}]
            """);

    assertThat(
            BfclCallMatcher.exactCallsMatch(
                mapper,
                List.of(ToolCall.of(0, "weather", "{\"zipcode\":\"88252\"}")),
                expected,
                List.of()))
        .isTrue();
    assertThat(
            BfclCallMatcher.exactCallsMatch(
                mapper,
                List.of(ToolCall.of(0, "weather", "{\"zipcode\":\"10001\"}")),
                expected,
                List.of()))
        .isFalse();
    assertThat(
            BfclCallMatcher.exactCallsMatch(
                mapper,
                List.of(ToolCall.of(0, "weather", "{\"zipcode\":\"88252\",\"extra\":true}")),
                expected,
                List.of()))
        .isFalse();
  }

  @Test
  void treatsParallelCallsAsAnUnorderedMultiset() throws Exception {
    var expected = mapper.readTree("[{\"first\":{\"x\":[1]}},{\"second\":{\"y\":[2]}}]");
    var actual =
        List.of(ToolCall.of(0, "second", "{\"y\":2}"), ToolCall.of(1, "first", "{\"x\":1}"));

    assertThat(BfclCallMatcher.exactCallsMatch(mapper, actual, expected, List.of())).isTrue();
  }

  @Test
  void appliesThePinnedBfclStringNormalization() throws Exception {
    var expected = mapper.readTree("[{\"weather\":{\"city\":[\"New York, NY\"]}}]");
    var tools =
        List.of(
            new ToolSpec(
                "weather",
                "Weather",
                """
                {"type":"object","properties":{"city":{"type":"string"}}}
                """));

    assertThat(
            BfclCallMatcher.exactCallsMatch(
                mapper,
                List.of(ToolCall.of(0, "weather", "{\"city\":\"new-york ny\"}")),
                expected,
                tools))
        .isTrue();
  }

  @Test
  void recursivelyChecksObjectAlternativesUsingTheirSchema() throws Exception {
    var expected =
        mapper.readTree(
            """
            [{"update_user_info":{"user_id":[43523],"update_info":[{"name":["John Doe"],"email":["johndoe@email.com"]}],"database":["CustomerInfo",""]}}]
            """);
    var tools =
        List.of(
            new ToolSpec(
                "update_user_info",
                "Update",
                """
                {"type":"object","properties":{"user_id":{"type":"integer"},"update_info":{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"}}},"database":{"type":"string"}}}
                """));

    assertThat(
            BfclCallMatcher.exactCallsMatch(
                mapper,
                List.of(
                    ToolCall.of(
                        0,
                        "update_user_info",
                        "{\"user_id\":43523,\"update_info\":{\"name\":\"john-doe\",\"email\":\"johndoe@email.com\"}}")),
                expected,
                tools))
        .isTrue();
    assertThat(
            BfclCallMatcher.exactCallsMatch(
                mapper,
                List.of(
                    ToolCall.of(
                        0,
                        "update_user_info",
                        "{\"user_id\":43523,\"update_info\":{\"name\":\"john-doe\",\"email\":\"wrong@example.com\"}}")),
                expected,
                tools))
        .isFalse();
  }

  @Test
  void recursivelyChecksObjectArraysUsingTheirItemSchema() throws Exception {
    var expected =
        mapper.readTree(
            """
            [{"schedule":{"stops":[[{"name":["Metro Parkway"],"minutes":[8]}]]}}]
            """);
    var tools =
        List.of(
            new ToolSpec(
                "schedule",
                "Schedule",
                """
                {"type":"object","properties":{"stops":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"minutes":{"type":"integer"}},"required":["name","minutes"]}}}}
                """));

    assertThat(
            BfclCallMatcher.exactCallsMatch(
                mapper,
                List.of(
                    ToolCall.of(
                        0, "schedule", "{\"stops\":[{\"name\":\"metro-parkway\",\"minutes\":8}]}")),
                expected,
                tools))
        .isTrue();
    assertThat(
            BfclCallMatcher.exactCallsMatch(
                mapper,
                List.of(
                    ToolCall.of(
                        0, "schedule", "{\"stops\":[{\"name\":\"Metro Parkway\",\"minutes\":9}]}")),
                expected,
                tools))
        .isFalse();
  }

  @Test
  void validatesDeclaredArgumentsRequiredFieldsTypesAndEnums() {
    var tools =
        List.of(
            new ToolSpec(
                "weather",
                "Weather",
                """
                {"type":"object","properties":{"zipcode":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["zipcode"]}
                """));

    assertThat(
            BfclCallMatcher.callsValidate(
                mapper,
                List.of(
                    ToolCall.of(0, "weather", "{\"zipcode\":\"88252\",\"units\":\"fahrenheit\"}")),
                tools))
        .isTrue();
    assertThat(
            BfclCallMatcher.callsValidate(mapper, List.of(ToolCall.of(0, "weather", "{}")), tools))
        .isFalse();
    assertThat(
            BfclCallMatcher.callsValidate(
                mapper, List.of(ToolCall.of(0, "weather", "{\"zipcode\":88252}")), tools))
        .isFalse();
    assertThat(
            BfclCallMatcher.callsValidate(
                mapper,
                List.of(ToolCall.of(0, "weather", "{\"zipcode\":\"88252\",\"units\":\"kelvin\"}")),
                tools))
        .isFalse();
    assertThat(
            BfclCallMatcher.callsValidate(
                mapper,
                List.of(ToolCall.of(0, "weather", "{\"zipcode\":\"88252\",\"x\":1}")),
                tools))
        .isFalse();
  }
}
