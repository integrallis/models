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
package com.integrallis.models.runtime.chat;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.ToolCall;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ToolCallScannerTest {

  @Nested
  static class HarmonyCalls {

    @Test
    void extractsTheRecipientAndArgumentsFromACommentaryCall() {
      String output =
          "<|channel|>analysis<|message|>I need the weather.<|end|>"
              + "<|start|>assistant to=functions.get-weather-for-zipcode"
              + "<|channel|>commentary json<|message|>{\"zipcode\":\"88252\"}<|call|>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.HARMONY);

      assertThat(result.content()).isEqualTo("I need the weather.");
      assertThat(result.toolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get-weather-for-zipcode");
                assertThat(call.argumentsJson()).isEqualTo("{\"zipcode\":\"88252\"}");
              });
    }

    @Test
    void extractsACallThatContinuesTheAssistantHeaderAlreadyInThePrompt() {
      String output =
          " to=functions.get-weather-for-zipcode<|channel|>commentary json"
              + "<|message|>{\"zipcode\":\"88252\"}";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.HARMONY);

      assertThat(result.content()).isEmpty();
      assertThat(result.toolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get-weather-for-zipcode");
                assertThat(call.argumentsJson()).isEqualTo("{\"zipcode\":\"88252\"}");
              });
    }

    @Test
    void extractsFinalChannelTextWithoutInventingACall() {
      String output = "<|channel|>final<|message|>It is raining.<|end|>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.HARMONY);

      assertThat(result.hasCalls()).isFalse();
      assertThat(result.content()).isEqualTo("It is raining.");
    }
  }

  @Nested
  static class TaggedCalls {

    @Test
    void extractsASingleQwenStyleCall() {
      String output =
          "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Austin\"}}\n</tool_call>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);

      assertThat(result.hasCalls()).isTrue();
      assertThat(result.toolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get_weather");
                assertThat(call.argumentsJson()).isEqualTo("{\"city\": \"Austin\"}");
                assertThat(call.id()).isEqualTo("000000000");
              });
      assertThat(result.content()).isEmpty();
    }

    @Test
    void keepsProseThatPrecedesTheCall() {
      String output =
          "Let me look that up.\n<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);

      assertThat(result.content()).isEqualTo("Let me look that up.");
      assertThat(result.toolCalls()).hasSize(1);
    }

    @Test
    void extractsParallelCallsAndNumbersThemInOrder() {
      String output =
          "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":1}}</tool_call>"
              + "<tool_call>{\"name\":\"b\",\"arguments\":{\"y\":2}}</tool_call>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);

      assertThat(result.toolCalls()).extracting(ToolCall::name).containsExactly("a", "b");
      assertThat(result.toolCalls())
          .extracting(ToolCall::id)
          .containsExactly("000000000", "000000001");
    }

    @Test
    void stopsAtTheFirstCallWhenTheFamilyForbidsParallelCalls() {
      // Llama 3.x templates raise when asked to render more than one call, so accepting a second
      // here would produce a conversation that cannot be rendered back.
      String output = "{\"name\":\"a\",\"parameters\":{}}\n{\"name\":\"b\",\"parameters\":{}}";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.LLAMA3);

      assertThat(result.toolCalls()).extracting(ToolCall::name).containsExactly("a");
    }

    @Test
    void toleratesAnUnclosedSection() {
      // Generation can stop on max-tokens mid-call; the JSON is still complete.
      String output = "<tool_call>{\"name\":\"ping\",\"arguments\":{}}";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);

      assertThat(result.toolCalls()).extracting(ToolCall::name).containsExactly("ping");
    }

    @Test
    void extractsNeedle2ParallelCallsFromOneArrayWrappedSection() {
      String output =
          "<think>two actions</think>\n<tool_call>["
              + "{\"name\":\"set_lights\",\"arguments\":{\"room\":\"bedroom\",\"brightness\":20}},"
              + "{\"name\":\"lock_door\",\"arguments\":{\"door\":\"front\"}}"
              + "]</tool_call><|im_end|>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.NEEDLE2);

      assertThat(result.toolCalls())
          .extracting(ToolCall::name)
          .containsExactly("set_lights", "lock_door");
      assertThat(result.toolCalls())
          .extracting(ToolCall::argumentsJson)
          .containsExactly("{\"room\":\"bedroom\",\"brightness\":20}", "{\"door\":\"front\"}");
    }
  }

  @Nested
  static class BareJson {

    @Test
    void extractsALlamaStyleCallKeyedOnParameters() {
      String output = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Austin\"}}";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.LLAMA3);

      assertThat(result.toolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get_weather");
                assertThat(call.argumentsJson()).isEqualTo("{\"city\": \"Austin\"}");
              });
    }

    @Test
    void acceptsEitherArgumentKeyRegardlessOfTheDeclaredOne() {
      // Models drift between the two spellings; both frameworks tolerate it, so we do too.
      assertThat(
              ToolCallScanner.scan("{\"name\":\"a\",\"arguments\":{\"k\":1}}", ToolSyntax.LLAMA3)
                  .toolCalls())
          .hasSize(1);
      assertThat(
              ToolCallScanner.scan(
                      "<tool_call>{\"name\":\"a\",\"parameters\":{\"k\":1}}</tool_call>",
                      ToolSyntax.QWEN)
                  .toolCalls())
          .hasSize(1);
    }
  }

  @Nested
  static class MarkdownFences {

    @Test
    void stripsAFencedJsonCall() {
      // The single most common real-world failure: models wrap output in a code fence.
      String output = "```json\n{\"name\":\"ping\",\"parameters\":{}}\n```";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.LLAMA3);

      assertThat(result.toolCalls()).extracting(ToolCall::name).containsExactly("ping");
    }

    @Test
    void stripsAFenceWrappingATaggedCall() {
      String output = "```\n<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>\n```";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);

      assertThat(result.toolCalls()).extracting(ToolCall::name).containsExactly("ping");
    }
  }

  @Nested
  static class JsonExtents {

    @Test
    void handlesNestedObjectsInArguments() {
      String output =
          "<tool_call>{\"name\":\"a\",\"arguments\":{\"o\":{\"p\":{\"q\":1}},\"r\":2}}</tool_call>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);

      assertThat(result.toolCalls())
          .singleElement()
          .satisfies(
              call ->
                  assertThat(call.argumentsJson()).isEqualTo("{\"o\":{\"p\":{\"q\":1}},\"r\":2}"));
    }

    @Test
    void ignoresBracesAndQuotesInsideStringValues() {
      String output =
          "<tool_call>{\"name\":\"a\",\"arguments\":{\"q\":\"}{ \\\" not json\"}}</tool_call>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);

      assertThat(result.toolCalls())
          .singleElement()
          .satisfies(
              call -> assertThat(call.argumentsJson()).isEqualTo("{\"q\":\"}{ \\\" not json\"}"));
    }

    @Test
    void defaultsToAnEmptyObjectWhenArgumentsAreAbsent() {
      String output = "<tool_call>{\"name\":\"ping\"}</tool_call>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);

      assertThat(result.toolCalls())
          .singleElement()
          .satisfies(call -> assertThat(call.argumentsJson()).isEqualTo("{}"));
    }
  }

  @Nested
  static class Degradation {

    @Test
    void returnsPlainTextWhenTheFamilyDeclaresNoToolFormat() {
      String output = "<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>";

      ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.NONE);

      assertThat(result.hasCalls()).isFalse();
      assertThat(result.content()).isEqualTo(output);
    }

    @Test
    void returnsPlainTextWhenNoCallIsPresent() {
      ToolCallScanner.Result result = ToolCallScanner.scan("The answer is 4.", ToolSyntax.QWEN);

      assertThat(result.hasCalls()).isFalse();
      assertThat(result.content()).isEqualTo("The answer is 4.");
    }

    @Test
    void neverThrowsOnMalformedOutput() {
      String[] malformed = {
        "<tool_call>not json at all</tool_call>",
        "<tool_call>{\"name\":</tool_call>",
        "<tool_call>{\"arguments\":{}}</tool_call>", // no name
        "<tool_call>{",
        "{\"name\":\"\",\"arguments\":{}}", // blank name
        "",
      };

      for (String output : malformed) {
        ToolCallScanner.Result result = ToolCallScanner.scan(output, ToolSyntax.QWEN);
        assertThat(result).as("output %s", output).isNotNull();
        assertThat(result.toolCalls()).as("output %s", output).isEmpty();
      }
    }

    @Test
    void toleratesNullOutput() {
      ToolCallScanner.Result result = ToolCallScanner.scan(null, ToolSyntax.QWEN);

      assertThat(result.hasCalls()).isFalse();
      assertThat(result.content()).isEmpty();
    }
  }
}
