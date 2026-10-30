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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolSpec;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FlintToolCallConstraintTest {
  private static final ToolSpec TOOL =
      new ToolSpec(
          "set_mode",
          "Set mode",
          """
      {"type":"object","properties":{
        "mode":{"type":"string","enum":["cool","heat"]},
        "enabled":{"type":"boolean"},
        "label":{"type":"string"}},"required":["mode","enabled"],"additionalProperties":false}
      """);
  private static final String CALL =
      "<tool_call>set_mode<arg_key>mode</arg_key><arg_value>cool</arg_value>"
          + "<arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>";

  @Test
  void acceptsRawStringsTypedValuesOptionalArgumentsAndParallelCalls() {
    var constraint = compile(TOOL);
    feed(constraint, CALL);
    assertThat(constraint.isComplete()).isFalse(); // Closing one call must not force early stop.
    assertThat(constraint.allows(0)).isTrue();
    feed(
        constraint,
        "\n"
            + CALL.replace(
                "</tool_call>",
                "<arg_key>label</arg_key><arg_value>Hello!\nworld</arg_value></tool_call>"));
    assertThat(constraint.allows(0)).isTrue();
    var recovered = ToolCallScanner.scan(CALL + CALL, ToolSyntax.FLINT, List.of(TOOL));
    assertThat(recovered.toolCalls()).hasSize(2);
    assertThat(recovered.toolCalls().getFirst().argumentsJson())
        .isEqualTo("{\"mode\":\"cool\",\"enabled\":true}");
  }

  @Test
  void preservesRefusalAndIgnoresCallExamplesInsideReasoning() {
    var refusal = compile(TOOL);
    feed(refusal, "<think>Do not execute <tool_call>made_up</tool_call>.</think>I cannot do that.");
    assertThat(refusal.allows(0)).isTrue();
    var reasoning = compile(TOOL);
    feed(reasoning, "<think>Consider <tool_call>made_up</tool_call>.");
    assertThat(reasoning.allows(0)).isFalse();
    feed(reasoning, "</think>" + CALL);
    assertThat(reasoning.allows(0)).isTrue();
  }

  @Test
  void doesNotPermitMissingRequiredWrongTypesUnknownNamesOrDuplicateKeys() {
    for (String invalid :
        List.of(
            CALL.replace("set_mode", "delete_all"),
            CALL.replace("cool", "cold"),
            CALL.replace("true", "\"true\""),
            CALL.replace("<arg_key>enabled</arg_key><arg_value>true</arg_value>", ""),
            CALL.replace("enabled", "unexpected"),
            CALL.replace(
                "</tool_call>",
                "<arg_key>enabled</arg_key><arg_value>false</arg_value></tool_call>"),
            CALL + "unframed prose",
            "</tool_call>",
            "<|assistant|>")) {
      assertThatThrownBy(() -> feed(compile(TOOL), invalid))
          .as(invalid)
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void rejectsEarlyEndAndControlInjectionAcrossFragments() {
    var partial = compile(TOOL);
    feed(partial, "<tool_ca");
    assertThat(partial.allows(0)).isFalse();
    feed(partial, "ll>set_mode<arg_key>mode</arg_key><arg_value>cool");
    assertThat(partial.allows(0)).isFalse();
    assertThatThrownBy(() -> feed(partial, "<|assistant|>"))
        .isInstanceOf(IllegalArgumentException.class);
    var nested = compile(TOOL);
    feed(nested, "<think>reasoning");
    assertThatThrownBy(() -> feed(nested, "<think>")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void acceptsNestedJsonAndEscapedStringFragmentsButPreservesTheirTypes() {
    ToolSpec nested =
        new ToolSpec(
            "write",
            "Write document",
            """
        {"type":"object","properties":{"document":{"type":"object","properties":{
          "title":{"type":"string"},"counts":{"type":"array","items":{"type":"integer"}},
          "ratio":{"type":"number"},"absent":{"type":"null"}},
          "required":["title","counts","ratio","absent"]}},"required":["document"]}
        """);
    String output =
        "<tool_call>write<arg_key>document</arg_key><arg_value>"
            + "{\"title\":\"Hello \\\"world\\\"\\n\",\"counts\":[1,-2,0],\"ratio\":1.25e-2,\"absent\":null}"
            + "</arg_value></tool_call>";
    var constraint = compile(nested);
    feed(constraint, output);
    assertThat(constraint.allows(0)).isTrue();
    assertThat(ToolCallScanner.scan(output, ToolSyntax.FLINT, List.of(nested)).hasCalls()).isTrue();
    assertThatThrownBy(() -> feed(compile(nested), output.replace("[1,-2,0]", "[1.5]")))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void refusesUnsupportedSchemasInsteadOfDroppingTheirValidationRules() {
    for (String schema :
        List.of(
            "{\"type\":\"object\",\"$ref\":\"#/definitions/args\"}",
            "{\"type\":\"object\",\"properties\":{\"count\":{\"type\":\"integer\",\"minimum\":3}}}",
            "{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\",\"pattern\":\"x+\"}}}",
            "{\"type\":\"object\",\"required\":[\"missing\"]}",
            "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":[\"string\",\"null\"]}}}",
            "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\",\"const\":\"<tool_call>\"}}}")) {
      assertThatThrownBy(() -> compile(new ToolSpec("test", "test", schema)))
          .as(schema)
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void checksTokensThatCrossSeveralProtocolBoundaries() {
    Tokenizer whole =
        new CharacterTokenizer() {
          @Override
          public String decode(int token) {
            return switch (token) {
              case 1 -> "<think>Example <tool_call>ignored</tool_call></think>" + CALL + CALL;
              case 2 -> CALL + "<tool_call>unknown</tool_call>";
              default -> "";
            };
          }
        };
    var constraint = FlintToolCallConstraint.compile(whole, List.of(TOOL));
    assertThat(constraint.allows(2)).isFalse();
    constraint.accept(1);
    assertThat(constraint.allows(0)).isTrue();
  }

  private static FlintToolCallConstraint compile(ToolSpec tool) {
    return FlintToolCallConstraint.compile(new CharacterTokenizer(), List.of(tool));
  }

  private static void feed(FlintToolCallConstraint constraint, String text) {
    text.chars().forEach(constraint::accept);
  }

  private static class CharacterTokenizer implements Tokenizer {
    @Override
    public int[] encode(String text) {
      return text.chars().toArray();
    }

    @Override
    public String decode(int[] tokens) {
      return new String(tokens, 0, tokens.length);
    }

    @Override
    public String decode(int token) {
      return Character.toString(token);
    }

    @Override
    public int vocabSize() {
      return 65536;
    }

    @Override
    public int bosToken() {
      return 1;
    }

    @Override
    public int eosToken() {
      return 0;
    }
  }
}
