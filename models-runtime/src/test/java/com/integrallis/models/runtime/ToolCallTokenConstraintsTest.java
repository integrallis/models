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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ToolCallTokenConstraintsTest {

  @Test
  void compilesNeedle2ArrayGrammarFromTheDeclaredToolSchema() {
    var tool =
        new ToolSpec(
            "set_mode",
            "Set the mode",
            """
            {"type":"object","properties":{"mode":{"type":"string","enum":["cool","heat"]}},"required":["mode"]}
            """);

    Optional<TokenConstraint> compiled =
        ToolCallTokenConstraints.compile(
            NEEDLE_TOKENIZER,
            ToolSyntax.NEEDLE2,
            List.of(tool),
            ignored -> List.of("{\"mode\":\"cool\"}"));

    assertThat(compiled).isPresent();
    TokenConstraint constraint = compiled.orElseThrow();
    constraint.accept(1);
    constraint.accept(2);
    constraint.accept(3);
    assertThat(constraint.allows(4)).isTrue();
    assertThat(constraint.allows(7)).isFalse();
    assertThat(constraint.allows(8)).isFalse();
    constraint.accept(4);
    constraint.accept(5);
    constraint.accept(6);
    assertThat(constraint.isComplete()).isTrue();
  }

  @Test
  void needle2GrammarAllowsARefusalAndParallelCalls() {
    var tool =
        new ToolSpec(
            "echo",
            "Echo text",
            """
            {"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}
            """);

    TokenConstraint refusal =
        ToolCallTokenConstraints.compile(
                NEEDLE_TOKENIZER, ToolSyntax.NEEDLE2, List.of(tool), ignored -> List.of())
            .orElseThrow();
    refusal.accept(2);
    refusal.accept(9);
    refusal.accept(6);
    assertThat(refusal.isComplete()).isTrue();

    TokenConstraint parallel =
        ToolCallTokenConstraints.compile(
                NEEDLE_TOKENIZER, ToolSyntax.NEEDLE2, List.of(tool), ignored -> List.of())
            .orElseThrow();
    parallel.accept(2);
    parallel.accept(3);
    parallel.accept(10);
    parallel.accept(11);
    parallel.accept(10);
    parallel.accept(5);
    parallel.accept(6);
    assertThat(parallel.isComplete()).isTrue();
  }

  private static final Tokenizer NEEDLE_TOKENIZER = new NeedleTokenizer();

  private static final class NeedleTokenizer implements Tokenizer {
    private static final String[] TOKENS = {
      "</s>",
      "<think>because</think>\n",
      "<tool_call>",
      "[",
      "{\"name\":\"set_mode\",\"arguments\":{\"mode\":\"cool\"}}",
      "]",
      "</tool_call>",
      "{\"name\":\"invented\",\"arguments\":{\"mode\":\"cool\"}}",
      "{\"name\":\"set_mode\",\"arguments\":{\"mode\":\"invalid\"}}",
      "[]",
      "{\"name\":\"echo\",\"arguments\":{\"text\":\"hello\"}}",
      ","
    };

    @Override
    public int[] encode(String text) {
      throw new AssertionError("not used");
    }

    @Override
    public String decode(int[] tokens) {
      throw new AssertionError("not used");
    }

    @Override
    public String decode(int token) {
      return TOKENS[token];
    }

    @Override
    public int vocabSize() {
      return TOKENS.length;
    }

    @Override
    public int bosToken() {
      return 0;
    }

    @Override
    public int eosToken() {
      return 0;
    }
  }
}
