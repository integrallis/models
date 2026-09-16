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
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.ActivatedToolDecisionPolicy;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class ActivatedHybridDevelopmentCliTest {
  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void evaluatesStrictGeneratedCallsAgainstBfclAlternatives() throws Exception {
    var item = callCase();

    var correct =
        ActivatedHybridDevelopmentCli.evaluate(
            mapper,
            item,
            ActivatedToolDecisionPolicy.Decision.CALL_PRIMARY,
            40,
            8,
            true,
            "<tool_call>{\"name\":\"weather\",\"arguments\":{\"zipcode\":\"88252\"}}</tool_call>",
            10);
    var malformed =
        ActivatedHybridDevelopmentCli.evaluate(
            mapper,
            item,
            ActivatedToolDecisionPolicy.Decision.CALL_PRIMARY,
            40,
            8,
            true,
            "Here: <tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call>",
            10);

    assertThat(correct.syntaxValid()).isTrue();
    assertThat(correct.schemaValid()).isTrue();
    assertThat(correct.exact()).isTrue();
    assertThat(malformed.syntaxValid()).isFalse();
    assertThat(malformed.schemaValid()).isFalse();
    assertThat(malformed.exact()).isFalse();
  }

  @Test
  void treatsAnAbstentionAsCorrectOnlyWhenNoToolApplies() throws Exception {
    var missedCall =
        ActivatedHybridDevelopmentCli.evaluate(
            mapper, callCase(), ActivatedToolDecisionPolicy.Decision.NO_CALL, 20, 0, true, "", 10);
    var correctNoCall =
        ActivatedHybridDevelopmentCli.evaluate(
            mapper,
            noCallCase(),
            ActivatedToolDecisionPolicy.Decision.NO_CALL,
            20,
            0,
            true,
            "",
            10);

    assertThat(missedCall.exact()).isFalse();
    assertThat(correctNoCall.exact()).isTrue();
    assertThat(correctNoCall.syntaxValid()).isTrue();
    assertThat(correctNoCall.schemaValid()).isTrue();
  }

  @Test
  void requiresEveryFrozenDevelopmentFloorAndPhysicalPrefix() throws Exception {
    List<ActivatedHybridDevelopmentCli.Observation> passing = new ArrayList<>();
    for (int index = 0; index < 50; index++) {
      boolean called = index < 48;
      boolean exact = index < 43;
      passing.add(observation("call-" + index, true, called, exact, true));
    }
    for (int index = 0; index < 25; index++) {
      boolean called = index == 0;
      passing.add(observation("none-" + index, false, called, !called, true));
    }

    assertThat(ActivatedHybridDevelopmentCli.summarize(passing).passed()).isTrue();

    List<ActivatedHybridDevelopmentCli.Observation> copied = new ArrayList<>(passing);
    ActivatedHybridDevelopmentCli.Observation first = copied.getFirst();
    copied.set(
        0,
        new ActivatedHybridDevelopmentCli.Observation(
            first.id(),
            first.kind(),
            first.callExpected(),
            first.decision(),
            first.baseMargin(),
            first.specialistMargin(),
            false,
            first.output(),
            first.syntaxValid(),
            first.schemaValid(),
            first.exact(),
            first.elapsedMillis()));

    assertThat(ActivatedHybridDevelopmentCli.summarize(copied).passed()).isFalse();
  }

  private ActivatedHybridDevelopmentCli.SourceCase callCase() throws Exception {
    return new ActivatedHybridDevelopmentCli.SourceCase(
        "weather",
        "simple",
        true,
        ModelPrompt.control("prompt"),
        List.of(
            new ToolSpec(
                "weather",
                "Weather",
                "{\"type\":\"object\",\"properties\":{\"zipcode\":{\"type\":\"string\"}},\"required\":[\"zipcode\"]}")),
        mapper.readTree("[{\"weather\":{\"zipcode\":[\"88252\"]}}]"));
  }

  private ActivatedHybridDevelopmentCli.SourceCase noCallCase() throws Exception {
    return new ActivatedHybridDevelopmentCli.SourceCase(
        "none",
        "irrelevance",
        false,
        ModelPrompt.control("prompt"),
        callCase().tools(),
        mapper.readTree("[]"));
  }

  private static ActivatedHybridDevelopmentCli.Observation observation(
      String id, boolean expected, boolean called, boolean exact, boolean physicallyShared) {
    return new ActivatedHybridDevelopmentCli.Observation(
        id,
        expected ? "simple" : "irrelevance",
        expected,
        called
            ? ActivatedToolDecisionPolicy.Decision.CALL_PRIMARY
            : ActivatedToolDecisionPolicy.Decision.NO_CALL,
        40,
        called ? 8 : 0,
        physicallyShared,
        called ? "<tool_call>{}</tool_call>" : "",
        true,
        true,
        exact,
        10);
  }
}
