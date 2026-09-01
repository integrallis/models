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
import com.fasterxml.jackson.databind.SerializationFeature;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.ToolCallTokenConstraints;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.util.List;
import org.junit.jupiter.api.Test;

class Needle2ToolQualificationTest {
  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void loadsTheCompleteVersionedUpstreamPlaygroundSuite() throws Exception {
    Needle2ToolQualification.Suite suite = Needle2ToolQualification.loadSuite(mapper);

    assertThat(suite.schemaVersion()).isEqualTo(1);
    assertThat(suite.suiteId()).isEqualTo("needle2-upstream-playground-v1");
    assertThat(suite.sourceRevision()).hasSize(40);
    assertThat(suite.cases())
        .extracting(Needle2ToolQualification.Case::id)
        .containsExactly(
            "home",
            "robot",
            "device",
            "gallery",
            "email",
            "currency",
            "document",
            "sentiment",
            "crowded",
            "repeat",
            "array",
            "flight",
            "refuse",
            "spring-weather-zipcode");
  }

  @Test
  void evaluatesParallelCallsSchemaAndExpectedArguments() throws Exception {
    Needle2ToolQualification.Case item =
        Needle2ToolQualification.loadSuite(mapper).cases().getFirst();
    String output =
        "<think>route both actions</think>\n"
            + "<tool_call>[{\"name\":\"set_lights\",\"arguments\":"
            + "{\"room\":\"bedroom\",\"state\":\"on\",\"brightness\":20}},"
            + "{\"name\":\"lock_door\",\"arguments\":{\"door\":\"front door\"}}]"
            + "</tool_call><|im_end|>";

    Needle2ToolQualification.CaseResult result =
        Needle2ToolQualification.evaluate(mapper, item, output, 1234);

    assertThat(result.structured()).isTrue();
    assertThat(result.selectionExact()).isTrue();
    assertThat(result.schemaValid()).isTrue();
    assertThat(result.declaredArgumentsOnly()).isTrue();
    assertThat(result.expectedArgumentMatches()).isEqualTo(4);
    assertThat(result.expectedArguments()).isEqualTo(4);
    assertThat(result.passed()).isTrue();
  }

  @Test
  void rejectsUnknownFieldsEvenWhenJsonSchemaWouldAllowThemByDefault() throws Exception {
    Needle2ToolQualification.Case item =
        Needle2ToolQualification.loadSuite(mapper).cases().stream()
            .filter(candidate -> candidate.id().equals("sentiment"))
            .findFirst()
            .orElseThrow();
    String output =
        "<tool_call>[{\"name\":\"classify_sentiment\",\"arguments\":"
            + "{\"sentiment\":\"negative\",\"urgent\":true}}]</tool_call>";

    Needle2ToolQualification.CaseResult result =
        Needle2ToolQualification.evaluate(mapper, item, output, 10);

    assertThat(result.structured()).isTrue();
    assertThat(result.selectionExact()).isTrue();
    assertThat(result.schemaValid()).isTrue();
    assertThat(result.declaredArgumentsOnly()).isFalse();
    assertThat(result.passed()).isFalse();
  }

  @Test
  void recognizesThePublishedEmptyArrayAsAnIntentionalRefusal() throws Exception {
    Needle2ToolQualification.Case item =
        Needle2ToolQualification.loadSuite(mapper).cases().stream()
            .filter(candidate -> candidate.id().equals("refuse"))
            .findFirst()
            .orElseThrow();

    Needle2ToolQualification.CaseResult result =
        Needle2ToolQualification.evaluate(mapper, item, "<tool_call>[]</tool_call><|im_end|>", 25);

    assertThat(result.structured()).isTrue();
    assertThat(result.refusalExpected()).isTrue();
    assertThat(result.refusalCorrect()).isTrue();
    assertThat(result.passed()).isTrue();
  }

  @Test
  void appliesTheDeclaredConformancePolicyWithoutRounding() {
    var passing = result("a", true, true, true, true, 9, 10, false, true);
    var refusal = result("refuse", true, true, true, true, 0, 0, true, true);
    var regression = result("spring-weather-zipcode", true, true, true, true, 0, 0, false, true);

    Needle2ToolQualification.Summary summary =
        Needle2ToolQualification.summarize(List.of(passing, refusal, regression));

    assertThat(summary.structuredOutputRate()).isEqualTo(1.0);
    assertThat(summary.toolSelectionExactRate()).isEqualTo(1.0);
    assertThat(summary.schemaValidityRate()).isEqualTo(1.0);
    assertThat(summary.declaredArgumentsOnlyRate()).isEqualTo(1.0);
    assertThat(summary.expectedArgumentAccuracy()).isEqualTo(0.9);
    assertThat(summary.refusalAccuracy()).isEqualTo(1.0);
    assertThat(summary.qualified()).isTrue();

    assertThat(
            Needle2ToolQualification.summarize(
                    List.of(
                        result("perfect", true, true, true, true, 10, 10, false, true), refusal))
                .qualified())
        .isFalse();

    assertThat(
            Needle2ToolQualification.summarize(
                    List.of(
                        result("perfect", true, true, true, true, 10, 10, false, true),
                        refusal,
                        regression))
                .qualified())
        .isTrue();

    assertThat(
            Needle2ToolQualification.summarize(
                    List.of(
                        result("bad", true, true, true, false, 10, 10, false, false),
                        refusal,
                        regression))
                .qualified())
        .isFalse();
  }

  @Test
  void everyUpstreamToolSchemaCompilesIntoTheNeedleGrammar() throws Exception {
    Needle2ToolQualification.Suite suite = Needle2ToolQualification.loadSuite(mapper);

    for (Needle2ToolQualification.Case item : suite.cases()) {
      assertThat(
              ToolCallTokenConstraints.compile(
                  new CompileOnlyTokenizer(),
                  ChatTemplate.NEEDLE2.toolSyntax(),
                  item.toolSpecs(mapper),
                  ignored -> List.of()))
          .as(item.id())
          .isPresent();
    }
  }

  @Test
  void robotFixturePreservesTheOfficialPromptSchemasByteForByte() throws Exception {
    ObjectMapper reportMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    Needle2ToolQualification.Case robot =
        Needle2ToolQualification.loadSuite(reportMapper).cases().stream()
            .filter(item -> item.id().equals("robot"))
            .findFirst()
            .orElseThrow();

    assertThat(robot.toolSpecs(reportMapper))
        .containsExactly(
            new ToolSpec(
                "move",
                "Drive the robot in a direction.",
                "{\"type\":\"object\",\"properties\":{\"direction\":{\"type\":\"string\","
                    + "\"enum\":[\"forward\",\"backward\",\"left\",\"right\"]},"
                    + "\"distance_m\":{\"type\":\"number\",\"description\":\"Distance in meters.\"}},"
                    + "\"required\":[\"direction\",\"distance_m\"]}"),
            new ToolSpec(
                "rotate",
                "Rotate the robot in place.",
                "{\"type\":\"object\",\"properties\":{\"direction\":{\"type\":\"string\","
                    + "\"enum\":[\"left\",\"right\"]},\"degrees\":{\"type\":\"number\"}},"
                    + "\"required\":[\"direction\",\"degrees\"]}"),
            new ToolSpec(
                "gripper",
                "Open or close the gripper.",
                "{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\","
                    + "\"enum\":[\"open\",\"close\"]}},\"required\":[\"action\"]}"));

    String rendered =
        ChatTemplate.NEEDLE2
            .render(List.of(ChatMessage.user(robot.query())), robot.toolSpecs(reportMapper))
            .text();
    assertThat(rendered).hasSize(829);
    assertThat(rendered.hashCode()).isEqualTo(-849368063);
  }

  private static Needle2ToolQualification.CaseResult result(
      String id,
      boolean structured,
      boolean selection,
      boolean schema,
      boolean declaredOnly,
      int matches,
      int expected,
      boolean refusalExpected,
      boolean refusalCorrect) {
    return new Needle2ToolQualification.CaseResult(
        id,
        1,
        "output",
        structured,
        selection,
        schema,
        declaredOnly,
        matches,
        expected,
        refusalExpected,
        refusalCorrect,
        structured
            && selection
            && schema
            && declaredOnly
            && matches == expected
            && (!refusalExpected || refusalCorrect),
        List.of());
  }

  private static final class CompileOnlyTokenizer implements Tokenizer {
    @Override
    public int[] encode(String text) {
      throw new AssertionError("not used while compiling");
    }

    @Override
    public String decode(int[] tokens) {
      throw new AssertionError("not used while compiling");
    }

    @Override
    public String decode(int token) {
      throw new AssertionError("not used while compiling");
    }

    @Override
    public int vocabSize() {
      return 1;
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
