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
            "refuse");
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
        Needle2ToolQualification.loadSuite(mapper).cases().getLast();

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

    Needle2ToolQualification.Summary summary =
        Needle2ToolQualification.summarize(List.of(passing, refusal));

    assertThat(summary.structuredOutputRate()).isEqualTo(1.0);
    assertThat(summary.toolSelectionExactRate()).isEqualTo(1.0);
    assertThat(summary.schemaValidityRate()).isEqualTo(1.0);
    assertThat(summary.declaredArgumentsOnlyRate()).isEqualTo(1.0);
    assertThat(summary.expectedArgumentAccuracy()).isEqualTo(0.9);
    assertThat(summary.refusalAccuracy()).isEqualTo(1.0);
    assertThat(summary.qualified()).isTrue();

    assertThat(
            Needle2ToolQualification.summarize(
                    List.of(result("bad", true, true, true, false, 10, 10, false, false), refusal))
                .qualified())
        .isFalse();
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
}
