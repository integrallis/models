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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.bench.ActivatedAnswerabilityQualificationCli.Arm;
import com.integrallis.models.bench.ActivatedAnswerabilityQualificationCli.Case;
import com.integrallis.models.bench.ActivatedAnswerabilityQualificationCli.CaseResult;
import com.integrallis.models.bench.ActivatedAnswerabilityQualificationCli.Document;
import com.integrallis.models.bench.ActivatedAnswerabilityQualificationCli.Message;
import com.integrallis.models.bench.ActivatedAnswerabilityQualificationCli.Summary;
import java.util.List;
import org.junit.jupiter.api.Test;

class ActivatedAnswerabilityQualificationCliTest {
  private static final Case CASE =
      new Case(
          "case-1",
          "unanswerable",
          List.of(
              new Message("user", "first"),
              new Message("assistant", "reply"),
              new Message("user", "what \"now\"?")),
          List.of(new Document(1, "quote \" and \\ backslash\nnewline"), new Document(2, "é")));

  @Test
  void rendersDocumentsExactlyAsThePythonTojsonFilterWould() {
    assertThat(ActivatedAnswerabilityQualificationCli.documentJson(CASE.documents().get(0)))
        .isEqualTo("{\"doc_id\": 1, \"text\": \"quote \\\" and \\\\ backslash\\nnewline\"}");
    assertThat(ActivatedAnswerabilityQualificationCli.documentJson(CASE.documents().get(1)))
        .as("non-ASCII stays literal because Transformers renders with ensure_ascii=False")
        .isEqualTo("{\"doc_id\": 2, \"text\": \"é\"}");
    assertThat(ActivatedAnswerabilityQualificationCli.documentsSystemMessage(CASE.documents()))
        .startsWith(ActivatedAnswerabilityQualificationCli.DOCUMENTS_PREFIX + "\n{\"doc_id\": 1")
        .contains("}\n{\"doc_id\": 2")
        .endsWith(ActivatedAnswerabilityQualificationCli.DOCUMENTS_SUFFIX);
  }

  @Test
  void keepsCallerContentAsTextAndOnlyDelimitersAndTheMarkerAsControl() {
    ModelPrompt prompt = ActivatedAnswerabilityQualificationCli.prompt(CASE, Arm.SPECIALIST);

    List<ModelPrompt.Segment> segments = prompt.segments();
    assertThat(segments.getLast().kind()).isEqualTo(ModelPrompt.SegmentKind.CONTROL);
    assertThat(segments.getLast().text())
        .as("adjacent control segments merge; the marker must still close the prompt")
        .endsWith(ActivatedAnswerabilityQualificationCli.INVOCATION);
    assertThat(
            segments.stream()
                .filter(segment -> segment.kind() == ModelPrompt.SegmentKind.TEXT)
                .map(ModelPrompt.Segment::text))
        .containsExactly(
            ActivatedAnswerabilityQualificationCli.documentsSystemMessage(CASE.documents()),
            "first",
            "reply",
            "what \"now\"?");
    assertThat(
            segments.stream()
                .filter(segment -> segment.kind() == ModelPrompt.SegmentKind.CONTROL)
                .map(ModelPrompt.Segment::text))
        .containsExactly(
            "<|start_of_role|>system<|end_of_role|>",
            "<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>",
            "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
            "<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>",
            "<|end_of_text|>\n" + ActivatedAnswerabilityQualificationCli.INVOCATION);
  }

  @Test
  void baseArmPrependsTheFixedInstructionToTheSameDocumentsPrompt() {
    ModelPrompt specialist = ActivatedAnswerabilityQualificationCli.prompt(CASE, Arm.SPECIALIST);
    ModelPrompt base = ActivatedAnswerabilityQualificationCli.prompt(CASE, Arm.BASE);

    assertThat(base.segments()).hasSameSizeAs(specialist.segments());
    assertThat(base.segments().get(1).text())
        .isEqualTo(
            ActivatedAnswerabilityQualificationCli.BASE_INSTRUCTION
                + "\n\n"
                + specialist.segments().get(1).text());
    assertThat(base.segments().subList(2, base.segments().size()))
        .isEqualTo(specialist.segments().subList(2, specialist.segments().size()));
  }

  @Test
  void predictionForgivesOnlySurroundingWhitespace() {
    assertThat(ActivatedAnswerabilityQualificationCli.prediction(" answerable\n"))
        .isEqualTo("answerable");
    assertThat(ActivatedAnswerabilityQualificationCli.prediction("unanswerable"))
        .isEqualTo("unanswerable");
    assertThat(ActivatedAnswerabilityQualificationCli.prediction("Answerable")).isEmpty();
    assertThat(ActivatedAnswerabilityQualificationCli.prediction("answerable.")).isEmpty();
    assertThat(
            ActivatedAnswerabilityQualificationCli.prediction(
                "{\"answerability\": \"answerable\"}"))
        .isEmpty();
    assertThat(ActivatedAnswerabilityQualificationCli.prediction(null)).isEmpty();
  }

  @Test
  void summarizesBalancedAccuracyAndRequiresSharingOnlyForTheSpecialistArm() {
    List<CaseResult> results =
        List.of(
            new CaseResult("a1", "answerable", "answerable", true, true, true, 10, 1, "answerable"),
            new CaseResult("a2", "answerable", "unanswerable", true, false, true, 10, 1, "x"),
            new CaseResult("u1", "unanswerable", "unanswerable", true, true, false, 10, 1, "y"),
            new CaseResult("u2", "unanswerable", "unanswerable", true, true, true, 10, 1, "z"));

    Summary specialist = ActivatedAnswerabilityQualificationCli.summarize(results, Arm.SPECIALIST);
    Summary base = ActivatedAnswerabilityQualificationCli.summarize(results, Arm.BASE);

    assertThat(specialist.balancedAccuracy()).isEqualTo(0.75);
    assertThat(specialist.structuredRate()).isEqualTo(1.0);
    assertThat(specialist.physicallyShared()).isEqualTo(3);
    assertThat(specialist.executionPassed())
        .as("one unshared case fails the specialist run")
        .isFalse();
    assertThat(base.executionPassed()).isTrue();
    assertThat(
            ActivatedAnswerabilityQualificationCli.summarize(List.of(), Arm.BASE).executionPassed())
        .isFalse();
  }

  @Test
  void loadsCasesFromTheFrozenWindowAndRejectsMalformedOnes() throws Exception {
    ObjectMapper json = new ObjectMapper();
    JsonNode window =
        json.readTree(
            """
            {"schemaVersion": 1, "windowSha256": "w", "suites": [{"name": "s", "source": {"file": "f"},
              "cases": [{"id": "c", "label": "answerable",
                "messages": [{"role": "user", "text": "q"}],
                "documents": [{"doc_id": 1, "text": "d"}]}]}]}
            """);

    JsonNode suite = ActivatedAnswerabilityQualificationCli.suite(window, "s");
    assertThat(ActivatedAnswerabilityQualificationCli.cases(suite))
        .containsExactly(
            new Case(
                "c",
                "answerable",
                List.of(new Message("user", "q")),
                List.of(new Document(1, "d"))));
    assertThatThrownBy(() -> ActivatedAnswerabilityQualificationCli.suite(window, "missing"))
        .hasMessageContaining("no suite named missing");

    JsonNode endsWithAssistant =
        json.readTree(
            """
            {"cases": [{"id": "c", "label": "answerable",
              "messages": [{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}],
              "documents": [{"doc_id": 1, "text": "d"}]}]}
            """);
    assertThatThrownBy(() -> ActivatedAnswerabilityQualificationCli.cases(endsWithAssistant))
        .hasMessageContaining("end with a user turn");

    JsonNode badLabel =
        json.readTree(
            """
            {"cases": [{"id": "c", "label": "partial",
              "messages": [{"role": "user", "text": "q"}],
              "documents": [{"doc_id": 1, "text": "d"}]}]}
            """);
    assertThatThrownBy(() -> ActivatedAnswerabilityQualificationCli.cases(badLabel))
        .hasMessageContaining("invalid case");
  }

  @Test
  void parsesArmsStrictlyAndRequiresTheModelsRevision() {
    String[] valid = {
      "--model",
      "m",
      "--adapter",
      "a",
      "--models-revision",
      "a".repeat(40),
      "--window",
      "w",
      "--suite",
      "s",
      "--arm",
      "Specialist",
      "--report",
      "r",
      "--limit",
      "3"
    };
    ActivatedAnswerabilityQualificationCli.Configuration configuration =
        ActivatedAnswerabilityQualificationCli.parse(valid);
    assertThat(configuration.arm()).isEqualTo(Arm.SPECIALIST);
    assertThat(configuration.limit()).isEqualTo(3);

    String[] badArm = valid.clone();
    badArm[11] = "hybrid";
    assertThatThrownBy(() -> ActivatedAnswerabilityQualificationCli.parse(badArm))
        .hasMessageContaining("--arm must be specialist or base");

    String[] badRevision = valid.clone();
    badRevision[5] = "main";
    assertThatThrownBy(() -> ActivatedAnswerabilityQualificationCli.parse(badRevision))
        .hasMessageContaining("40-character");
  }
}
