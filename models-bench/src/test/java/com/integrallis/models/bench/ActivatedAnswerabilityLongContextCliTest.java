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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.bench.ActivatedAnswerabilityLongContextCli.Case;
import com.integrallis.models.bench.ActivatedAnswerabilityLongContextCli.CaseResult;
import com.integrallis.models.bench.ActivatedAnswerabilityLongContextCli.Suite;
import com.integrallis.models.bench.ActivatedAnswerabilityLongContextCli.Summary;
import com.integrallis.models.runtime.chat.GraniteDocumentsPrompt;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class ActivatedAnswerabilityLongContextCliTest {
  private static final Case ANSWERABLE =
      new Case("lc-01", "Blue Line 17", "KX-4471", "Blue Line 17", "answerable");
  private static final Case UNANSWERABLE =
      new Case("lc-06", "Purple Line 5", "HB-7740", "Gold Line 9", "unanswerable");
  private static final Set<String> CODES = Set.of("KX-4471", "HB-7740");

  @Test
  void bundledSuiteIsValidAndBalancedAsTheGateDeclares() throws Exception {
    Suite suite = ActivatedAnswerabilityLongContextCli.loadSuite();

    assertThat(suite.targetPrefixTokens()).isEqualTo(4_096);
    assertThat(suite.cases()).hasSize(8);
    assertThat(suite.cases().stream().filter(c -> "answerable".equals(c.label())).count())
        .isEqualTo(5);
    assertThat(suite.cases().stream().filter(c -> "unanswerable".equals(c.label())).count())
        .isEqualTo(3);
  }

  @Test
  void answerableCasesNeedTheirCodeAndUnanswerableCasesNeedARefusalWithoutAnyCode() {
    assertThat(
            ActivatedAnswerabilityLongContextCli.correctAnswer(
                "The archive code for Blue Line 17 is KX-4471.", ANSWERABLE, CODES))
        .isTrue();
    assertThat(
            ActivatedAnswerabilityLongContextCli.correctAnswer(
                "The archive code is kx-4471", ANSWERABLE, CODES))
        .as("code matching ignores case")
        .isTrue();
    assertThat(
            ActivatedAnswerabilityLongContextCli.correctAnswer(
                "The documents do not mention it.", ANSWERABLE, CODES))
        .isFalse();
    assertThat(
            ActivatedAnswerabilityLongContextCli.correctAnswer(
                "The question cannot be answered based on the available data.",
                UNANSWERABLE,
                CODES))
        .isTrue();
    assertThat(
            ActivatedAnswerabilityLongContextCli.correctAnswer(
                "The archive code for Gold Line 9 is HB-7740.", UNANSWERABLE, CODES))
        .as("borrowing another route's code is a hallucination, not a refusal")
        .isFalse();
    assertThat(
            ActivatedAnswerabilityLongContextCli.correctAnswer(
                "Gold Line 9 is a scenic route.", UNANSWERABLE, CODES))
        .as("no code but no refusal is not a correct unanswerable reply")
        .isFalse();
  }

  @Test
  void promptPlantsTheFactFirstFillsTheSecondDocumentAndAsksAboutTheRequestedRoute() {
    ModelPrompt prompt =
        ActivatedAnswerabilityLongContextCli.prompt(UNANSWERABLE, " transit transit");

    String rendered =
        prompt.segments().stream().map(ModelPrompt.Segment::text).reduce("", String::concat);
    assertThat(rendered)
        .contains(
            "{\"doc_id\": 1, \"text\": \"Transit archive fact: route Purple Line 5 has archive code HB-7740.\"}")
        .contains("{\"doc_id\": 2, \"text\": \"Background records: transit transit\"}")
        .contains(
            "<|start_of_role|>user<|end_of_role|>What is the archive code for route Gold Line 9?<|end_of_text|>")
        .endsWith(GraniteDocumentsPrompt.ASSISTANT_MARKER);
    assertThat(prompt.segments().getLast().kind()).isEqualTo(ModelPrompt.SegmentKind.CONTROL);
  }

  @Test
  void summaryRequiresEveryRetentionContractAndBothCorrectnessFloors() {
    List<CaseResult> results = new ArrayList<>();
    for (int index = 0; index < 8; index++) {
      results.add(result("c" + index, index < 5 ? "answerable" : "unanswerable", true, true, true));
    }
    Summary all = ActivatedAnswerabilityLongContextCli.summarize(results, 4_096);
    assertThat(all.qualified()).isTrue();
    assertThat(all.nativeCorrect()).isEqualTo(8);

    List<CaseResult> twoNativeMisses = new ArrayList<>(results);
    twoNativeMisses.set(0, result("c0", "answerable", true, false, true));
    twoNativeMisses.set(1, result("c1", "answerable", true, false, true));
    assertThat(ActivatedAnswerabilityLongContextCli.summarize(twoNativeMisses, 4_096).qualified())
        .as("6 of 8 native-correct answers meet the 0.75 floor")
        .isTrue();

    List<CaseResult> lostRetention = new ArrayList<>(results);
    lostRetention.set(
        2,
        new CaseResult(
            "c2", "answerable", 4_096, true, true, false, false, true, "", "", "", 1, 1, 1));
    assertThat(ActivatedAnswerabilityLongContextCli.summarize(lostRetention, 4_096).qualified())
        .as("a native-correct answer lost on the shared branch fails the gate")
        .isFalse();

    List<CaseResult> unshared = new ArrayList<>(results);
    unshared.set(
        3,
        new CaseResult(
            "c3", "answerable", 4_096, true, true, true, true, false, "", "", "", 1, 1, 1));
    assertThat(ActivatedAnswerabilityLongContextCli.summarize(unshared, 4_096).qualified())
        .isFalse();

    List<CaseResult> shortPrefix = new ArrayList<>(results);
    shortPrefix.set(
        4,
        new CaseResult(
            "c4", "answerable", 4_000, true, true, true, true, true, "", "", "", 1, 1, 1));
    assertThat(ActivatedAnswerabilityLongContextCli.summarize(shortPrefix, 4_096).qualified())
        .isFalse();

    List<CaseResult> threeSpecialistMisses = new ArrayList<>(results);
    for (int index = 0; index < 3; index++) {
      threeSpecialistMisses.set(index, result("c" + index, "answerable", false, true, true));
    }
    assertThat(
            ActivatedAnswerabilityLongContextCli.summarize(threeSpecialistMisses, 4_096)
                .qualified())
        .as("5 of 8 specialist labels is below the 0.75 floor")
        .isFalse();
  }

  @Test
  void suiteValidationRejectsMislabeledOrCollidingCases() {
    List<Case> cases = new ArrayList<>();
    for (int index = 0; index < 8; index++) {
      boolean answerable = index < 5;
      cases.add(
          new Case(
              "c" + index,
              "Route " + index,
              "CODE-" + index,
              answerable ? "Route " + index : "Absent " + index,
              answerable ? "answerable" : "unanswerable"));
    }
    ActivatedAnswerabilityLongContextCli.validateSuite(new Suite(1, "w", 4_096, cases));

    List<Case> mislabeled = new ArrayList<>(cases);
    mislabeled.set(0, new Case("c0", "Route 0", "CODE-0", "Route 0", "unanswerable"));
    assertThatThrownBy(
            () ->
                ActivatedAnswerabilityLongContextCli.validateSuite(
                    new Suite(1, "w", 4_096, mislabeled)))
        .hasMessageContaining("exactly when answerable");

    List<Case> colliding = new ArrayList<>(cases);
    colliding.set(5, new Case("c5", "Route 5", "CODE-5", "Route 1", "unanswerable"));
    assertThatThrownBy(
            () ->
                ActivatedAnswerabilityLongContextCli.validateSuite(
                    new Suite(1, "w", 4_096, colliding)))
        .hasMessageContaining("another case plants");

    assertThatThrownBy(
            () ->
                ActivatedAnswerabilityLongContextCli.validateSuite(new Suite(1, "w", 2_048, cases)))
        .hasMessageContaining("invalid answerability long-context suite");
  }

  private static CaseResult result(
      String id, String label, boolean specialist, boolean nativeCorrect, boolean retained) {
    return new CaseResult(
        id, label, 4_096, specialist, nativeCorrect, retained, true, true, "", "", "", 1, 1, 1);
  }
}
