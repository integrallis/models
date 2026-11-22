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
package com.integrallis.models.decisions;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The lettered-prompt decision path: what the model is shown, and what is read back. */
@Tag("unit")
class LetterDecisionEvaluatorTest {

  private static final String STATE = "The invoice was paid twice.";
  private static final Choice ROUTE =
      new Choice(
          "Which team owns this?",
          List.of("billing", "support"),
          Map.of("billing", "money was charged or refunded"));
  private static final Noul DUPLICATE = new Noul("Was the customer charged more than once?");
  private static final Score URGENCY =
      new Score("How urgent is this?", List.of("low", "medium", "high"));

  /** Softmax over the letter logits, computed here rather than by the code under test. */
  private static double[] expected(String prompt, int options) {
    float[] logits = ScriptedPrefixBackend.logitsAfter(prompt);
    double[] weights = new double[options];
    double total = 0;
    for (int index = 0; index < options; index++) {
      weights[index] = Math.exp(logits['A' + index]);
      total += weights[index];
    }
    for (int index = 0; index < options; index++) {
      weights[index] /= total;
    }
    return weights;
  }

  @Test
  void theSuffixPutsTheRubricBeforeTheQuestionAndTheLettersAfterIt() {
    assertThat(LetterDecisionEvaluator.questionSuffix(ROUTE))
        .isEqualTo(
            "\nOptions:\n- billing: money was charged or refunded\n- support: support"
                + "\nWhich team owns this?\nA: billing\nB: support\nAnswer:");
    assertThat(LetterDecisionEvaluator.questionSuffix(DUPLICATE))
        .isEqualTo("\nWas the customer charged more than once?\nA: false\nB: true\nAnswer:");
  }

  @Test
  void theLetterTokenIsTheLastTokenOfSpaceLetter() {
    ScriptedPrefixBackend backend = new ScriptedPrefixBackend();

    assertThat(LetterDecisionEvaluator.letterTokens(backend.tokenizer(), 3))
        .containsExactly('A', 'B', 'C');
  }

  @Test
  void oneQuestionReadsTheWholePromptOnceAndScoresTheLetters() {
    ScriptedPrefixBackend backend = new ScriptedPrefixBackend();
    String prompt = STATE + LetterDecisionEvaluator.questionSuffix(ROUTE);

    LetterDecisionBatch batch =
        new LetterDecisionEvaluator(backend, 1.0).evaluate(STATE, List.of(ROUTE));

    assertThat(backend.scoredPrompts).containsExactly(prompt);
    assertThat(batch.verdicts()).hasSize(1);
    assertThat(batch.verdicts().get(0).probabilities())
        .containsExactly(expected(prompt, 2), within(1e-12));
    assertThat(batch.statePrefills()).isEqualTo(1);
    assertThat(batch.sharedPrefixProven()).isFalse();
    assertThat(batch.inputTokens()).isEqualTo(prompt.length());
  }

  @Test
  void severalQuestionsReadTheStateOnceAndGiveTheSameAnswersAsAskingEachAlone() {
    List<AnswerSpace> spaces = List.of(ROUTE, DUPLICATE, URGENCY);
    ScriptedPrefixBackend shared = new ScriptedPrefixBackend();

    LetterDecisionBatch batch = new LetterDecisionEvaluator(shared, 1.0).evaluate(STATE, spaces);

    assertThat(shared.freezes).isEqualTo(1);
    assertThat(batch.statePrefills()).isEqualTo(1);
    assertThat(batch.sharedPrefixProven()).isTrue();
    assertThat(shared.scoredPrompts.get(0)).isEqualTo(STATE);
    assertThat(batch.inputTokens())
        .isEqualTo(
            STATE.length()
                + spaces.stream()
                    .mapToInt(space -> LetterDecisionEvaluator.questionSuffix(space).length())
                    .sum());
    for (int index = 0; index < spaces.size(); index++) {
      AnswerSpace space = spaces.get(index);
      String prompt = STATE + LetterDecisionEvaluator.questionSuffix(space);
      assertThat(batch.verdicts().get(index).space()).isEqualTo(space);
      assertThat(batch.verdicts().get(index).probabilities())
          .containsExactly(expected(prompt, space.size()), within(1e-12));
      assertThat(
              new LetterDecisionEvaluator(new ScriptedPrefixBackend(), 1.0)
                  .evaluate(STATE, List.of(space))
                  .verdicts()
                  .get(0)
                  .probabilities())
          .containsExactly(batch.verdicts().get(index).probabilities(), within(1e-12));
    }
  }

  @Test
  void aBackendThatCannotShareAPrefixReadsTheStateForEveryQuestionAndSaysSo() {
    ScriptedPrefixBackend backend = new ScriptedPrefixBackend();
    backend.sharedPrefixes = false;

    LetterDecisionBatch batch =
        new LetterDecisionEvaluator(backend, 1.0).evaluate(STATE, List.of(ROUTE, DUPLICATE));

    assertThat(backend.freezes).isZero();
    assertThat(batch.statePrefills()).isEqualTo(2);
    assertThat(batch.sharedPrefixProven()).isFalse();
    assertThat(batch.verdicts().get(1).probabilities())
        .containsExactly(
            expected(STATE + LetterDecisionEvaluator.questionSuffix(DUPLICATE), 2), within(1e-12));
  }

  @Test
  void sharingIsReportedAsUnprovenWhenTheBackendCannotProveIt() {
    ScriptedPrefixBackend backend = new ScriptedPrefixBackend();
    backend.proveSharing = false;

    LetterDecisionBatch batch =
        new LetterDecisionEvaluator(backend, 1.0).evaluate(STATE, List.of(ROUTE, DUPLICATE));

    assertThat(batch.statePrefills()).isEqualTo(1);
    assertThat(batch.sharedPrefixProven()).isFalse();
  }

  @Test
  void temperatureFlattensTheDistributionWithoutChangingItsOrder() {
    double[] cold =
        new LetterDecisionEvaluator(new ScriptedPrefixBackend(), 1.0)
            .evaluate(STATE, List.of(URGENCY))
            .verdicts()
            .get(0)
            .probabilities();
    double[] warm =
        new LetterDecisionEvaluator(new ScriptedPrefixBackend(), 4.0)
            .evaluate(STATE, List.of(URGENCY))
            .verdicts()
            .get(0)
            .probabilities();

    double coldSpread =
        java.util.Arrays.stream(cold).max().orElseThrow()
            - java.util.Arrays.stream(cold).min().orElseThrow();
    double warmSpread =
        java.util.Arrays.stream(warm).max().orElseThrow()
            - java.util.Arrays.stream(warm).min().orElseThrow();
    assertThat(warmSpread).isLessThan(coldSpread);
  }

  @Test
  void anEmptyStateOrNoQuestionsIsRefused() {
    LetterDecisionEvaluator evaluator =
        new LetterDecisionEvaluator(new ScriptedPrefixBackend(), 1.0);

    assertThatThrownBy(() -> evaluator.evaluate("", List.of(ROUTE)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("no tokens");
    assertThatThrownBy(() -> evaluator.evaluate(STATE, List.of()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("at least one");
  }
}
