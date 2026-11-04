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

import com.integrallis.models.decisions.Choice;
import java.net.URL;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The triage task's loading rules and its metrics, without a model.
 *
 * <p>Deliberately model-free: the arithmetic and the refusals are what a benchmark gets wrong
 * silently, and they are exactly the parts that do not need weights to check. A run against a real
 * model answers a different question and costs minutes.
 */
class TriageTaskTest {

  private static final String QUESTION = "Which team should handle this customer message?";

  private static Path fixture(String name) {
    URL resource = TriageTaskTest.class.getClassLoader().getResource("triage/" + name);
    assertThat(resource).as("fixture %s", name).isNotNull();
    return Path.of(resource.getPath());
  }

  private static TriageTask load() {
    return TriageTask.load(
        fixture("fixture-messages.tsv"), fixture("fixture-rules.tsv"), QUESTION, 0);
  }

  @Test
  void loadsOutcomesInRuleOrderAndMessagesInFileOrder() {
    TriageTask task = load();

    // Rule order is prompt order: it decides which letter each outcome is rendered as, so it must
    // be the file's order and not, say, alphabetical.
    assertThat(task.outcomes()).containsExactly("BILLING", "ACCOUNT", "ORDER", "TECHNICAL");
    assertThat(task.messages()).hasSize(10);
    assertThat(task.messages().getFirst().text()).startsWith("I was charged twice");
    assertThat(task.criteria())
        .containsEntry(
            "TECHNICAL", "something is broken, erroring, slow, or not working as documented");
  }

  @Test
  void normalisesLabelCasingSoACorpusCasingIsNotADifferentOutcome() {
    TriageTask task = load();

    // The fixture spells the same outcomes as BILLING/billing and ACCOUNT/Account.
    assertThat(task.messages().stream().map(TriageTask.Message::expected).distinct())
        .containsExactlyInAnyOrder("BILLING", "ACCOUNT", "ORDER", "TECHNICAL");
  }

  @Test
  void skipsCommentsAndBlankLines() {
    TriageTask task = load();

    assertThat(task.messages()).noneSatisfy(message -> assertThat(message.text()).startsWith("#"));
    assertThat(task.messages()).hasSize(10);
  }

  @Test
  void limitTakesThePrefixRatherThanASample() {
    TriageTask task =
        TriageTask.load(fixture("fixture-messages.tsv"), fixture("fixture-rules.tsv"), QUESTION, 3);

    assertThat(task.messages()).hasSize(3);
    assertThat(task.messages().getLast().text()).startsWith("I want a refund");
  }

  @Test
  void refusesAMessageLabelThatNoRuleCovers() {
    // The failure this prevents is quiet: an uncovered label cannot be chosen, so it reads as the
    // model being wrong on every one of those cases.
    assertThatThrownBy(
            () ->
                TriageTask.load(
                    fixture("fixture-messages-unknown-label.tsv"),
                    fixture("fixture-rules.tsv"),
                    QUESTION,
                    0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("NEWSLETTER")
        .hasMessageContaining("FEEDBACK");
  }

  @Test
  void theAnswerSpaceCarriesTheRulesNotJustTheLabels() {
    Choice space = load().space();

    assertThat(space.question()).isEqualTo(QUESTION);
    assertThat(space.options()).containsExactly("BILLING", "ACCOUNT", "ORDER", "TECHNICAL");
    // A space without criteria renders no rubric, which measured 15 points worse on our own set.
    assertThat(space.criteria()).hasSize(4);
  }

  @Test
  void scoresAccuracyAndWhatAGateWouldHaveDecided() {
    TriageTask task =
        TriageTask.load(fixture("fixture-messages.tsv"), fixture("fixture-rules.tsv"), QUESTION, 4);
    // Messages are BILLING, BILLING, BILLING, ACCOUNT. Three right, one wrong, and the wrong one
    // is confident -- which is the case a gate is supposed to catch and does not.
    List<String> chosen = List.of("BILLING", "BILLING", "ORDER", "ACCOUNT");
    List<Double> confidence = List.of(0.95, 0.60, 0.97, 0.91);

    TriageTask.Scored scored = task.score(chosen, confidence, 0.90);

    assertThat(scored.cases()).isEqualTo(4);
    assertThat(scored.accuracy()).isEqualTo(0.75);
    // Three of four clear 0.90, and two of those three are right.
    assertThat(scored.coverageAtGate()).isEqualTo(0.75);
    assertThat(scored.accuracyAtGate())
        .isCloseTo(2.0 / 3.0, org.assertj.core.data.Offset.offset(1e-9));
    assertThat(scored.confusions()).containsEntry("BILLING -> ORDER", 1);
    assertThat(scored.perOutcome()).containsKeys("BILLING", "ACCOUNT");
    assertThat(scored.perOutcome().get("BILLING")).containsExactly(2, 3);
  }

  @Test
  void reportsNoAccuracyWhenAGateDecidesNothing() {
    TriageTask task =
        TriageTask.load(fixture("fixture-messages.tsv"), fixture("fixture-rules.tsv"), QUESTION, 2);

    TriageTask.Scored scored = task.score(List.of("BILLING", "BILLING"), List.of(0.4, 0.5), 0.99);

    // Not zero: nothing was decided, so there is no accuracy to report, and zero would read as
    // "it got them all wrong".
    assertThat(scored.coverageAtGate()).isZero();
    assertThat(scored.accuracyAtGate()).isNaN();
  }

  @Test
  void refusesAnswersThatDoNotLineUpWithTheMessages() {
    TriageTask task = load();

    assertThatThrownBy(() -> task.score(List.of("BILLING"), List.of(0.9), 0.9))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("one answer and one confidence per message");
  }

  @Test
  void refusesABlankQuestion() {
    assertThatThrownBy(
            () ->
                TriageTask.load(
                    fixture("fixture-messages.tsv"), fixture("fixture-rules.tsv"), "  ", 0))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
