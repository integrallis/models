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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class RagVisibleAnswerAuditTest {
  private final RagCase question =
      new RagCase("cost", "What is the cost?", List.of("policy-a"), List.of("75 dollars"), true);
  private final List<RetrievedDocument> retrieved =
      List.of(
          new RetrievedDocument(
              new RagDocument("policy-a", "Cost", "The cost is 75 dollars."), 8, 1));

  @Test
  void doesNotCreditRequiredFactsInsideAContradictoryNegativeClaim() {
    assertThat(
            RagVisibleAnswerAudit.assess(
                    question, retrieved, "The cost is not $75 [policy-a].", 20, 256)
                .requiredFactsAndCitationsPass())
        .isFalse();
  }

  @Test
  void neverCreditsFactsInsideReasoning() {
    var a =
        RagVisibleAnswerAudit.assess(
            question,
            retrieved,
            "<think>The cost is 75 dollars [policy-a].</think>The cost is",
            96,
            96);
    assertThat(a.requiredFactsAndCitationsPass()).isFalse();
    assertThat(a.visibleText()).isEqualTo("The cost is");
    assertThat(a.reasoningText()).contains("75 dollars");
    assertThat(a.budgetReached()).isTrue();
  }

  @Test
  void rejectsUnclosedAndMismatchedReasoning() {
    for (String text :
        List.of(
            "<think>The cost is 75 dollars [policy-a].",
            "<think></reasoning>The cost is 75 dollars [policy-a].")) {
      var a = RagVisibleAnswerAudit.assess(question, retrieved, text, 20, 256);
      assertThat(a.reasoningComplete()).isFalse();
      assertThat(a.requiredFactsAndCitationsPass()).isFalse();
    }
  }

  @Test
  void scoresOnlyCompleteVisibleFactsWithTrustedCitations() {
    var a =
        RagVisibleAnswerAudit.assess(
            question,
            retrieved,
            "<think>Check the cost.</think>The cost is $75 [policy-a].",
            30,
            256);
    assertThat(a.requiredFactsAndCitationsPass()).isTrue();
    assertThat(a.visibleText()).isEqualTo("The cost is $75 [policy-a].");
  }

  @Test
  void numericSubstringsAndIdentifierPrefixesDoNotCountAsFacts() {
    assertThat(
            RagVisibleAnswerAudit.assess(
                    question, retrieved, "The cost is $175 [policy-a].", 20, 256)
                .requiredFactsAndCitationsPass())
        .isFalse();
    var id = new RagCase("form", "Which form?", List.of("policy-a"), List.of("T-19"), true);
    assertThat(
            RagVisibleAnswerAudit.assess(id, retrieved, "Use T-190 [policy-a].", 20, 256)
                .requiredFactsAndCitationsPass())
        .isFalse();
  }

  @Test
  void abstentionMustBeVisibleAndExact() {
    var q = new RagCase("unknown", "Unknown?", List.of(), List.of(), false);
    assertThat(
            RagVisibleAnswerAudit.assess(
                    q, List.of(), "<think>No evidence.</think>INSUFFICIENT_CONTEXT", 20, 256)
                .requiredFactsAndCitationsPass())
        .isTrue();
    assertThat(
            RagVisibleAnswerAudit.assess(
                    q, List.of(), "<think>INSUFFICIENT_CONTEXT</think>Probably 75 dollars", 20, 256)
                .requiredFactsAndCitationsPass())
        .isFalse();
  }
}
