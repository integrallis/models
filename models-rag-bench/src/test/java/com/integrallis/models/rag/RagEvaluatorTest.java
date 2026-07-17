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

class RagEvaluatorTest {

  @Test
  void scoresRetrievalFactsAndCitationsWithoutAnLlmJudge() {
    RagDocument expected = new RagDocument("policy-a", "Policy A", "irrelevant");
    RagDocument distractor = new RagDocument("policy-b", "Policy B", "irrelevant");
    RagCase testCase =
        new RagCase(
            "case-a", "question", List.of("policy-a"), List.of("30 days", "75 dollars"), true);

    RagEvaluation evaluation =
        RagEvaluator.evaluate(
            testCase,
            List.of(
                new RetrievedDocument(distractor, 2.0f, 1),
                new RetrievedDocument(expected, 1.0f, 2)),
            "The deadline is 30 days and the cost is 75 dollars [policy-a].");

    assertThat(evaluation.retrievalRecall()).isEqualTo(1.0);
    assertThat(evaluation.reciprocalRank()).isEqualTo(0.5);
    assertThat(evaluation.factCoverage()).isEqualTo(1.0);
    assertThat(evaluation.citationRecall()).isEqualTo(1.0);
    assertThat(evaluation.citationPrecision()).isEqualTo(1.0);
    assertThat(evaluation.correct()).isTrue();
  }

  @Test
  void onlyExactInsufficientContextCountsAsAValidAbstention() {
    RagCase testCase = new RagCase("unknown", "question", List.of(), List.of(), false);

    assertThat(RagEvaluator.evaluate(testCase, List.of(), "INSUFFICIENT_CONTEXT").correct())
        .isTrue();
    assertThat(RagEvaluator.evaluate(testCase, List.of(), "I think it is 50 dollars.").correct())
        .isFalse();
  }

  @Test
  void normalizesCurrencySymbolsAndPluralDollars() {
    RagDocument expected = new RagDocument("policy-a", "Policy A", "irrelevant");
    RagCase testCase =
        new RagCase(
            "currency",
            "question",
            List.of("policy-a"),
            List.of("75 dollar", "3,500 dollar"),
            true);

    RagEvaluation evaluation =
        RagEvaluator.evaluate(
            testCase,
            List.of(new RetrievedDocument(expected, 1.0f, 1)),
            "The deductible is $75 and the limit is $3,500 [policy-a].");

    assertThat(evaluation.factCoverage()).isEqualTo(1.0);
    assertThat(evaluation.correct()).isTrue();
  }
}
