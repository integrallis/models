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

class RagCorpusTest {

  @Test
  void defaultCorpusHasStableUniqueDocumentsAndCases() {
    RagCorpus corpus = RagCorpus.loadDefault();

    assertThat(corpus.documents()).hasSize(12);
    assertThat(corpus.documents()).extracting(RagDocument::id).doesNotHaveDuplicates();
    assertThat(corpus.cases()).hasSize(9);
    assertThat(corpus.cases()).extracting(RagCase::id).doesNotHaveDuplicates();
    assertThat(corpus.cases()).filteredOn(RagCase::answerable).hasSize(8);
    assertThat(corpus.fingerprint())
        .isEqualTo("4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c");
  }

  @Test
  void codingWorkloadHasIndependentDocumentsAndEvaluationCases() {
    RagCorpus corpus = RagCorpus.load(RagWorkload.CODING);

    assertThat(corpus.documents()).hasSize(12);
    assertThat(corpus.documents()).extracting(RagDocument::id).doesNotHaveDuplicates();
    assertThat(corpus.cases()).hasSize(9);
    assertThat(corpus.cases()).extracting(RagCase::id).doesNotHaveDuplicates();
    assertThat(corpus.cases()).filteredOn(RagCase::answerable).hasSize(8);
    assertThat(corpus.cases())
        .extracting(RagCase::id)
        .contains("list-copy-null-handling", "virtual-thread-executor", "unsupported-cuda-kernel");
    assertThat(corpus.fingerprint())
        .isEqualTo("6841c286837b4c45c06fe8d103b2e044b61a1bfe75a61b64fa04c7ca31b20e45");
  }

  @Test
  void domainWorkloadsHaveIndependentDocumentsAndEvaluationCases() {
    List<RagWorkload> workloads =
        List.of(
            RagWorkload.FINANCE,
            RagWorkload.HEALTHCARE,
            RagWorkload.LEGAL,
            RagWorkload.MATH,
            RagWorkload.MULTILINGUAL,
            RagWorkload.SQL,
            RagWorkload.SUMMARIZATION,
            RagWorkload.TRANSPORTATION);

    assertThat(workloads).extracting(RagWorkload::id).doesNotHaveDuplicates();
    assertThat(workloads)
        .extracting(workload -> RagCorpus.load(workload).fingerprint())
        .doesNotHaveDuplicates();
    for (RagWorkload workload : workloads) {
      RagCorpus corpus = RagCorpus.load(workload);
      assertThat(corpus.documents()).as("%s documents", workload.id()).hasSize(12);
      assertThat(corpus.documents())
          .as("%s document IDs", workload.id())
          .extracting(RagDocument::id)
          .doesNotHaveDuplicates();
      assertThat(corpus.cases()).as("%s cases", workload.id()).hasSize(9);
      assertThat(corpus.cases())
          .as("%s case IDs", workload.id())
          .extracting(RagCase::id)
          .doesNotHaveDuplicates();
      assertThat(corpus.cases())
          .as("%s answerable cases", workload.id())
          .filteredOn(RagCase::answerable)
          .hasSize(8);
    }
  }

  @Test
  void breakGlassOracleAcceptsTheNumberOfApprovingManagersAskedFor() {
    RagCorpus corpus = RagCorpus.loadDefault();
    RagCase testCase =
        corpus.cases().stream()
            .filter(value -> value.id().equals("break-glass"))
            .findFirst()
            .orElseThrow();
    RagDocument document =
        corpus.documents().stream()
            .filter(value -> value.id().equals("security-access"))
            .findFirst()
            .orElseThrow();

    RagEvaluation evaluation =
        RagEvaluator.evaluate(
            testCase,
            List.of(new RetrievedDocument(document, 3.0f, 1)),
            "The code name is Cobalt-17 and two managers approve it [security-access].");
    RagEvaluation canonicalEvaluation =
        RagEvaluator.evaluate(
            testCase,
            List.of(new RetrievedDocument(document, 3.0f, 1)),
            "Cobalt-17 requires two on-call managers [security-access].");

    assertThat(evaluation.factCoverage()).isEqualTo(1.0);
    assertThat(evaluation.correct()).isTrue();
    assertThat(canonicalEvaluation.correct()).isTrue();
  }

  @Test
  void legalOracleAcceptsAnEquivalentConfidentialityBreachPhrase() {
    RagCorpus corpus = RagCorpus.load(RagWorkload.LEGAL);
    RagCase testCase =
        corpus.cases().stream()
            .filter(value -> value.id().equals("cobalt-liability-cap"))
            .findFirst()
            .orElseThrow();
    RagDocument document =
        corpus.documents().stream()
            .filter(value -> value.id().equals("legal-cobalt-cap"))
            .findFirst()
            .orElseThrow();

    RagEvaluation evaluation =
        RagEvaluator.evaluate(
            testCase,
            List.of(new RetrievedDocument(document, 3.0f, 1)),
            "Cobalt's liability cap equals fees paid during the prior 12 months, and breaches of "
                + "confidentiality are excluded from this cap. [legal-cobalt-cap]");

    assertThat(evaluation.factCoverage()).isEqualTo(1.0);
    assertThat(evaluation.correct()).isTrue();
  }

  @Test
  void healthcareOracleAcceptsConciseRetentionAndNurseAnswers() {
    RagCorpus corpus = RagCorpus.load(RagWorkload.HEALTHCARE);
    RagCase cedar =
        corpus.cases().stream()
            .filter(value -> value.id().equals("cedar-storage-retention"))
            .findFirst()
            .orElseThrow();
    RagCase fir =
        corpus.cases().stream()
            .filter(value -> value.id().equals("fir-reconciliation-owner"))
            .findFirst()
            .orElseThrow();
    RagDocument cedarDocument =
        corpus.documents().stream()
            .filter(value -> value.id().equals("health-cedar-specimens"))
            .findFirst()
            .orElseThrow();
    RagDocument firDocument =
        corpus.documents().stream()
            .filter(value -> value.id().equals("health-fir-reconciliation"))
            .findFirst()
            .orElseThrow();

    RagEvaluation cedarEvaluation =
        RagEvaluator.evaluate(
            cedar,
            List.of(new RetrievedDocument(cedarDocument, 3.0f, 1)),
            "7-year retention is after study closure at minus 80 degrees Celsius "
                + "[health-cedar-specimens].");
    RagEvaluation firEvaluation =
        RagEvaluator.evaluate(
            fir,
            List.of(new RetrievedDocument(firDocument, 3.0f, 1)),
            "Fir performs medication reconciliation at every admission and discharge. "
                + "The nurse records completion. [health-fir-reconciliation]");

    assertThat(cedarEvaluation.correct()).isTrue();
    assertThat(firEvaluation.correct()).isTrue();
  }

  @Test
  void mathOracleAcceptsAnEquivalentSequenceStartPhrase() {
    RagCorpus corpus = RagCorpus.load(RagWorkload.MATH);
    RagCase testCase =
        corpus.cases().stream()
            .filter(value -> value.id().equals("delta-sixth-term"))
            .findFirst()
            .orElseThrow();
    RagDocument document =
        corpus.documents().stream()
            .filter(value -> value.id().equals("math-delta-sequence"))
            .findFirst()
            .orElseThrow();

    RagEvaluation evaluation =
        RagEvaluator.evaluate(
            testCase,
            List.of(new RetrievedDocument(document, 3.0f, 1)),
            "The start of the Delta sequence is 7, its common difference is 4, and its sixth "
                + "term is 27. [math-delta-sequence]");

    assertThat(evaluation.factCoverage()).isEqualTo(1.0);
    assertThat(evaluation.correct()).isTrue();
  }

  /**
   * Every answerable case must survive the grounding screen its own documents will face.
   *
   * <p>Not a hand-written restatement of the rule: it calls {@link GroundedAnswerPolicy#assess}
   * with the case's relevant documents, so it cannot drift from the policy and cannot be satisfied
   * by a rule that only looks right. Scores are set above the floor on purpose -- retrieval quality
   * is a separate concern, and what is being checked here is the part a corpus author controls:
   * document size, absence of injection markers, and the named-entity overlap between a question
   * and its source.
   *
   * <p>Written after authoring a workload that failed every case. Retrieval was fine -- the correct
   * document came back at 4.3 to 7.7 against a floor of 2.0 -- but seven of nine questions were
   * rejected as QUESTION_MISMATCH, because every one began with the capitalised word "Summarize",
   * which counts as a named entity and appeared in no document. The two that passed were the two
   * that also named a product the document mentions. A corpus can retrieve perfectly and still be
   * unusable, and nothing caught it until three models had been run against it.
   */
  @Test
  void everyAnswerableCasePassesTheGroundingContextScreen() {
    GroundedAnswerPolicy policy = GroundedAnswerPolicy.productionDefault();
    List<String> rejected = new java.util.ArrayList<>();
    for (RagWorkload workload : RagWorkload.values()) {
      RagCorpus corpus = RagCorpus.load(workload);
      java.util.Map<String, RagDocument> byId = new java.util.HashMap<>();
      corpus.documents().forEach(document -> byId.put(document.id(), document));
      for (RagCase ragCase : corpus.cases()) {
        if (!ragCase.answerable()) {
          continue;
        }
        List<GroundingDocument> retrieved = new java.util.ArrayList<>();
        int rank = 1;
        for (String id : ragCase.relevantDocumentIds()) {
          RagDocument document = byId.get(id);
          assertThat(document)
              .describedAs(
                  "%s case %s names a document that does not exist: %s",
                  workload.id(), ragCase.id(), id)
              .isNotNull();
          retrieved.add(
              new GroundingDocument(
                  document.id(), document.title(), document.text(), 8.0f, rank++));
        }
        GroundingContextDecision decision = policy.assess(ragCase.question(), retrieved);
        if (decision != GroundingContextDecision.ACCEPTED) {
          rejected.add(workload.id() + "/" + ragCase.id() + " -> " + decision);
        }
      }
    }
    assertThat(rejected).isEmpty();
  }
}
