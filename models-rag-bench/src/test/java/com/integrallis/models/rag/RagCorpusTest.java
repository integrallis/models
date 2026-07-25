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
}
