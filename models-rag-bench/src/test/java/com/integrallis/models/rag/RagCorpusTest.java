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
        .isEqualTo("6eeb61d5a4b48addb298889a2357cfbcbe7339c044308ba8cd23dcb27c603cb2");
  }
}
