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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RagQualificationCliTest {

  private static final Path REPOSITORY_ROOT = Path.of(System.getProperty("models.repositoryRoot"));
  private static final Path EVIDENCE_ROOT =
      REPOSITORY_ROOT.resolve(
          "benchmark-results/certified-20260724/rag/native-q8-prefix-cache/" + "smollm2-360m-q8_0");

  @Test
  void emitsAuditableQualificationEvidenceForCertifiedReports(@TempDir Path temporaryDirectory)
      throws Exception {
    Path candidate = EVIDENCE_ROOT.resolve("smollm2-360m-rust-ffm-prefix-cache-grounded.json");
    List<Path> comparators =
        List.of(
            EVIDENCE_ROOT.resolve("smollm2-360m-llama-current-grounded.json"),
            EVIDENCE_ROOT.resolve("smollm2-360m-ollama-current-grounded.json"));
    Path output = temporaryDirectory.resolve("qualification.json");

    RagQualificationEvidence evidence =
        RagQualificationCli.run(
            new String[] {
              "--candidate",
              candidate.toString(),
              "--comparator",
              comparators.get(0).toString(),
              "--comparator",
              comparators.get(1).toString(),
              "--output",
              output.toString(),
              "--require-qualified"
            });

    assertThat(evidence.schemaVersion()).isEqualTo(1);
    assertThat(evidence.policyId()).isEqualTo(RagProductionQualificationPolicy.POLICY_ID);
    assertThat(evidence.candidate().path()).isEqualTo(candidate.toString());
    assertThat(evidence.candidate().sha256()).hasSize(64);
    assertThat(evidence.comparators())
        .extracting(RagQualificationReportReference::path)
        .containsExactlyElementsOf(comparators.stream().map(Path::toString).toList());
    assertThat(evidence.qualification().qualified()).isTrue();
    assertThat(evidence.qualification().qualifyingComparators()).containsExactly("ollama");
    assertThat(evidence.qualification().comparisons())
        .extracting(RagComparatorAssessment::comparatorBackend)
        .containsExactly("llama.cpp", "ollama");
    assertThat(new ObjectMapper().readValue(output.toFile(), RagQualificationEvidence.class))
        .isEqualTo(evidence);
  }

  @Test
  void rejectsMissingComparatorEvidence() {
    assertThatThrownBy(
            () ->
                RagQualificationCli.run(
                    new String[] {"--candidate", EVIDENCE_ROOT.resolve("missing.json").toString()}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("--comparator");
  }
}
