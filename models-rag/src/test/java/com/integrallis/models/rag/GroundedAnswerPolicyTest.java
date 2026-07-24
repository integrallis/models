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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GroundedAnswerPolicyTest {
  private static final GroundingDocument HIGH_CONFIDENCE =
      new GroundingDocument(
          "payments-settlement",
          "Approved domestic ACH claims settle within 2 business days. "
              + "International claim wires settle within 5 business days.",
          8.4f,
          1);
  private static final GroundingDocument LOW_CONFIDENCE =
      new GroundingDocument(
          "claims-auto-glass", "Windshield repair has a 75 dollar deductible.", 1.2f, 1);
  private final GroundedAnswerPolicy policy = new GroundedAnswerPolicy(2.0f);

  @Test
  void abstainsBeforeTrustingAWeakRetrieval() {
    GroundedAnswer answer =
        policy.apply(
            "What deductible applies to a lunar rover?",
            List.of(LOW_CONFIDENCE),
            "The deductible is 75 dollars. [claims-auto-glass]");

    assertThat(answer.text()).isEqualTo("INSUFFICIENT_CONTEXT");
    assertThat(answer.rawText()).contains("75 dollars");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.RETRIEVAL_ABSTENTION);
  }

  @Test
  void preservesAnAnswerWithOnlyTrustedCitations() {
    String generated =
        "Domestic claims take 2 days and international wires take 5 days. "
            + "[payments-settlement]";

    GroundedAnswer answer =
        policy.apply("How long do both payment types take?", List.of(HIGH_CONFIDENCE), generated);

    assertThat(answer.text()).isEqualTo(generated);
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER);
  }

  @Test
  void usesExtractiveEvidenceWhenTheModelOmitsCitations() {
    GroundedAnswer answer =
        policy.apply(
            "How long do both payment types take?",
            List.of(HIGH_CONFIDENCE),
            "Both payment types settle within 2 days.");

    assertThat(answer.text())
        .isEqualTo(
            "Approved domestic ACH claims settle within 2 business days. "
                + "International claim wires settle within 5 business days. "
                + "[payments-settlement]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }

  @Test
  void usesExtractiveEvidenceForAnUnsupportedCitation() {
    GroundedAnswer answer =
        policy.apply(
            "How long do both payment types take?",
            List.of(HIGH_CONFIDENCE),
            "Both settle within 2 days. [sources: 1, 2]");

    assertThat(answer.text()).endsWith("[payments-settlement]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }

  @Test
  void usesExtractiveEvidenceWhenATrustedCitationCarriesAnUnsupportedClaim() {
    GroundingDocument telemedicine =
        new GroundingDocument(
            "health-telemedicine",
            "Behavioral health telemedicine has a 15 dollar copay and allows 20 visits each year.",
            8.0f,
            1);

    GroundedAnswer answer =
        policy.apply(
            "What copay and annual limit apply to behavioral health telemedicine?",
            List.of(telemedicine),
            "The copay is 15 dollars and the annual visit count is unlimited. "
                + "[health-telemedicine]");

    assertThat(answer.text())
        .isEqualTo(
            "Behavioral health telemedicine has a 15 dollar copay and allows 20 visits each year. "
                + "[health-telemedicine]");
    assertThat(answer.rawText()).contains("unlimited");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }

  @Test
  void usesExtractiveEvidenceWhenAConfidentRetrievalIsFollowedByARefusal() {
    GroundedAnswer answer =
        policy.apply(
            "How long do both payment types take?",
            List.of(HIGH_CONFIDENCE),
            "INSUFFICIENT_CONTEXT.");

    assertThat(answer.text()).contains("5 business days").endsWith("[payments-settlement]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }
}
