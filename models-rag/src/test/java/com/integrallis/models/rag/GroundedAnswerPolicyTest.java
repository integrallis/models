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
          "Payment settlement",
          "Approved domestic ACH claims settle within 2 business days. "
              + "International claim wires settle within 5 business days.",
          8.4f,
          1);
  private static final GroundingDocument LOW_CONFIDENCE =
      new GroundingDocument(
          "claims-auto-glass",
          "Auto glass claims",
          "Windshield repair has a 75 dollar deductible.",
          1.2f,
          1);
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
  void derivesTrustedCitationsWhenTheModelAnswerIsOtherwiseSupported() {
    String generated = "Domestic claims settle within 2 business days.";

    GroundedAnswer answer =
        policy.apply("How long do both payment types take?", List.of(HIGH_CONFIDENCE), generated);

    assertThat(answer.text())
        .isEqualTo("Domestic claims settle within 2 business days. [payments-settlement]");
    assertThat(answer.rawText()).isEqualTo(generated);
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER_WITH_DERIVED_CITATIONS);
  }

  @Test
  void acceptsABoundedLeadingContextAttribution() {
    GroundingDocument autoGlass =
        new GroundingDocument(
            "claims-auto-glass",
            "Auto glass claims",
            "Northstar Mutual auto glass claims must be reported through the Aurora portal within "
                + "30 calendar days. Windshield repair has a 75 dollar deductible. A police "
                + "report is not required.",
            8.0f,
            1);
    String generated =
        "According to the context provided, Northstar Mutual auto glass claims must be reported "
            + "through the Aurora portal within 30 calendar days. Windshield repair has a 75 "
            + "dollar deductible. A police report is not required.";

    GroundedAnswer answer =
        policy.apply(
            "Where and when must Northstar Mutual auto glass claims be reported?",
            List.of(autoGlass),
            generated);

    assertThat(answer.text()).isEqualTo(generated + " [claims-auto-glass]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER_WITH_DERIVED_CITATIONS);
  }

  @Test
  void doesNotDeriveCitationsForAnUnsupportedUncitedClaim() {
    GroundedAnswer answer =
        policy.apply(
            "How long do both payment types take?",
            List.of(HIGH_CONFIDENCE),
            "Both payment types settle instantly.");

    assertThat(answer.text()).contains("2 business days").endsWith("[payments-settlement]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }

  @Test
  void doesNotDeriveCitationsForAQuestionRestatement() {
    GroundingDocument toolchain =
        new GroundingDocument(
            "gradle-java-toolchain",
            "Java build toolchain",
            "The build selects JDK 25 with "
                + "java.toolchain.languageVersion.set(JavaLanguageVersion.of(25)). "
                + "Java compilation also sets options.release to 25.",
            8.0f,
            1);

    GroundedAnswer answer =
        policy.apply(
            "Which expression selects the Gradle Java toolchain, and which release does "
                + "compilation target?",
            List.of(toolchain),
            "The expression selects the Gradle Java toolchain, and which release does "
                + "compilation target?");

    assertThat(answer.text())
        .contains("JavaLanguageVersion.of(25)")
        .endsWith("[gradle-java-toolchain]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }

  @Test
  void doesNotDeriveCitationsWhenOneQuestionClauseHasNoEvidenceAnchor() {
    GroundingDocument mapper =
        new GroundingDocument(
            "jackson-wire-format",
            "JSON wire mapper",
            "The wire mapper calls findAndRegisterModules so Java time values use registered "
                + "modules. It disables DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES.",
            8.0f,
            1);

    GroundedAnswer answer =
        policy.apply(
            "Which mapper method registers Java time support, and which deserialization feature "
                + "is disabled?",
            List.of(mapper),
            "The wire mapper registers Java time support, and "
                + "DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES is disabled.");

    assertThat(answer.text()).contains("findAndRegisterModules").endsWith("[jackson-wire-format]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }

  @Test
  void requiresAClauseSpecificQuestionTermForEveryConjunct() {
    GroundingDocument settlement =
        new GroundingDocument(
            "payments-settlement",
            "Payment settlement",
            "Approved domestic claims paid by ACH settle within 2 business days. "
                + "International claim wires settle within 5 business days and may incur an "
                + "intermediary bank fee.",
            8.4f,
            1);
    GroundedAnswer answer =
        policy.apply(
            "When do approved domestic ACH claims and international claim wires settle?",
            List.of(settlement),
            "Approved domestic claims paid by ACH settle within 2 business days. "
                + "[payments-settlement]");

    assertThat(answer.text())
        .contains("2 business days")
        .contains("5 business days")
        .endsWith("[payments-settlement]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }

  @Test
  void acceptsConjunctsWithDistinctEvidenceDespiteInflection() {
    GroundingDocument idempotency =
        new GroundingDocument(
            "api-idempotency",
            "Claims API idempotency",
            "Claims API clients send the Idempotency-Key header on create requests. "
                + "Keys remain valid for a 24 hour replay window.",
            8.0f,
            1);
    String generated =
        "Claims API clients send the Idempotency-Key header on create requests. "
            + "Keys remain valid for a 24 hour replay window. [api-idempotency]";

    GroundedAnswer answer =
        policy.apply(
            "Which header makes Claims API creates idempotent and how long can a key be replayed?",
            List.of(idempotency),
            generated);

    assertThat(answer.text()).isEqualTo(generated);
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER);
  }

  @Test
  void acceptsSafeProvenanceDiscourseAndIrregularThrowInflection() {
    GroundingDocument listCopy =
        new GroundingDocument(
            "java-list-copy",
            "Freezing parsed values",
            "The CatalogLoader freezes parsed SKU values by calling List.copyOf(values). "
                + "The method throws NullPointerException when any element is null.",
            8.0f,
            1);
    String generated =
        "The Java method that freezes parsed SKU values is List.copyOf(values), and the exception "
            + "thrown when an element is null is NullPointerException. The source context for this "
            + "information is \"[java-list-copy] Freezing parsed values\".";

    GroundedAnswer answer =
        policy.apply(
            "Which Java method freezes parsed SKU values, and what exception is thrown if an "
                + "element is null?",
            List.of(listCopy),
            generated);

    assertThat(answer.text()).isEqualTo(generated);
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER);
  }

  @Test
  void acceptsSupportedParaphrasesForEveryQuestionConjunct() {
    GroundingDocument compute =
        new GroundingDocument(
            "java-map-compute",
            "Atomic session cache initialization",
            "SessionCache initializes entries with ConcurrentHashMap.computeIfAbsent. "
                + "If the mapping function returns null, computeIfAbsent returns null and records "
                + "no mapping for that key.",
            8.0f,
            1);
    String generated =
        "The map operation is ConcurrentHashMap.computeIfAbsent which initializes SessionCache "
            + "entries. When the mapping function returns null, it records no mapping for that key.";

    GroundedAnswer answer =
        policy.apply(
            "Which map operation initializes SessionCache entries, and what happens when its "
                + "mapping function returns null?",
            List.of(compute),
            generated);

    assertThat(answer.text()).isEqualTo(generated + " [java-map-compute]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER_WITH_DERIVED_CITATIONS);
  }

  @Test
  void preservesConjunctionsInsideSupportedAnswerLists() {
    GroundingDocument retries =
        new GroundingDocument(
            "http-retry-idempotency",
            "HTTP retry rules",
            "The SDK retries GET and HEAD requests up to three attempts. A POST is retried only "
                + "when it carries an Idempotency-Key header.",
            8.0f,
            1);
    String generated =
        "The HTTP methods retried up to three attempts are GET and HEAD. The header that permits "
            + "retrying POST is the Idempotency-Key header.";

    GroundedAnswer answer =
        policy.apply(
            "Which HTTP methods are retried up to three attempts, and what header permits retrying "
                + "POST?",
            List.of(retries),
            generated);

    assertThat(answer.text()).isEqualTo(generated + " [http-retry-idempotency]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER_WITH_DERIVED_CITATIONS);
  }

  @Test
  void validatesSupportedMatrixAnswersWithoutTreatingLatexDelimitersAsClaims() {
    GroundingDocument matrix =
        new GroundingDocument(
            "math-amber-matrix",
            "Amber matrix",
            "For the Amber matrix [[3, 1], [2, 4]], the determinant is 10. The trace is 7.",
            8.0f,
            1);
    String generated =
        "For the Amber matrix \\(\\begin{bmatrix} 3 & 1 \\\\ 2 & 4 \\end{bmatrix}\\), "
            + "the determinant is 10 and the trace is 7";

    GroundedAnswer answer =
        policy.apply(
            "What are the determinant and trace of the Amber matrix?", List.of(matrix), generated);

    assertThat(answer.text()).isEqualTo(generated + " [math-amber-matrix]");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER_WITH_DERIVED_CITATIONS);
  }

  @Test
  void doesNotIgnoreArbitraryLatexCommandsDuringClaimValidation() {
    GroundingDocument matrix =
        new GroundingDocument(
            "math-amber-matrix",
            "Amber matrix",
            "For the Amber matrix [[3, 1], [2, 4]], the determinant is 10. The trace is 7.",
            8.0f,
            1);

    GroundedAnswer answer =
        policy.apply(
            "What are the determinant and trace of the Amber matrix?",
            List.of(matrix),
            "The determinant is 10 and the trace is 7 \\recommend{unsafe}");

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
            "Behavioral telemedicine",
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
  void preservesAnExplicitModelAbstentionDespiteAConfidentRetrieval() {
    GroundedAnswer answer =
        policy.apply(
            "How long do both payment types take?",
            List.of(HIGH_CONFIDENCE),
            "INSUFFICIENT_CONTEXT.");

    assertThat(answer.text()).isEqualTo("INSUFFICIENT_CONTEXT");
    assertThat(answer.rawText()).isEqualTo("INSUFFICIENT_CONTEXT.");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ABSTENTION);
  }

  @Test
  void preservesARepeatedExplicitModelAbstentionWithACitation() {
    String generated = "INSUFFICIENT_CONTEXT\n[payments-settlement] INSUFFICIENT_CONTEXT";

    GroundedAnswer answer =
        policy.apply("How long do both payment types take?", List.of(HIGH_CONFIDENCE), generated);

    assertThat(answer.text()).isEqualTo("INSUFFICIENT_CONTEXT");
    assertThat(answer.rawText()).isEqualTo(generated);
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ABSTENTION);
  }
}
