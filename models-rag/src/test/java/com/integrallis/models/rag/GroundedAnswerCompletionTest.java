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
class GroundedAnswerCompletionTest {
  private final GroundedAnswerPolicy policy = GroundedAnswerPolicy.productionDefault();

  private static GroundingDocument doc(String text) {
    return new GroundingDocument("evidence", "Service rules", text, 8.0f, 1);
  }

  @Test
  void acceptsSecondPersonAddressWithoutChangingFactualClaims() {
    var evidence =
        doc("Glass claims must be reported within 30 calendar days. The deductible is 75 dollars.");
    var answer =
        policy.apply(
            "How long do I have to report a glass claim and what is the deductible?",
            List.of(evidence),
            "You have 30 calendar days to report a glass claim and the deductible is $75 [evidence].");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER);
  }

  @Test
  void acceptsRedundantNoPrefixButPreservesExplicitNegation() {
    var evidence =
        doc(
            "In-network remote visits do not require a referral. The copay is 15 dollars per visit.");
    var answer =
        policy.apply(
            "Do remote visits require a referral and what is the copay?",
            List.of(evidence),
            "No, in-network remote visits do not require a referral, and the copay is $15 per visit [evidence].");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER);
    assertThat(answer.text()).startsWith("No,");
  }

  @Test
  void rejectsDroppedOrBorrowedNegation() {
    var evidence =
        doc("Remote visits do not require a referral. Office visits require a referral.");
    for (String text :
        List.of(
            "Remote visits require a referral [evidence].",
            "Office visits do not require a referral [evidence].")) {
      var answer =
          policy.apply(
              "Do remote visits or office visits require a referral?", List.of(evidence), text);
      assertThat(answer.decision()).as(text).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
    }
  }

  @Test
  void rejectsNegationBorrowedFromQuestionOrUnrelatedEvidence() {
    var evidence = doc("Remote visits require a referral. A receipt is not required.");
    var answer =
        policy.apply(
            "Do remote visits not require a referral?",
            List.of(evidence),
            "No, remote visits do not require a referral [evidence].");
    assertThat(answer.decision()).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
  }

  @Test
  void rejectsPolarityChangesEvenWhenAllPositiveWordsAreSupported() {
    var evidence = doc("Portal referrals are not mandatory. Office referrals are mandatory.");
    for (String text :
        List.of(
            "Portal referrals are mandatory [evidence].",
            "Office referrals are not mandatory [evidence].")) {
      assertThat(policy.apply("What is the referral policy?", List.of(evidence), text).decision())
          .as(text)
          .isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    assertThat(
            policy
                .apply(
                    "What is the referral policy?",
                    List.of(evidence),
                    "Portal referrals are not mandatory [evidence].")
                .decision())
        .isEqualTo(GroundingDecision.MODEL_ANSWER);
  }

  @Test
  void rejectsDanglingConditionAndMissingRouteEvenWithTrustedCitation() {
    var evidence =
        doc(
            "A priority one incident must be acknowledged within 15 minutes. If unresolved after 30 minutes, route OPAL-P2 pages the incident commander.");
    for (String text :
        List.of(
            "A priority one incident must be acknowledged within 15 minutes, and if unresolved",
            "A priority one incident must be acknowledged within 15 minutes, and if unresolved [evidence]",
            "A priority one incident must be acknowledged within 15 minutes [evidence].")) {
      var answer =
          policy.apply(
              "How quickly is a priority one incident acknowledged and which route escalates it?",
              List.of(evidence),
              text);
      assertThat(answer.decision()).as(text).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    var complete =
        policy.apply(
            "How quickly is a priority one incident acknowledged and which route escalates it?",
            List.of(evidence),
            "A priority one incident must be acknowledged within 15 minutes, and if unresolved after 30 minutes, route OPAL-P2 pages the incident commander [evidence].");
    assertThat(complete.decision()).isEqualTo(GroundingDecision.MODEL_ANSWER);
  }

  @Test
  void rejectsChangedAmountsAndIdentifiersAfterAddressNormalization() {
    var evidence = doc("Glass claims use form G-42 and have a deductible of 75 dollars.");
    for (String text :
        List.of(
            "You have a deductible of $175 and use form G-42 [evidence].",
            "You use form G-24 and have a deductible of $75 [evidence].")) {
      var answer =
          policy.apply(
              "Which form do I use and what deductible do I have?", List.of(evidence), text);
      assertThat(answer.decision()).as(text).isEqualTo(GroundingDecision.EXTRACTIVE_FALLBACK);
    }
  }
}
