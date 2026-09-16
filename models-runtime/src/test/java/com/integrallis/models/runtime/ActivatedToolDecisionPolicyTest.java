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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class ActivatedToolDecisionPolicyTest {
  private static final int CALL = 4_913;
  private static final int NO_CALL = 19_536;
  private static final ActivatedToolDecisionPolicy POLICY =
      new ActivatedToolDecisionPolicy(CALL, NO_CALL, 4.8f, 1.3f, 33.7f, 39.4f);

  @Test
  void acceptsTheSpecialistPrimaryDecision() {
    assertThat(POLICY.decide(score(20, 0), score(5, 0)))
        .isEqualTo(ActivatedToolDecisionPolicy.Decision.CALL_PRIMARY);
  }

  @Test
  void rescuesAQualifiedBaseSpecialistDisagreement() {
    assertThat(POLICY.decide(score(38, 0), score(2, 0)))
        .isEqualTo(ActivatedToolDecisionPolicy.Decision.CALL_RESCUE);
  }

  @Test
  void abstainsWhenTheBaseMarginExceedsTheFalseCallCeiling() {
    assertThat(POLICY.decide(score(40, 0), score(2, 0)))
        .isEqualTo(ActivatedToolDecisionPolicy.Decision.NO_CALL);
  }

  @Test
  void abstainsWhenTheSpecialistOrDisagreementDoesNotClearItsStrictBoundary() {
    assertThat(POLICY.decide(score(38, 0), score(1.3f, 0)))
        .isEqualTo(ActivatedToolDecisionPolicy.Decision.NO_CALL);
    assertThat(POLICY.decide(score(35, 0), score(1.3f, 0)))
        .isEqualTo(ActivatedToolDecisionPolicy.Decision.NO_CALL);
  }

  @Test
  void rejectsScoresForAnotherDecisionVocabulary() {
    ToolDecisionScore incompatible = new ToolDecisionScore(7, 2, 8, 0);

    assertThatThrownBy(() -> POLICY.decide(incompatible, score(5, 0)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("token IDs");
  }

  @Test
  void rejectsAnImpossibleOrNonfinitePolicy() {
    assertThatThrownBy(
            () -> new ActivatedToolDecisionPolicy(CALL, NO_CALL, Float.NaN, 1.3f, 33.7f, 39.4f))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> new ActivatedToolDecisionPolicy(CALL, NO_CALL, 4.8f, 4.8f, 33.7f, 39.4f))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> new ActivatedToolDecisionPolicy(CALL, NO_CALL, 4.8f, 1.3f, 38.2f, 39.4f))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static ToolDecisionScore score(float callLogit, float noCallLogit) {
    return new ToolDecisionScore(CALL, callLogit, NO_CALL, noCallLogit);
  }
}
