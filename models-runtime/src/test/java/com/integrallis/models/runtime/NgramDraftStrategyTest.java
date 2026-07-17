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

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class NgramDraftStrategyTest {

  @Test
  void usesARequestScaleCooldownAfterOneWeakVerificationWindow() {
    SpeculativeGenerationOptions options = SpeculativeGenerationOptions.builder().build();

    assertThat(options.adaptationWindow()).isEqualTo(1);
    assertThat(options.cooldownTokens()).isEqualTo(256);
  }

  @Test
  void draftsTheContinuationOfTheMostRecentPriorSuffix() {
    NgramDraftStrategy strategy =
        new NgramDraftStrategy(
            SpeculativeGenerationOptions.builder()
                .ngramSize(4)
                .minimumDraftTokens(3)
                .maximumDraftTokens(3)
                .build());

    int[] draft = strategy.propose(List.of(2, 3, 4, 5, 2, 3, 4), 5, 3);

    assertThat(draft).containsExactly(2, 3, 4);
  }

  @Test
  void expandsPastTheMinimumDraftOnlyAfterAFullConfidenceProbe() {
    NgramDraftStrategy strategy =
        new NgramDraftStrategy(
            SpeculativeGenerationOptions.builder()
                .ngramSize(4)
                .minimumDraftTokens(3)
                .maximumDraftTokens(7)
                .build());
    List<Integer> history = List.of(2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4);

    int[] probe = strategy.propose(history, 5, 7);
    strategy.recordVerification(probe.length, probe.length, 1);
    int[] trusted = strategy.propose(history, 5, 7);

    assertThat(probe).containsExactly(6, 7, 8);
    assertThat(trusted).containsExactly(6, 7, 8, 9, 2, 3, 4);
  }

  @Test
  void suppressesDraftingTemporarilyAfterAWeakAcceptanceWindow() {
    NgramDraftStrategy strategy =
        new NgramDraftStrategy(
            SpeculativeGenerationOptions.builder()
                .adaptationWindow(1)
                .minimumAcceptanceRate(0.75f)
                .cooldownTokens(5)
                .build());

    strategy.recordVerification(4, 1, 7);

    assertThat(strategy.isSuppressed(7)).isTrue();
    assertThat(strategy.isSuppressed(11)).isTrue();
    assertThat(strategy.isSuppressed(12)).isFalse();
  }
}
