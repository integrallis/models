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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.runtime.ActivatedToolDecisionPolicy;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class ActivatedHybridLiveDecisionCliTest {

  @Test
  void freezesTheV21PolicyFromExposedV20Margins() {
    assertThat(ActivatedHybridLiveDecisionCli.POLICY.callTokenId()).isEqualTo(4913);
    assertThat(ActivatedHybridLiveDecisionCli.POLICY.noCallTokenId()).isEqualTo(19536);
    assertThat(ActivatedHybridLiveDecisionCli.POLICY.primarySpecialistMargin()).isEqualTo(4.5f);
    assertThat(ActivatedHybridLiveDecisionCli.POLICY.rescueSpecialistMargin())
        .isEqualTo(1.3521204f);
  }

  @Test
  void requiresAllCasesAndBothFrozenClassFloors() {
    List<ActivatedHybridLiveDecisionCli.Observation> passing = new ArrayList<>();
    for (int index = 0; index < 50; index++) {
      passing.add(observation("call-" + index, true, index < 48, true));
    }
    for (int index = 0; index < 25; index++) {
      passing.add(observation("none-" + index, false, index == 0, true));
    }

    assertThat(ActivatedHybridLiveDecisionCli.summarize(passing).passed()).isTrue();
    assertThat(ActivatedHybridLiveDecisionCli.summarize(passing.subList(0, 74)).passed()).isFalse();

    List<ActivatedHybridLiveDecisionCli.Observation> missedCall = new ArrayList<>(passing);
    missedCall.set(47, observation("call-47", true, false, true));
    assertThat(ActivatedHybridLiveDecisionCli.summarize(missedCall).passed()).isFalse();

    List<ActivatedHybridLiveDecisionCli.Observation> falseCall = new ArrayList<>(passing);
    falseCall.set(51, observation("none-1", false, true, true));
    assertThat(ActivatedHybridLiveDecisionCli.summarize(falseCall).passed()).isFalse();

    List<ActivatedHybridLiveDecisionCli.Observation> copiedPrefix = new ArrayList<>(passing);
    copiedPrefix.set(0, observation("call-0", true, true, false));
    assertThat(ActivatedHybridLiveDecisionCli.summarize(copiedPrefix).passed()).isFalse();
  }

  private static ActivatedHybridLiveDecisionCli.Observation observation(
      String id, boolean expected, boolean called, boolean physical) {
    return new ActivatedHybridLiveDecisionCli.Observation(
        id,
        expected ? "simple" : "irrelevance",
        expected,
        called
            ? ActivatedToolDecisionPolicy.Decision.CALL_PRIMARY
            : ActivatedToolDecisionPolicy.Decision.NO_CALL,
        40,
        called ? 6 : 0,
        physical,
        10);
  }
}
