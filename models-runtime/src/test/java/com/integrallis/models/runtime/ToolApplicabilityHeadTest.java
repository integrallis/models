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

class ToolApplicabilityHeadTest {

  @Test
  void reproducesTheFrozenFloatProductAndOrderedDoubleAccumulation() {
    ToolApplicabilityHead head =
        new ToolApplicabilityHead(
            new float[] {1, 2}, new float[] {2, 4}, new float[] {0.5f, -0.25f}, 0.125f);

    ToolApplicabilityScore score = head.score(new float[] {3, 6});

    assertThat(score.score()).isEqualTo(0.375);
    assertThat(score.shouldCall()).isTrue();
  }

  @Test
  void rejectsMismatchedOrNonFiniteArtifactsAndHiddenStates() {
    assertThatThrownBy(
            () ->
                new ToolApplicabilityHead(new float[] {0}, new float[] {1, 2}, new float[] {1}, 0))
        .isInstanceOf(IllegalArgumentException.class);
    ToolApplicabilityHead head =
        new ToolApplicabilityHead(new float[] {0}, new float[] {1}, new float[] {1}, 0);
    assertThatThrownBy(() -> head.score(new float[] {Float.NaN}))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
