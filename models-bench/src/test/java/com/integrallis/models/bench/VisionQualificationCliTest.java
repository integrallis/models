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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class VisionQualificationCliTest {
  @Test
  void greedySelectionKeepsFirstTokenOnTiesAndDoesNotMutateLogits() {
    float[] logits = {-4f, -1f, -1f, -3f};
    assertThat(VisionQualificationCli.bestToken(logits)).isEqualTo(1);
    assertThat(logits).containsExactly(-4f, -1f, -1f, -3f);
  }

  @Test
  void nonfiniteLogitsFailEvenWhenTheyCannotWinTheArgmax() {
    for (float invalid :
        new float[] {Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY}) {
      assertThatThrownBy(() -> VisionQualificationCli.bestToken(new float[] {10, invalid, 1}))
          .isInstanceOf(IllegalStateException.class)
          .hasMessage("Nonfinite logits");
    }
  }
}
