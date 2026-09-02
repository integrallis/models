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
package com.integrallis.models.backend.purejava.mobilemoe;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MobileMoeMathTest {

  @Test
  void normalizesEachAttentionHeadWithoutLearnedWeights() {
    float[] values = {3.0f, 4.0f, 0.0f, 5.0f};

    MobileMoeMath.normalizeHeads(values, 2, 1.0e-5f);

    assertThat(values[0]).isCloseTo(0.8485278f, within(1.0e-6f));
    assertThat(values[1]).isCloseTo(1.1313704f, within(1.0e-6f));
    assertThat(values[2]).isCloseTo(0.0f, within(1.0e-6f));
    assertThat(values[3]).isCloseTo(1.4142130f, within(1.0e-6f));
  }
}
