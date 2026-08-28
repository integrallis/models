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
package com.integrallis.models.backend.purejava.cact;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Needle2ProbeHeadTest {

  @Test
  void probePoolingAndProjectionMatchTheReferenceFormula() {
    Needle2ProbeHead head =
        new Needle2ProbeHead(
            2,
            1,
            1,
            new float[] {0.0f, 0.0f},
            new float[] {0.5f, 0.25f},
            new float[] {1.0f},
            false);
    Needle2ProbeHead.Accumulator accumulator = head.newAccumulator();

    accumulator.accept(new float[] {1.0f, 3.0f});
    accumulator.accept(new float[] {3.0f, 5.0f});

    assertThat(accumulator.finish()).containsExactly(new float[] {3.0f}, within(1.0e-6f));
  }

  @Test
  void contrastiveProjectionIsUnitNormalized() {
    Needle2ProbeHead head =
        new Needle2ProbeHead(
            2,
            1,
            2,
            new float[] {0.0f, 0.0f},
            new float[] {1.0f, 0.0f, 0.0f, 1.0f},
            new float[] {0.0f, 0.0f},
            true);
    Needle2ProbeHead.Accumulator accumulator = head.newAccumulator();

    accumulator.accept(new float[] {1.0f, 3.0f});
    accumulator.accept(new float[] {3.0f, 5.0f});

    assertThat(accumulator.finish())
        .containsExactly(new float[] {0.4472136f, 0.8944272f}, within(1.0e-6f));
  }

  @Test
  void onlineSoftmaxRemainsStableForLargeProbeScores() {
    Needle2ProbeHead head =
        new Needle2ProbeHead(
            1, 1, 1, new float[] {1000.0f}, new float[] {1.0f}, new float[] {0.0f}, false);
    Needle2ProbeHead.Accumulator accumulator = head.newAccumulator();

    accumulator.accept(new float[] {1.0f});
    accumulator.accept(new float[] {2.0f});

    assertThat(accumulator.finish()[0]).isFinite().isCloseTo(2.0f, within(1.0e-6f));
  }
}
