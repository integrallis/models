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
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import java.util.List;
import org.junit.jupiter.api.Test;

class VirtualModelQualificationComparisonCliTest {

  @Test
  void qualifiesOnlyARepeatedCorrectMaterialImprovement() {
    var result =
        VirtualModelQualificationComparisonCli.evaluate(
            List.of(150_000L, 140_000L, 145_000L),
            List.of(110_000L, 105_000L, 108_000L),
            true,
            0.05);

    assertThat(result.controlMedianMillis()).isEqualTo(145_000L);
    assertThat(result.hybridMedianMillis()).isEqualTo(108_000L);
    assertThat(result.improvement()).isCloseTo(0.25517, within(0.00001));
    assertThat(result.qualified()).isTrue();
  }

  @Test
  void rejectsAnIncorrectOrMarginalComposite() {
    assertThat(
            VirtualModelQualificationComparisonCli.evaluate(
                    List.of(100L, 101L, 102L), List.of(90L, 91L, 92L), false, 0.05)
                .qualified())
        .isFalse();
    assertThat(
            VirtualModelQualificationComparisonCli.evaluate(
                    List.of(100L, 101L, 102L), List.of(98L, 99L, 100L), true, 0.05)
                .qualified())
        .isFalse();
  }

  @Test
  void requiresAtLeastThreeFreshProcessesPerArm() {
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VirtualModelQualificationComparisonCli.evaluate(
                    List.of(100L, 101L), List.of(90L, 91L), true, 0.05))
        .withMessageContaining("at least three");
  }

  private static org.assertj.core.data.Offset<Double> within(double value) {
    return org.assertj.core.data.Offset.offset(value);
  }
}
