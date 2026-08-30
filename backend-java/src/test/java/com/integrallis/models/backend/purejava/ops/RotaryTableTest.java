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
package com.integrallis.models.backend.purejava.ops;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Offset.offset;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RotaryTableTest {

  @Test
  void reusesFactorsForEveryHeadAtTheSamePosition() {
    RotaryTable table = new RotaryTable(4, 10_000.0f, 0.25f);
    float[] first = {1.0f, 2.0f, 3.0f, 4.0f};
    float[] second = {5.0f, 6.0f, 7.0f, 8.0f};

    table.prepare(9);
    table.apply(first, 0, false);
    table.prepare(9);
    table.apply(second, 0, false);

    assertThat(table.preparationCount()).isEqualTo(1);
  }

  @Test
  void cachedFactorsMatchDirectStandardAndNeoxRope() {
    RotaryTable table = new RotaryTable(4, 10_000.0f, 0.25f);
    float[] standard = {99.0f, 1.0f, 2.0f, 3.0f, 4.0f, 98.0f};
    float[] expectedStandard = standard.clone();
    float[] neox = {97.0f, 5.0f, 6.0f, 7.0f, 8.0f, 96.0f};
    float[] expectedNeox = neox.clone();

    TensorOps.rope(expectedStandard, 1, 9, 4, 10_000.0f, 0.25f);
    TensorOps.ropeNeox(expectedNeox, 1, 9, 4, 10_000.0f, 0.25f);
    table.prepare(9);
    table.apply(standard, 1, false);
    table.apply(neox, 1, true);

    assertThat(standard).containsExactly(expectedStandard);
    assertThat(neox).containsExactly(expectedNeox);
  }

  @Test
  void batchFactorsArePreparedOnceAndAddressedByToken() {
    RotaryTable table = new RotaryTable(4, 10_000.0f, 0.25f);
    float[] first = {1.0f, 2.0f, 3.0f, 4.0f};
    float[] second = {5.0f, 6.0f, 7.0f, 8.0f};
    float[] expectedFirst = first.clone();
    float[] expectedSecond = second.clone();
    TensorOps.rope(expectedFirst, 0, 9, 4, 10_000.0f, 0.25f);
    TensorOps.rope(expectedSecond, 0, 10, 4, 10_000.0f, 0.25f);

    table.prepareBatch(9, 2);
    table.applyBatch(first, 0, 0, false);
    table.applyBatch(second, 0, 1, false);

    assertThat(first).containsExactly(expectedFirst);
    assertThat(second).containsExactly(expectedSecond);
    assertThat(table.preparationCount()).isEqualTo(2);
  }

  @Test
  void ggufFrequencyFactorsPreserveGemma4ProportionalRopeLayout() {
    RotaryTable table =
        new RotaryTable(8, 1_000_000.0f, 1.0f, new float[] {1.0f, 1.0e30f, 1.0e30f, 1.0e30f});
    float[] head = {1, 2, 3, 4, 5, 6, 7, 8};

    table.prepare(2);
    table.apply(head, 0, true);

    assertThat(head[0]).isCloseTo(-4.9626341f, offset(1.0e-6f));
    assertThat(head[4]).isCloseTo(-1.1714368f, offset(1.0e-6f));
    assertThat(head).containsExactly(head[0], 2, 3, 4, head[4], 6, 7, 8);
  }

  @Test
  void frequencyFactorsMustMatchTheRotaryPairCountAndRemainPositive() {
    assertThatThrownBy(() -> new RotaryTable(8, 10_000.0f, 1.0f, new float[] {1, 1}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("frequencyFactors");
    assertThatThrownBy(
            () -> new RotaryTable(4, 10_000.0f, 1.0f, new float[] {1, Float.POSITIVE_INFINITY}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("frequencyFactors[1]");
  }

  @Test
  void yarnMatchesPinnedFreeTokenGptOssFrequenciesAndMagnitude() {
    RotaryTable table = RotaryTable.yarn(64, 150_000.0f, 32.0f, 32.0f, 1.0f, 4_096, false);
    float[] head = new float[64];
    for (int index = 0; index < head.length; index++) {
      head[index] = index + 1.0f;
    }

    table.prepare(8_192);
    table.apply(head, 0, true);

    // FreeToken bd372b6, layers/rotary.py non-truncating YaRN branch.
    assertThat(head[0]).isCloseTo(42.88368f, offset(0.002f));
    assertThat(head[15]).isCloseTo(-61.43231f, offset(0.002f));
    assertThat(head[31]).isCloseTo(42.87677f, offset(0.002f));
    assertThat(head[32]).isCloseTo(11.72366f, offset(0.002f));
    assertThat(head[47]).isCloseTo(-29.46214f, offset(0.002f));
    assertThat(head[63]).isCloseTo(86.28717f, offset(0.002f));
  }

  @Test
  void yarnAppliesItsAttentionMagnitudeAtPositionZero() {
    RotaryTable table = RotaryTable.yarn(4, 10_000.0f, 32.0f, 32.0f, 1.0f, 4_096, false);
    float[] head = {1.0f, 2.0f, 3.0f, 4.0f};

    table.prepare(0);
    table.apply(head, 0, true);

    float attentionFactor = 0.1f * (float) Math.log(32.0f) + 1.0f;
    assertThat(head)
        .containsExactly(
            new float[] {
              attentionFactor, 2 * attentionFactor, 3 * attentionFactor, 4 * attentionFactor
            },
            offset(1.0e-6f));
  }
}
