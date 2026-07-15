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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.ops.TensorOps;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RopeTableTest {

  @Test
  void reusesFactorsForEveryHeadAtTheSamePosition() {
    RopeTable table = new RopeTable(4, 10_000.0f, 0.25f);
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
    RopeTable table = new RopeTable(4, 10_000.0f, 0.25f);
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
    RopeTable table = new RopeTable(4, 10_000.0f, 0.25f);
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
}
