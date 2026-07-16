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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LogitBatchTest {

  @Test
  void exposesRowsWithoutChangingTheirFlatStorageOrder() {
    var batch = new LogitBatch(2, 3, new float[] {1.0f, 5.0f, 2.0f, 7.0f, 3.0f, 4.0f});

    assertThat(batch.tokenCount()).isEqualTo(2);
    assertThat(batch.vocabularySize()).isEqualTo(3);
    assertThat(batch.logit(1, 2)).isEqualTo(4.0f);
    assertThat(batch.argmax(0)).isEqualTo(1);
    assertThat(batch.argmax(1)).isZero();
    assertThat(batch.copyRow(1)).containsExactly(7.0f, 3.0f, 4.0f);
  }

  @Test
  void rejectsDimensionsThatDoNotMatchStorage() {
    assertThatThrownBy(() -> new LogitBatch(2, 3, new float[5]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("6");
  }

  @Test
  void rejectsOutOfRangeRowsAndTokens() {
    var batch = new LogitBatch(1, 2, new float[] {1.0f, 2.0f});

    assertThatThrownBy(() -> batch.logit(1, 0)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> batch.logit(0, 2)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  void snapshotCopiesActiveRowsFromReusableBackingStorage() {
    float[] reusable = {1.0f, 2.0f, 99.0f, 98.0f};
    var transientBatch = new LogitBatch(1, 2, reusable);

    LogitBatch snapshot = transientBatch.snapshot();
    reusable[0] = 7.0f;

    assertThat(snapshot.copyRow(0)).containsExactly(1.0f, 2.0f);
    assertThat(transientBatch.copyRow(0)).containsExactly(7.0f, 2.0f);
  }
}
