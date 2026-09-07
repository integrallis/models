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

import java.time.Duration;
import org.junit.jupiter.api.Test;

class ContinuousBatchingOptionsTest {

  @Test
  void requiresAnExplicitBatchSizeAndSuppliesBoundedLowLatencyDefaults() {
    assertThatThrownBy(() -> ContinuousBatchingOptions.builder().build())
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("maximumBatchSize");

    ContinuousBatchingOptions options =
        ContinuousBatchingOptions.builder().maximumBatchSize(4).build();

    assertThat(options.maximumBatchSize()).isEqualTo(4);
    assertThat(options.maximumQueuedRequests()).isEqualTo(128);
    assertThat(options.maximumPrefillChunkTokens()).isEqualTo(128);
    assertThat(options.batchPrefillAcrossSessions()).isFalse();
    assertThat(options.batchFormationDelay()).isEqualTo(Duration.ofMillis(1));
  }

  @Test
  void explicitlyEnablesCrossSessionPromptBatching() {
    ContinuousBatchingOptions options =
        ContinuousBatchingOptions.builder()
            .maximumBatchSize(4)
            .batchPrefillAcrossSessions(true)
            .build();

    assertThat(options.batchPrefillAcrossSessions()).isTrue();
  }

  @Test
  void rejectsInvalidCapacityAndDelayValues() {
    assertThatThrownBy(() -> ContinuousBatchingOptions.builder().maximumBatchSize(-1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> ContinuousBatchingOptions.builder().maximumQueuedRequests(0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> ContinuousBatchingOptions.builder().maximumPrefillChunkTokens(0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> ContinuousBatchingOptions.builder().batchFormationDelay(Duration.ofNanos(-1)))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
