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

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SamplingOptionsTest {

  @Nested
  static class Defaults {

    @Test
    void builderProducesCorrectDefaults() {
      SamplingOptions opts = SamplingOptions.builder().build();

      assertThat(opts.temperature()).isEqualTo(1.0f);
      assertThat(opts.topP()).isEqualTo(0.9f);
      assertThat(opts.topK()).isEqualTo(40);
      assertThat(opts.maxTokens()).isEqualTo(256);
      assertThat(opts.seed()).isNull();
      assertThat(opts.repetitionPenalty()).isEqualTo(1.0f);
    }
  }

  @Nested
  static class BuilderOverrides {

    @Test
    void overridesAllFields() {
      SamplingOptions opts =
          SamplingOptions.builder()
              .temperature(0.7f)
              .topP(0.95f)
              .topK(50)
              .maxTokens(100)
              .seed(42L)
              .repetitionPenalty(1.1f)
              .build();

      assertThat(opts.temperature()).isEqualTo(0.7f);
      assertThat(opts.topP()).isEqualTo(0.95f);
      assertThat(opts.topK()).isEqualTo(50);
      assertThat(opts.maxTokens()).isEqualTo(100);
      assertThat(opts.seed()).isEqualTo(42L);
      assertThat(opts.repetitionPenalty()).isEqualTo(1.1f);
    }

    @Test
    void zeroTemperatureAllowed() {
      SamplingOptions opts = SamplingOptions.builder().temperature(0.0f).build();
      assertThat(opts.temperature()).isZero();
    }
  }

  @Nested
  static class Validation {

    @Test
    void negativeTemperatureRejected() {
      assertThatThrownBy(() -> SamplingOptions.builder().temperature(-0.1f).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("temperature");
    }

    @Test
    void zeroTopKRejected() {
      assertThatThrownBy(() -> SamplingOptions.builder().topK(0).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("topK");
    }

    @Test
    void negativeTopKRejected() {
      assertThatThrownBy(() -> SamplingOptions.builder().topK(-1).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("topK");
    }

    @Test
    void zeroTopPRejected() {
      assertThatThrownBy(() -> SamplingOptions.builder().topP(0.0f).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("topP");
    }

    @Test
    void topPAboveOneRejected() {
      assertThatThrownBy(() -> SamplingOptions.builder().topP(1.1f).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("topP");
    }

    @Test
    void zeroMaxTokensRejected() {
      assertThatThrownBy(() -> SamplingOptions.builder().maxTokens(0).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("maxTokens");
    }

    @Test
    void repetitionPenaltyBelowOneRejected() {
      assertThatThrownBy(() -> SamplingOptions.builder().repetitionPenalty(0.9f).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("repetitionPenalty");
    }
  }
}
