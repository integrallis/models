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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class HostedApiPricingTest {

  @Test
  void pricesUncachedCachedAndOutputTokensSeparately() {
    HostedApiPricing pricing =
        new HostedApiPricing(
            "openai",
            "gpt-test",
            0.20,
            0.02,
            0.20,
            1.25,
            "2026-07-23",
            "https://example.test/pricing");

    assertThat(pricing.estimateUsd(1_000, 250, 0, 100)).isEqualTo(0.00028);
  }

  @Test
  void rejectsCacheTokenCountsOutsideTheTotalInput() {
    HostedApiPricing pricing =
        new HostedApiPricing("provider", "model", 1, 1, 1, 1, "2026-07-23", "https://example.test");

    assertThatThrownBy(() -> pricing.estimateUsd(10, 8, 3, 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("cache");
  }
}
