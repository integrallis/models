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

import org.junit.jupiter.api.Test;

class GenerationResultTest {

  @Test
  void derivesInterTokenLatencyAndDecodeThroughput() {
    GenerationResult result = new GenerationResult("answer", 100, 11, 250, 1_250, 400, 50);

    assertThat(result.tpotMillis()).isEqualTo(100);
    assertThat(result.decodeTokensPerSecond()).isEqualTo(10);
  }

  @Test
  void recordsNormalizedHostedUsageAndCost() {
    GenerationResult result =
        new GenerationResult("answer", 100, 20, 10, 11, 250, 1_250, 0, 0, 0, 0, 0.00028);

    assertThat(result.inputTokens()).isEqualTo(100);
    assertThat(result.cacheReadInputTokens()).isEqualTo(20);
    assertThat(result.cacheWriteInputTokens()).isEqualTo(10);
    assertThat(result.outputTokens()).isEqualTo(11);
    assertThat(result.estimatedApiCostUsd()).isEqualTo(0.00028);
  }
}
