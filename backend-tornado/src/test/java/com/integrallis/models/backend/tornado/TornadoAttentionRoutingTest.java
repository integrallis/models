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
package com.integrallis.models.backend.tornado;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TornadoAttentionRoutingTest {

  @Test
  void aKernelThatNeverRanSaysSoAndNamesTheGateThatClosed() {
    Map<String, Long> reasons = new LinkedHashMap<>();
    reasons.put("sliding-window attention is not supported", 40L);

    TornadoAttentionRouting routing = new TornadoAttentionRouting(0, 0, 40, 0, 0, 0, 0.0, reasons);

    assertThat(routing.ranOnDevice()).isFalse();
    assertThat(routing.deviceLayerSteps()).isZero();
    assertThat(routing.summary())
        .contains("device-layer-steps=0")
        .contains("refusals=40")
        .contains("sliding-window attention is not supported=40");
  }

  @Test
  void theSummaryOrdersRefusalsByHowOftenTheyFired() {
    Map<String, Long> reasons = new LinkedHashMap<>();
    reasons.put("rare", 1L);
    reasons.put("common", 99L);

    String summary = new TornadoAttentionRouting(1, 2, 100, 3, 4, 5, 6.0, reasons).summary();

    assertThat(summary.indexOf("common=99")).isLessThan(summary.indexOf("rare=1"));
    assertThat(summary).contains("device-layer-steps=3").contains("decode=1").contains("prefill=2");
  }

  @Test
  void aKernelThatRanReportsBothPathsAndOmitsAnEmptyRefusalList() {
    TornadoAttentionRouting routing =
        new TornadoAttentionRouting(12, 3, 0, 800, 1, 4, 2.5, new LinkedHashMap<>());

    assertThat(routing.ranOnDevice()).isTrue();
    assertThat(routing.deviceLayerSteps()).isEqualTo(15);
    assertThat(routing.summary()).doesNotContain("refused:");
  }

  @Test
  void theRefusalHistogramIsAnImmutableSnapshot() {
    Map<String, Long> reasons = new LinkedHashMap<>();
    reasons.put("a", 1L);
    TornadoAttentionRouting routing = new TornadoAttentionRouting(0, 0, 1, 0, 0, 0, 0.0, reasons);
    reasons.put("b", 2L);

    assertThat(routing.refusalReasons()).containsOnlyKeys("a");
    assertThatThrownBy(() -> routing.refusalReasons().put("c", 3L))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  void theAbsentRoutingIsEmpty() {
    assertThat(TornadoAttentionRouting.none().ranOnDevice()).isFalse();
    assertThat(TornadoAttentionRouting.none().refusalReasons()).isEmpty();
    assertThat(TornadoAttentionRouting.none().planCount()).isZero();
  }
}
