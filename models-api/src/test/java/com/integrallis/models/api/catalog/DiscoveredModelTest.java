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
package com.integrallis.models.api.catalog;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Validation tests for what a catalog may report.
 *
 * <p>A catalog's whole purpose is to replace hand-written guesses with measurements, so a figure
 * that cannot be a measurement — negative latency, zero throughput, a success rate above one — is
 * rejected at construction rather than allowed to reach a router that would score on it.
 */
@Tag("unit")
class DiscoveredModelTest {

  private static DiscoveredModel model(double successRate, int contextWindow) {
    return new DiscoveredModel(
        "m",
        true,
        Set.of("code"),
        contextWindow,
        0.0,
        0.0,
        1024L,
        new DiscoveredModel.Performance(100, 30.0),
        Map.of("code", 0.8),
        successRate);
  }

  @Test
  void keepsWhatACatalogMeasured() {
    DiscoveredModel model = model(0.99, 4096);

    assertThat(model.id()).isEqualTo("m");
    assertThat(model.local()).isTrue();
    assertThat(model.contextWindow()).isEqualTo(4096);
    assertThat(model.performance().timeToFirstTokenMillis()).isEqualTo(100);
    assertThat(model.performance().tokensPerSecond()).isEqualTo(30.0);
    assertThat(model.quality()).containsEntry("code", 0.8);
  }

  @Test
  void treatsAbsentPerformanceAsUnmeasuredRatherThanZero() {
    DiscoveredModel model =
        new DiscoveredModel("m", true, Set.of(), 4096, 0.0, 0.0, 0L, null, Map.of(), 1.0);

    // Null means nothing was measured on this hardware. A zeroed Performance would read as a model
    // that answers instantly, which is the opposite of the truth.
    assertThat(model.performance()).isNull();
  }

  @Test
  void copiesCollectionsSoACatalogCannotMutateWhatItReported() {
    Set<String> tags = new HashSet<>(Set.of("code"));
    Map<String, Double> quality = new HashMap<>(Map.of("code", 0.8));
    DiscoveredModel model =
        new DiscoveredModel(
            "m",
            true,
            tags,
            4096,
            0.0,
            0.0,
            1024L,
            new DiscoveredModel.Performance(100, 30.0),
            quality,
            1.0);

    tags.add("sql");
    quality.put("sql", 0.9);

    assertThat(model.tags()).containsExactly("code");
    assertThat(model.quality()).containsOnlyKeys("code");
  }

  @Test
  void rejectsDescriptionsThatCannotBeTrue() {
    assertThatThrownBy(() -> model(1.5, 4096))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("successRate");
    assertThatThrownBy(() -> model(-0.1, 4096)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> model(1.0, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("contextWindow");
    assertThatThrownBy(
            () -> new DiscoveredModel(" ", true, Set.of(), 10, 0, 0, 0L, null, Map.of(), 1.0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blank");
    assertThatThrownBy(
            () -> new DiscoveredModel("m", false, Set.of(), 10, -1, 0, 0L, null, Map.of(), 1.0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("costs");
  }

  @Test
  void rejectsPerformanceThatCannotBeAMeasurement() {
    assertThatThrownBy(() -> new DiscoveredModel.Performance(-1, 10.0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("timeToFirstTokenMillis");
    assertThatThrownBy(() -> new DiscoveredModel.Performance(10, 0.0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("tokensPerSecond");
    assertThatThrownBy(() -> new DiscoveredModel.Performance(10, -5.0))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNullsRatherThanCarryingThem() {
    assertThatThrownBy(
            () -> new DiscoveredModel("m", true, null, 10, 0, 0, 0L, null, Map.of(), 1.0))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(
            () -> new DiscoveredModel("m", true, Set.of(), 10, 0, 0, 0L, null, null, 1.0))
        .isInstanceOf(NullPointerException.class);
  }
}
