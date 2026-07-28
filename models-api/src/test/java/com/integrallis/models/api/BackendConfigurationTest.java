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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BackendConfigurationTest {

  @Test
  void snapshotsProfileInputsAndEnrichesBackendDiagnostics() {
    Map<String, String> environment = new HashMap<>(Map.of("profile", "measured"));
    Map<String, String> recommendations = new HashMap<>(Map.of("models.test.setting", "true"));
    List<OptimizationDecision> decisions =
        new ArrayList<>(
            List.of(
                new OptimizationDecision(
                    "profile.measured",
                    OptimizationStatus.ENABLED,
                    "the measured runtime matches",
                    Map.of("source", "catalog"))));
    BackendConfiguration configuration =
        new BackendConfiguration(environment, recommendations, decisions);

    environment.clear();
    recommendations.clear();
    decisions.clear();

    assertThat(configuration.environment()).containsEntry("profile", "measured");
    assertThat(configuration.recommendations()).containsEntry("models.test.setting", "true");
    assertThat(
            configuration.enrich(
                new BackendDiagnostics("test", "v1", Map.of("runtime", "java"), List.of())))
        .satisfies(
            diagnostics -> {
              assertThat(diagnostics.environment())
                  .containsEntry("runtime", "java")
                  .containsEntry("profile", "measured");
              assertThat(diagnostics.optimization("profile.measured")).isPresent();
            });
  }

  @Test
  void rejectsAProfileThatDuplicatesAnExistingOptimizationDecision() {
    BackendConfiguration configuration =
        new BackendConfiguration(
            Map.of(),
            Map.of(),
            List.of(
                new OptimizationDecision(
                    "duplicate", OptimizationStatus.ENABLED, "profile decision", Map.of())));
    BackendDiagnostics diagnostics =
        new BackendDiagnostics(
            "test",
            "v1",
            Map.of(),
            List.of(
                new OptimizationDecision(
                    "duplicate", OptimizationStatus.DISABLED, "runtime decision", Map.of())));

    assertThatThrownBy(() -> configuration.enrich(diagnostics))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("duplicate optimization id");
  }
}
