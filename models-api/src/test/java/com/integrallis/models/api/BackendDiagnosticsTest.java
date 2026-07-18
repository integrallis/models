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

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BackendDiagnosticsTest {

  @Test
  void exposesImmutableOptimizationDecisionsById() {
    Map<String, String> environment = new java.util.HashMap<>(Map.of("compiler", "hotspot-c2"));
    List<OptimizationDecision> optimizations =
        new java.util.ArrayList<>(
            List.of(
                new OptimizationDecision(
                    "batched-prefill",
                    OptimizationStatus.ENABLED,
                    "all projection tensors have batched kernels",
                    Map.of("batch-size", "32"))));

    BackendDiagnostics diagnostics =
        new BackendDiagnostics("pure-java", "pure-java-v1", environment, optimizations);
    environment.put("compiler", "changed");
    optimizations.clear();

    assertThat(diagnostics.environment()).containsEntry("compiler", "hotspot-c2");
    assertThat(diagnostics.optimizations()).hasSize(1);
    assertThat(diagnostics.optimization("batched-prefill")).isPresent();
    assertThat(diagnostics.optimization("unknown")).isEmpty();
  }
}
