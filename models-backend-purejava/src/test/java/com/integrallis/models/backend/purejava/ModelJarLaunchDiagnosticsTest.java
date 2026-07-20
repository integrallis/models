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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.OptimizationStatus;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;
import org.modeljars.ModelPerformanceProfile;
import org.modeljars.ModelPerformanceProfileRegistry;

@Tag("unit")
class ModelJarLaunchDiagnosticsTest {

  private static final String DECISION_ID = "modeljars.profile.qwen3.0.6b.q4.0.epyc.milan.jdk25";

  @Test
  void enablesOnlyAnExactRuntimeWithEveryStartupArgument() {
    Fixture fixture = fixture();
    List<String> inputArguments = fixture.profile().javaLaunch().orElseThrow().jvmArguments();

    BackendDiagnostics diagnostics =
        ModelJarLaunchDiagnostics.enrich(
            base(),
            fixture.descriptor(),
            fixture.registry(),
            fixture.profile().runtimeSelector(),
            inputArguments);

    assertThat(diagnostics.optimization(DECISION_ID))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("recommended-runtime", "graal-jvmci")
                  .containsEntry("recommended-java-feature", "25")
                  .containsEntry("missing-jvm-arguments", "");
            });
  }

  @Test
  void reportsArgumentsThatRequireAProcessRestart() {
    Fixture fixture = fixture();
    List<String> required = fixture.profile().javaLaunch().orElseThrow().jvmArguments();

    BackendDiagnostics diagnostics =
        ModelJarLaunchDiagnostics.enrich(
            base(),
            fixture.descriptor(),
            fixture.registry(),
            fixture.profile().runtimeSelector(),
            List.of(required.getFirst()));

    assertThat(diagnostics.optimization(DECISION_ID))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(decision.reason()).contains("restart");
              assertThat(decision.settings().get("missing-jvm-arguments"))
                  .isEqualTo("-Dvectors.gguf.q4ShortPairwise=true");
            });
  }

  @Test
  void doesNotRecommendAProfileMeasuredForAnotherPlatform() {
    Fixture fixture = fixture();
    Map<String, String> runtime = new HashMap<>(fixture.profile().runtimeSelector());
    runtime.put("os", "macOS");

    BackendDiagnostics diagnostics =
        ModelJarLaunchDiagnostics.enrich(
            base(), fixture.descriptor(), fixture.registry(), runtime, List.of());

    assertThat(diagnostics.optimization(DECISION_ID))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED);
              assertThat(decision.settings().get("selector-mismatches")).contains("os");
            });
  }

  private static BackendDiagnostics base() {
    return new BackendDiagnostics("pure-java", "test-plan", Map.of(), new ArrayList<>());
  }

  private static Fixture fixture() {
    ModelJarRequirement requirement =
        ModelJarRequirement.forSource("hf://ggml-org/Qwen3-0.6B-GGUF")
            .versionRange("[3.0.0,4.0.0)")
            .variant("q4_0")
            .backend("pure-java")
            .build();
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(requirement).orElseThrow();
    ModelPerformanceProfileRegistry registry = ModelPerformanceProfileRegistry.fromClasspath();
    ModelPerformanceProfile profile =
        registry.profilesFor(descriptor).stream()
            .filter(value -> value.javaLaunch().isPresent())
            .findFirst()
            .orElseThrow();
    return new Fixture(descriptor, registry, profile);
  }

  private record Fixture(
      ModelJarDescriptor descriptor,
      ModelPerformanceProfileRegistry registry,
      ModelPerformanceProfile profile) {}
}
