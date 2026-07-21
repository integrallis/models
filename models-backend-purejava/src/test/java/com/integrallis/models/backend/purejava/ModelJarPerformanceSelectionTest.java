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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.OptimizationStatus;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarException;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;
import org.modeljars.ModelPerformanceProfile;
import org.modeljars.ModelPerformanceProfileRegistry;

@Tag("unit")
class ModelJarPerformanceSelectionTest {

  private static final String Q4_KERNEL = "models.purejava.q4Kernel";
  private static final String BATCHED_ATTENTION_VALUES = "models.purejava.batchedAttentionValues";
  private static final String DECISION_ID = "modeljars.profile.qwen3.0.6b.q4.0.epyc.milan.jdk25";

  @Test
  void appliesRecommendationsOnlyForAnExactRuntimeWithEveryStartupArgument() {
    Fixture fixture = fixture();
    List<String> inputArguments = fixture.profile().javaLaunch().orElseThrow().jvmArguments();

    ModelJarPerformanceSelection selection =
        ModelJarPerformanceSelection.evaluate(
            fixture.descriptor(),
            List.of(fixture.profile()),
            fixture.profile().runtimeSelector(),
            inputArguments);

    assertThat(selection.recommendations()).containsExactly(Map.entry(Q4_KERNEL, "short-pairwise"));
    assertThat(selection.enrich(base()).optimization(DECISION_ID))
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
  void mergesNonConflictingRecommendationsFromMatchingProfiles() {
    Fixture fixture = fixture();
    ModelPerformanceProfile q4Profile = fixture.profile();
    ModelPerformanceProfile batchedValuesProfile =
        new ModelPerformanceProfile(
            "qwen3_0_6b_q4_0_epyc_milan_jdk25_batched_values",
            q4Profile.modelAlias(),
            q4Profile.markerCoordinate(),
            q4Profile.artifactSha256(),
            q4Profile.backend(),
            q4Profile.runtimeSelector(),
            Map.of(BATCHED_ATTENTION_VALUES, "true"),
            q4Profile.javaLaunch(),
            q4Profile.evidence());

    ModelJarPerformanceSelection selection =
        ModelJarPerformanceSelection.evaluate(
            fixture.descriptor(),
            List.of(q4Profile, batchedValuesProfile),
            q4Profile.runtimeSelector(),
            q4Profile.javaLaunch().orElseThrow().jvmArguments());

    assertThat(selection.recommendations())
        .containsExactlyInAnyOrderEntriesOf(
            Map.of(Q4_KERNEL, "short-pairwise", BATCHED_ATTENTION_VALUES, "true"));
  }

  @Test
  void rejectsConflictingRecommendationsFromMatchingProfiles() {
    Fixture fixture = fixture();
    ModelPerformanceProfile q4Profile = fixture.profile();
    ModelPerformanceProfile conflictingProfile =
        new ModelPerformanceProfile(
            "qwen3_0_6b_q4_0_epyc_milan_jdk25_conflict",
            q4Profile.modelAlias(),
            q4Profile.markerCoordinate(),
            q4Profile.artifactSha256(),
            q4Profile.backend(),
            q4Profile.runtimeSelector(),
            Map.of(Q4_KERNEL, "widened"),
            q4Profile.javaLaunch(),
            q4Profile.evidence());

    assertThatThrownBy(
            () ->
                ModelJarPerformanceSelection.evaluate(
                    fixture.descriptor(),
                    List.of(q4Profile, conflictingProfile),
                    q4Profile.runtimeSelector(),
                    q4Profile.javaLaunch().orElseThrow().jvmArguments()))
        .isInstanceOf(ModelJarException.class)
        .hasMessageContaining("Conflicting performance recommendations")
        .hasMessageContaining(Q4_KERNEL);
  }

  @Test
  void withholdsRecommendationsWhenARequiredArgumentNeedsARestart() {
    Fixture fixture = fixture();

    ModelJarPerformanceSelection selection =
        ModelJarPerformanceSelection.evaluate(
            fixture.descriptor(),
            List.of(fixture.profile()),
            fixture.profile().runtimeSelector(),
            List.of());

    assertThat(selection.recommendations()).isEmpty();
    assertThat(selection.enrich(base()).optimization(DECISION_ID))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(decision.reason()).contains("restart");
              assertThat(decision.settings().get("missing-jvm-arguments"))
                  .isEqualTo("-Djdk.graal.MaximumInliningSize=10000");
            });
  }

  @Test
  void withholdsRecommendationsMeasuredForAnotherPlatform() {
    Fixture fixture = fixture();
    Map<String, String> runtime = new HashMap<>(fixture.profile().runtimeSelector());
    runtime.put("os", "macOS");

    ModelJarPerformanceSelection selection =
        ModelJarPerformanceSelection.evaluate(
            fixture.descriptor(), List.of(fixture.profile()), runtime, List.of());

    assertThat(selection.recommendations()).isEmpty();
    assertThat(selection.enrich(base()).optimization(DECISION_ID))
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
            .filter(value -> value.id().equals("qwen3_0_6b_q4_0_epyc_milan_jdk25"))
            .findFirst()
            .orElseThrow();
    return new Fixture(descriptor, profile);
  }

  private record Fixture(ModelJarDescriptor descriptor, ModelPerformanceProfile profile) {}
}
