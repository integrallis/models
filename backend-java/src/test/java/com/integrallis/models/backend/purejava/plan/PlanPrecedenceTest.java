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
package com.integrallis.models.backend.purejava.plan;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Which of the three tiers wins, and why there are three.
 *
 * <p>A recommendation is what a ModelJar says about itself, so a deployment-wide system property
 * beats it. An override is what the caller asked for on this one load, so it beats the property:
 * an application that sizes one runtime for its own sequences cannot say so through a JVM-global
 * property without changing every other runtime in the process. That was the complaint in #357 --
 * a ModelJars caller could bound a session's context only globally.
 */
class PlanPrecedenceTest {

  private static final String CONTEXT = PureJavaPlanConfiguration.MAX_CONTEXT_LENGTH_PROPERTY;

  @Test
  void anOverrideBeatsADeploymentSettingWhichBeatsARecommendation() {
    assertThat(
            PureJavaPlanConfiguration.from(
                    Map.of(CONTEXT, "4096"), Map.of(CONTEXT, "8192"), Map.of(CONTEXT, "2048"))
                .maxContextLength())
        .as("the caller's own request wins over both")
        .isEqualTo(4096);

    assertThat(
            PureJavaPlanConfiguration.from(Map.of(), Map.of(CONTEXT, "8192"), Map.of(CONTEXT, "2048"))
                .maxContextLength())
        .as("with no override, the deployment setting wins over the ModelJar's opinion")
        .isEqualTo(8192);

    assertThat(
            PureJavaPlanConfiguration.from(Map.of(), Map.of(), Map.of(CONTEXT, "2048"))
                .maxContextLength())
        .as("with neither, the recommendation applies")
        .isEqualTo(2048);
  }

  @Test
  void theTwoTierFormStillMeansWhatItMeant() {
    // The old signature is kept and delegates with no overrides, so every caller that has not been
    // taught about the third tier resolves exactly as before. Asserted rather than assumed,
    // because a silent change here would alter every plan in the build.
    assertThat(
            PureJavaPlanConfiguration.from(Map.of(CONTEXT, "8192"), Map.of(CONTEXT, "2048"))
                .maxContextLength())
        .isEqualTo(8192);
    assertThat(PureJavaPlanConfiguration.from(Map.of(), Map.of()).maxContextLength())
        .isEqualTo(PureJavaPlanConfiguration.defaults().maxContextLength());
  }

  @Test
  void anOverrideIsValidatedLikeTheOtherTiers() {
    // An unknown key in the override map is a caller mistake and has to say so, not be ignored:
    // a setting that silently does nothing is indistinguishable from one that does not work.
    org.assertj.core.api.Assertions.assertThatThrownBy(
            () ->
                PureJavaPlanConfiguration.from(
                    Map.of("models.purejava.notASetting", "1"), Map.of(), Map.of()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("override");
  }
}
