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

import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * How many tokens a session is sized for when nobody says: the model's declared maximum.
 *
 * <p>This file previously asserted a bound of 8192 chosen here. That was the wrong fix for #357 and
 * is deliberately recorded as such: the OOM came from allocating the ceiling at session open, not
 * from the ceiling being the model's own, and capping the default traded a crash for the quieter
 * failure of refusing prompts the model accepts. The ceiling is back to what the model declares,
 * and the allocation is what grows -- asserted in {@code Qwen35SessionGrowthTest}.
 */
class SessionContextSizingTest {

  /** Qwen3.5's declared context, the one that produced the failure. */
  private static final int LONG_CONTEXT = 262_144;

  /**
   * A per-model recommendation, with no deployment override in play.
   *
   * <p>{@code fromSystemProperties} folds in any real deployment property, so the assertion would
   * read whatever the surrounding JVM happened to set. Checking that it is absent keeps a green run
   * from meaning "the ambient configuration agreed with us".
   */
  private static PureJavaPlanConfiguration recommending(int maxContextLength) {
    assertThat(System.getProperty(PureJavaPlanConfiguration.MAX_CONTEXT_LENGTH_PROPERTY))
        .as("a deployment override would mask the recommendation under test")
        .isNull();
    return PureJavaPlanConfiguration.fromSystemProperties(
        Map.of(
            PureJavaPlanConfiguration.MAX_CONTEXT_LENGTH_PROPERTY,
            Integer.toString(maxContextLength)));
  }

  @Test
  void anUnconfiguredModelIsSizedForEverythingItDeclares() {
    // Including the long-context architecture that OOMed: the ceiling is not the defect.
    assertThat(
            PureJavaBackend.runtimeContextLength(
                LONG_CONTEXT, PureJavaPlanConfiguration.defaults()))
        .isEqualTo(LONG_CONTEXT);

    for (int declared : new int[] {512, 2048, 4096, 8192, 32_768, 131_072, 1_024_000}) {
      assertThat(
              PureJavaBackend.runtimeContextLength(declared, PureJavaPlanConfiguration.defaults()))
          .as("declared context %s", declared)
          .isEqualTo(declared);
    }
  }

  @Test
  void aRecommendationOrDeploymentSettingCanAskForLess() {
    // Still useful: a deployment that knows its prompts are short can cap the growth ceiling and
    // get a hard refusal rather than letting a runaway sequence grow into the heap.
    assertThat(PureJavaBackend.runtimeContextLength(LONG_CONTEXT, recommending(2048)))
        .isEqualTo(2048);
    assertThat(PureJavaBackend.runtimeContextLength(LONG_CONTEXT, recommending(131_072)))
        .isEqualTo(131_072);
  }

  @Test
  void anExplicitRequestNeverExceedsWhatTheModelCanAttendTo() {
    // Asking for more than the model declares is not a licence to allocate it: the cache would be
    // addressable beyond any position the model can represent.
    assertThat(PureJavaBackend.runtimeContextLength(4096, recommending(131_072))).isEqualTo(4096);
  }
}
