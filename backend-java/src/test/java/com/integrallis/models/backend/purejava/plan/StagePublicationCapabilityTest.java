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

import org.junit.jupiter.api.Test;

/**
 * Which GGUF executors the planner will stage work on, asked as a capability.
 *
 * <p>Written because the planner used to ask by name. Eight sites compared the executor against the
 * literal {@code "persistent"}, and when the vectors default became {@code "countdown"} in 0.1.33
 * every staged optimisation turned itself off -- correct output, quietly slower, and nothing failed
 * except three assertions that happened to check one of the flags. The version pin is what kept it
 * out of a release.
 *
 * <p>So the capability is asserted directly here: an executor that retains one publication across
 * ordered stages qualifies, one that would fall back to running the plan serially does not, and the
 * name list lives in exactly one place.
 */
class StagePublicationCapabilityTest {

  private static RuntimeFingerprint withExecutor(String executor) {
    return new RuntimeFingerprint(
        "25.0.3",
        "OpenJDK 64-Bit Server VM",
        "Eclipse Adoptium",
        "25.0.3+9",
        "hotspot-c2",
        "Linux",
        "amd64",
        "AMD EPYC-Milan Processor",
        "test-vector",
        true,
        256,
        256,
        false,
        false,
        false,
        true,
        true,
        true,
        executor,
        8,
        2,
        8);
  }

  @Test
  void executorsThatRetainAStagePublicationQualify() {
    // Both implement execute(GgufStagePlan) rather than inheriting the serial fallback.
    assertThat(withExecutor("persistent").sharesStagePublication()).isTrue();
    assertThat(withExecutor("countdown").sharesStagePublication()).isTrue();
  }

  @Test
  void executorsWithoutItDoNot() {
    // These run a plan stage by stage with no shared publication, so staging buys nothing.
    assertThat(withExecutor("common").sharesStagePublication()).isFalse();
    assertThat(withExecutor("dedicated").sharesStagePublication()).isFalse();
  }

  @Test
  void anUnknownExecutorIsTreatedAsIncapableRatherThanAssumedCapable() {
    // The safe direction: a future executor is opted in deliberately, so the worst case is a
    // missed optimisation rather than a staged plan published to something that runs it serially.
    assertThat(withExecutor("something-new").sharesStagePublication()).isFalse();
  }

  @Test
  void theDefaultVectorsExecutorQualifies() {
    // The regression itself. vectors 0.1.33 ships "countdown" as its default, and if that did not
    // qualify, every staged optimisation would silently switch off on a stock runtime.
    assertThat(RuntimeFingerprint.capture().sharesStagePublication())
        .as(
            "the executor this runtime actually resolved: %s",
            RuntimeFingerprint.capture().ggufExecutor())
        .isTrue();
  }
}
