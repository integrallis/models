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
package com.integrallis.models.backend.purejava.diagnostics;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import jdk.jfr.Category;
import jdk.jfr.Name;
import jdk.jfr.consumer.RecordedEvent;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PerformanceCliffsTest {

  @Test
  void eventIsNamedAndCategorizedForModels() {
    assertThat(PerformanceCliffEvent.class.getAnnotation(Name.class).value())
        .isEqualTo("com.integrallis.models.PerformanceCliff");
    assertThat(PerformanceCliffEvent.class.getAnnotation(Category.class).value())
        .containsExactly("Models");
  }

  @Test
  void cliffIdsAreUniqueKebabCase() {
    List<String> ids = Arrays.stream(PerformanceCliff.values()).map(PerformanceCliff::id).toList();

    assertThat(new HashSet<>(ids)).hasSameSizeAs(ids);
    assertThat(ids).allMatch(id -> id.matches("[a-z0-9]+(-[a-z0-9]+)*"));
    assertThat(PerformanceCliff.values()).allMatch(cliff -> !cliff.description().isBlank());
  }

  @Test
  void reportsEachReasonOncePerProcessNotPerCall() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      List<Boolean> first =
          IntStream.range(0, 10_000)
              .mapToObj(
                  call ->
                      PerformanceCliffs.report(
                          PerformanceCliff.ROW_BY_ROW_PROJECTION, "call " + call))
              .toList();
      boolean other = PerformanceCliffs.report(PerformanceCliff.VECTOR_WIDTH_CAPPED, "128 < 512");

      assertThat(first.getFirst()).isTrue();
      assertThat(first.subList(1, first.size())).containsOnly(false);
      assertThat(other).isTrue();
      assertThat(recording.count(PerformanceCliff.ROW_BY_ROW_PROJECTION)).isEqualTo(1);
      assertThat(recording.count(PerformanceCliff.VECTOR_WIDTH_CAPPED)).isEqualTo(1);
      RecordedEvent event = recording.events().getFirst();
      assertThat(event.getString("reason")).isEqualTo("row-by-row-projection");
      assertThat(event.getString("detail")).isEqualTo("call 0");
      assertThat(event.getStackTrace()).isNotNull();
    }
  }

  @Test
  void concurrentReportersCommitExactlyOneEvent() throws Exception {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      List<Thread> threads =
          IntStream.range(0, 16)
              .mapToObj(
                  thread ->
                      Thread.ofPlatform()
                          .start(
                              () -> {
                                for (int call = 0; call < 1_000; call++) {
                                  PerformanceCliffs.report(
                                      PerformanceCliff.NATIVE_GATED_DELTA_NET_UNAVAILABLE,
                                      "thread " + thread);
                                }
                              }))
              .toList();
      for (Thread thread : threads) {
        thread.join();
      }

      assertThat(recording.count(PerformanceCliff.NATIVE_GATED_DELTA_NET_UNAVAILABLE)).isEqualTo(1);
    }
  }

  @Test
  void reportedCliffsAppearInTheEnvironmentWithTheirFirstDetail() {
    PerformanceCliffs.reset();
    PerformanceCliffs.report(PerformanceCliff.PERSISTENT_EXECUTOR_NOT_USED, "executor=fork-join");
    PerformanceCliffs.report(PerformanceCliff.VECTOR_API_UNAVAILABLE, "provider=scalar");
    PerformanceCliffs.report(PerformanceCliff.VECTOR_API_UNAVAILABLE, "ignored second detail");

    assertThat(PerformanceCliffs.reported())
        .containsExactly(
            Map.entry(PerformanceCliff.VECTOR_API_UNAVAILABLE, "provider=scalar"),
            Map.entry(PerformanceCliff.PERSISTENT_EXECUTOR_NOT_USED, "executor=fork-join"));
    assertThat(PerformanceCliffs.environment())
        .containsExactlyInAnyOrderEntriesOf(
            Map.of(
                "performance-cliffs",
                "vector-api-unavailable,persistent-executor-not-used",
                "performance-cliff.vector-api-unavailable",
                "provider=scalar",
                "performance-cliff.persistent-executor-not-used",
                "executor=fork-join"));
  }

  @Test
  void enrichAddsCliffKeysAndLeavesEverythingElseUnchanged() {
    BackendDiagnostics base =
        new BackendDiagnostics(
            "pure-java",
            "pure-java-v20",
            Map.of("vector-api", "true"),
            List.of(new OptimizationDecision("x", OptimizationStatus.ENABLED, "y", Map.of())));
    PerformanceCliffs.reset();

    assertThat(PerformanceCliffs.enrich(base)).isEqualTo(base);

    PerformanceCliffs.report(PerformanceCliff.Q4_PAIRWISE_KERNEL_UNSUPPORTED, "requested=short");
    BackendDiagnostics enriched = PerformanceCliffs.enrich(base);

    assertThat(enriched.backend()).isEqualTo(base.backend());
    assertThat(enriched.planVersion()).isEqualTo(base.planVersion());
    assertThat(enriched.optimizations()).isEqualTo(base.optimizations());
    assertThat(enriched.environment())
        .containsEntry("vector-api", "true")
        .containsEntry("performance-cliffs", "q4-pairwise-kernel-unsupported")
        .containsEntry("performance-cliff.q4-pairwise-kernel-unsupported", "requested=short")
        .hasSize(3);
    PerformanceCliffs.reset();
    assertThat(PerformanceCliffs.reported()).isEmpty();
  }
}
