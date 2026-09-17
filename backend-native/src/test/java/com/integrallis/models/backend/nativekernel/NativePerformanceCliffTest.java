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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliff;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliffEvent;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliffs;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Test;

/** Native-kernel fallbacks report their named performance cliff once per process. */
class NativePerformanceCliffTest {

  @Test
  void libraryWithoutPollBudgetCapabilityReportsTheCliffOnce() throws Exception {
    List<RecordedEvent> events =
        record(
            () -> {
              for (int context = 0; context < 3; context++) {
                NativeKernelLibrary.applyPollBudget(null, MemorySegment.NULL, 4);
              }
            });

    assertThat(reasons(events)).containsExactly("native-poll-budget-unsupported");
    assertThat(PerformanceCliffs.reported().get(PerformanceCliff.NATIVE_POLL_BUDGET_UNSUPPORTED))
        .contains("poll-millis=4");
  }

  @Test
  void bundledLibraryAppliesThePollBudgetAndReportsNothing() throws Exception {
    Path library = Path.of(System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY));
    List<RecordedEvent> events =
        record(
            () -> {
              try (NativeKernelLibrary opened = NativeKernelLibrary.open(library, 2)) {
                assertThat(opened.supportsPollBudget()).isTrue();
              }
            });

    assertThat(reasons(events)).doesNotContain("native-poll-budget-unsupported");
  }

  @Test
  void rustFfmDiagnosticsIncludeCliffsTakenAfterLoad() {
    BackendDiagnostics loaded =
        new BackendDiagnostics(
            "rust-ffm", RustFfmBackend.PLAN_VERSION, Map.of("a", "b"), List.of());
    RustFfmBackend backend = new RustFfmBackend(mock(PureJavaBackend.class), loaded);
    PerformanceCliffs.reset();

    assertThat(backend.diagnostics()).isEqualTo(loaded);

    PerformanceCliffs.report(PerformanceCliff.NATIVE_GATED_DELTA_NET_UNAVAILABLE, "kernel=rust");

    assertThat(backend.diagnostics().environment())
        .containsEntry("a", "b")
        .containsEntry("performance-cliff.native-gated-delta-net-unavailable", "kernel=rust")
        .containsEntry("performance-cliffs", "native-gated-delta-net-unavailable");
    PerformanceCliffs.reset();
  }

  private static List<RecordedEvent> record(Runnable action) throws Exception {
    PerformanceCliffs.reset();
    Path dump = Files.createTempFile("native-cliffs", ".jfr");
    try (Recording recording = new Recording()) {
      recording.enable(PerformanceCliffEvent.NAME);
      recording.start();
      action.run();
      recording.stop();
      recording.dump(dump);
      return RecordingFile.readAllEvents(dump).stream()
          .filter(event -> event.getEventType().getName().equals(PerformanceCliffEvent.NAME))
          .toList();
    } finally {
      Files.deleteIfExists(dump);
    }
  }

  private static List<String> reasons(List<RecordedEvent> events) {
    return events.stream().map(event -> event.getString("reason")).toList();
  }
}
