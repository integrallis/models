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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

class ProcessMetricsTest {

  @Test
  void retainsRootProcessMetricsWhenTheOperatingSystemDeniesDescendantEnumeration() {
    long currentPid = ProcessHandle.current().pid();

    assertThatCode(
            () ->
                ProcessMetrics.capture(
                    currentPid,
                    ignored -> {
                      throw new RuntimeException("operation not permitted");
                    }))
        .doesNotThrowAnyException();
  }

  @Test
  void retainsCurrentAndHighWaterResidentMemorySeparately() {
    long currentPid = ProcessHandle.current().pid();

    ProcessMetrics.Snapshot snapshot =
        ProcessMetrics.capture(
            currentPid,
            ignored -> Stream.empty(),
            ignored -> new ProcessMemory.Snapshot(1_200, 900));

    assertThat(snapshot.highWaterBytes()).isEqualTo(1_200);
    assertThat(snapshot.residentBytes()).isEqualTo(900);
  }
}
