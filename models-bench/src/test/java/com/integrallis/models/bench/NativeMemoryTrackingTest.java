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

import org.junit.jupiter.api.Test;

class NativeMemoryTrackingTest {

  @Test
  void parsesJdk25SummaryUsingTheRequestedKilobyteScale() {
    String output =
        """
        Native Memory Tracking:

        Total: reserved=10057734KB, committed=229442KB
               malloc: 73274KB #97364, peak=91267KB #96538
               mmap:   reserved=9984460KB, committed=156168KB

        -                 Java Heap (reserved=8388608KB, committed=81920KB)
                                    (mmap: reserved=8388608KB, committed=81920KB)

        -                     Class (reserved=1049017KB, committed=3833KB)
                                    (classes #4995)

        -                    Thread (reserved=36968KB, committed=1380KB)
                                    (threads #36)

        -    Native Memory Tracking (reserved=1744KB, committed=1744KB)

        -                 Metaspace (reserved=65619KB, committed=24979KB)
        """;

    NativeMemoryTracking.Summary summary = NativeMemoryTracking.parse(output);

    assertThat(summary.available()).isTrue();
    assertThat(summary.totalReservedBytes()).isEqualTo(10_057_734L * 1_024);
    assertThat(summary.totalCommittedBytes()).isEqualTo(229_442L * 1_024);
    assertThat(summary.categories())
        .containsEntry(
            "Java Heap", new NativeMemoryTracking.Category(8_388_608L * 1_024, 81_920L * 1_024))
        .containsEntry(
            "Native Memory Tracking",
            new NativeMemoryTracking.Category(1_744L * 1_024, 1_744L * 1_024))
        .containsEntry(
            "Metaspace", new NativeMemoryTracking.Category(65_619L * 1_024, 24_979L * 1_024));
  }

  @Test
  void reportsDisabledNmtWithoutInventingMeasurements() {
    NativeMemoryTracking.Summary summary =
        NativeMemoryTracking.parse("Native memory tracking is not enabled");

    assertThat(summary.available()).isFalse();
    assertThat(summary.totalReservedBytes()).isZero();
    assertThat(summary.totalCommittedBytes()).isZero();
    assertThat(summary.categories()).isEmpty();
  }
}
