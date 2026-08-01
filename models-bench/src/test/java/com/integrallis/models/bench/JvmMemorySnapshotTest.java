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

import java.lang.management.MemoryUsage;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class JvmMemorySnapshotTest {

  @Test
  void recordsHeapNonHeapBufferPoolsAndNativeMemory() {
    NativeMemoryTracking.Summary nativeMemory =
        new NativeMemoryTracking.Summary(
            true,
            10_000,
            8_000,
            Map.of("Java Heap", new NativeMemoryTracking.Category(4_000, 3_000)));

    JvmMemorySnapshot snapshot =
        JvmMemorySnapshot.from(
            new MemoryUsage(0, 100, 200, 300),
            new MemoryUsage(0, 40, 80, 120),
            List.of(
                new JvmMemorySnapshot.BufferPoolUsage("direct", 2, 20, 24),
                new JvmMemorySnapshot.BufferPoolUsage("mapped", 1, 700, 700)),
            nativeMemory);

    assertThat(snapshot.heapUsedBytes()).isEqualTo(100);
    assertThat(snapshot.heapCommittedBytes()).isEqualTo(200);
    assertThat(snapshot.nonHeapUsedBytes()).isEqualTo(40);
    assertThat(snapshot.nonHeapCommittedBytes()).isEqualTo(80);
    assertThat(snapshot.bufferPools())
        .containsEntry("direct", new JvmMemorySnapshot.BufferPoolUsage("direct", 2, 20, 24))
        .containsEntry("mapped", new JvmMemorySnapshot.BufferPoolUsage("mapped", 1, 700, 700));
    assertThat(snapshot.nativeMemory()).isEqualTo(nativeMemory);
  }
}
