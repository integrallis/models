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

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/** Best-effort CPU and resident-memory metrics for a backend process tree. */
final class ProcessMetrics {

  private ProcessMetrics() {}

  static Snapshot capture(long rootPid) {
    if (rootPid <= 0) {
      return Snapshot.ZERO;
    }
    ProcessHandle root = ProcessHandle.of(rootPid).orElse(null);
    if (root == null) {
      return Snapshot.ZERO;
    }
    List<ProcessHandle> processes = new ArrayList<>();
    processes.add(root);
    root.descendants().forEach(processes::add);
    long highWaterBytes = 0;
    Duration cpu = Duration.ZERO;
    for (ProcessHandle process : processes) {
      highWaterBytes += ProcessMemory.highWaterBytes(process.pid());
      cpu = cpu.plus(process.info().totalCpuDuration().orElse(Duration.ZERO));
    }
    return new Snapshot(highWaterBytes, cpu);
  }

  record Snapshot(long highWaterBytes, Duration cpu) {
    private static final Snapshot ZERO = new Snapshot(0, Duration.ZERO);

    double cpuMillisSince(Snapshot earlier) {
      long nanos = cpu.minus(earlier.cpu).toNanos();
      return Math.max(0, nanos) / 1_000_000.0;
    }
  }
}
