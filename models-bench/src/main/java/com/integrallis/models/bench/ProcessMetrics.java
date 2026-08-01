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
import java.util.stream.Stream;

/** Best-effort CPU and resident-memory metrics for a backend process tree. */
final class ProcessMetrics {

  private ProcessMetrics() {}

  static Snapshot capture(long rootPid) {
    return capture(rootPid, ProcessHandle::descendants, ProcessMemory::snapshot);
  }

  static Snapshot capture(long rootPid, DescendantSource descendantSource) {
    return capture(rootPid, descendantSource, ProcessMemory::snapshot);
  }

  static Snapshot capture(
      long rootPid, DescendantSource descendantSource, ProcessMemorySource memorySource) {
    if (rootPid <= 0) {
      return Snapshot.ZERO;
    }
    ProcessHandle root = ProcessHandle.of(rootPid).orElse(null);
    if (root == null) {
      return Snapshot.ZERO;
    }
    List<ProcessHandle> processes = new ArrayList<>();
    processes.add(root);
    try (Stream<ProcessHandle> descendants = descendantSource.descendants(root)) {
      descendants.forEach(processes::add);
    } catch (RuntimeException ignored) {
      // Some operating systems deny process-tree enumeration; retain root-process metrics.
    }
    long highWaterBytes = 0;
    long residentBytes = 0;
    long anonymousResidentBytes = 0;
    long fileResidentBytes = 0;
    long sharedMemoryResidentBytes = 0;
    Duration cpu = Duration.ZERO;
    for (ProcessHandle process : processes) {
      ProcessMemory.Snapshot memory = memorySource.capture(process.pid());
      highWaterBytes += memory.highWaterBytes();
      residentBytes += memory.residentBytes();
      anonymousResidentBytes += memory.anonymousResidentBytes();
      fileResidentBytes += memory.fileResidentBytes();
      sharedMemoryResidentBytes += memory.sharedMemoryResidentBytes();
      cpu = cpu.plus(process.info().totalCpuDuration().orElse(Duration.ZERO));
    }
    return new Snapshot(
        highWaterBytes,
        residentBytes,
        anonymousResidentBytes,
        fileResidentBytes,
        sharedMemoryResidentBytes,
        cpu);
  }

  @FunctionalInterface
  interface DescendantSource {
    Stream<ProcessHandle> descendants(ProcessHandle root);
  }

  @FunctionalInterface
  interface ProcessMemorySource {
    ProcessMemory.Snapshot capture(long pid);
  }

  record Snapshot(
      long highWaterBytes,
      long residentBytes,
      long anonymousResidentBytes,
      long fileResidentBytes,
      long sharedMemoryResidentBytes,
      Duration cpu) {
    private static final Snapshot ZERO = new Snapshot(0, 0, 0, 0, 0, Duration.ZERO);

    double cpuMillisSince(Snapshot earlier) {
      long nanos = cpu.minus(earlier.cpu).toNanos();
      return Math.max(0, nanos) / 1_000_000.0;
    }
  }
}
