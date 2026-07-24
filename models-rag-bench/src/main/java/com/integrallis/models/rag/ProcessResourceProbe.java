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
package com.integrallis.models.rag;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Optional;
import java.util.stream.Stream;

final class ProcessResourceProbe {
  private ProcessResourceProbe() {}

  static Duration cpuDuration(long pid) {
    if (pid <= 0) {
      return Duration.ZERO;
    }
    return ProcessHandle.of(pid)
        .flatMap(handle -> handle.info().totalCpuDuration())
        .orElse(Duration.ZERO);
  }

  static long highWaterBytes(long pid) {
    if (pid <= 0) {
      return 0;
    }
    Path status = Path.of("/proc", Long.toString(pid), "status");
    if (!Files.isRegularFile(status)) {
      return 0;
    }
    try (Stream<String> lines = Files.lines(status)) {
      Optional<String> highWater = lines.filter(line -> line.startsWith("VmHWM:")).findFirst();
      if (highWater.isEmpty()) {
        return 0;
      }
      String[] parts = highWater.orElseThrow().trim().split("\\s+");
      return parts.length >= 2 ? Long.parseLong(parts[1]) * 1_024 : 0;
    } catch (IOException | NumberFormatException ignored) {
      return 0;
    }
  }
}
