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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** Best available process resident-memory high-water reading for the current platform. */
final class ProcessMemory {

  private ProcessMemory() {}

  static long highWaterBytes() {
    return highWaterBytes(ProcessHandle.current().pid());
  }

  static long highWaterBytes(long pid) {
    return snapshot(pid).highWaterBytes();
  }

  static Snapshot snapshot(long pid) {
    if (pid <= 0) {
      return Snapshot.ZERO;
    }
    Path status = Path.of("/proc", Long.toString(pid), "status");
    if (Files.isRegularFile(status)) {
      try {
        return parseLinuxStatusSnapshot(Files.readString(status, StandardCharsets.UTF_8));
      } catch (IOException ignored) {
        return Snapshot.ZERO;
      }
    }
    if (System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac")) {
      long residentBytes = currentMacResidentBytes(pid);
      return new Snapshot(residentBytes, residentBytes, 0, 0, 0);
    }
    return Snapshot.ZERO;
  }

  static long parseLinuxStatus(String status) {
    return parseLinuxStatusSnapshot(status).highWaterBytes();
  }

  static Snapshot parseLinuxStatusSnapshot(String status) {
    long highWaterBytes = 0;
    long residentBytes = 0;
    long anonymousResidentBytes = 0;
    long fileResidentBytes = 0;
    long sharedMemoryResidentBytes = 0;
    for (String line : status.lines().toList()) {
      if (line.startsWith("VmHWM:")) {
        highWaterBytes = kibibytes(line, "VmHWM:");
      } else if (line.startsWith("VmRSS:")) {
        residentBytes = kibibytes(line, "VmRSS:");
      } else if (line.startsWith("RssAnon:")) {
        anonymousResidentBytes = kibibytes(line, "RssAnon:");
      } else if (line.startsWith("RssFile:")) {
        fileResidentBytes = kibibytes(line, "RssFile:");
      } else if (line.startsWith("RssShmem:")) {
        sharedMemoryResidentBytes = kibibytes(line, "RssShmem:");
      }
    }
    return new Snapshot(
        highWaterBytes,
        residentBytes,
        anonymousResidentBytes,
        fileResidentBytes,
        sharedMemoryResidentBytes);
  }

  private static long kibibytes(String line, String prefix) {
    String digits = line.substring(prefix.length()).replaceAll("[^0-9]", "");
    return digits.isEmpty() ? 0 : Long.parseLong(digits) * 1024;
  }

  private static long currentMacResidentBytes(long pid) {
    try {
      Process process = new ProcessBuilder("ps", "-o", "rss=", "-p", Long.toString(pid)).start();
      String value =
          new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
      return process.waitFor() == 0 && !value.isBlank() ? Long.parseLong(value) * 1024 : 0;
    } catch (IOException | NumberFormatException ignored) {
      return 0;
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return 0;
    }
  }

  record Snapshot(
      long highWaterBytes,
      long residentBytes,
      long anonymousResidentBytes,
      long fileResidentBytes,
      long sharedMemoryResidentBytes) {
    private static final Snapshot ZERO = new Snapshot(0, 0, 0, 0, 0);
  }
}
