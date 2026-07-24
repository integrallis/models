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
import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Path;

/** Host and JVM identity captured with each benchmark. */
public record RagBenchmarkEnvironment(
    String hostname,
    String osName,
    String osVersion,
    String architecture,
    String cpuModel,
    int availableProcessors,
    long totalMemoryBytes,
    long maxHeapBytes,
    String javaVersion,
    String javaVendor,
    String vmName) {

  public static RagBenchmarkEnvironment capture() {
    java.lang.management.OperatingSystemMXBean standard =
        java.lang.management.ManagementFactory.getOperatingSystemMXBean();
    long totalMemory =
        standard instanceof com.sun.management.OperatingSystemMXBean extended
            ? extended.getTotalMemorySize()
            : 0;
    return new RagBenchmarkEnvironment(
        resolveHostname(),
        System.getProperty("os.name", "unknown"),
        System.getProperty("os.version", "unknown"),
        System.getProperty("os.arch", "unknown"),
        detectCpuModel(),
        Runtime.getRuntime().availableProcessors(),
        totalMemory,
        Runtime.getRuntime().maxMemory(),
        System.getProperty("java.version", "unknown"),
        System.getProperty("java.vendor", "unknown"),
        System.getProperty("java.vm.name", "unknown"));
  }

  private static String resolveHostname() {
    try {
      return InetAddress.getLocalHost().getHostName();
    } catch (IOException failure) {
      return "unknown";
    }
  }

  private static String detectCpuModel() {
    Path cpuInfo = Path.of("/proc/cpuinfo");
    if (Files.isRegularFile(cpuInfo)) {
      try {
        return Files.readAllLines(cpuInfo).stream()
            .filter(line -> line.startsWith("model name"))
            .map(line -> line.substring(line.indexOf(':') + 1).trim())
            .findFirst()
            .orElse("unknown");
      } catch (IOException ignored) {
        return "unknown";
      }
    }
    return System.getenv().getOrDefault("PROCESSOR_IDENTIFIER", "unknown");
  }
}
