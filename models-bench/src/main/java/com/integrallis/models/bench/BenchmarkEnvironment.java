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
import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** Hardware, operating system, and JVM identity captured with every report. */
public record BenchmarkEnvironment(
    String host,
    String osName,
    String osVersion,
    String architecture,
    String cpuModel,
    int processors,
    long physicalMemoryBytes,
    String javaVersion,
    String javaVendor,
    String vmName) {

  static BenchmarkEnvironment capture() {
    return new BenchmarkEnvironment(
        hostName(),
        System.getProperty("os.name"),
        System.getProperty("os.version"),
        System.getProperty("os.arch"),
        detectCpuModel(),
        Runtime.getRuntime().availableProcessors(),
        physicalMemory(),
        System.getProperty("java.version"),
        System.getProperty("java.vendor"),
        System.getProperty("java.vm.name"));
  }

  private static String hostName() {
    return chooseHostName(System.getenv("HOSTNAME"), readHostNameFile(), jvmHostName());
  }

  static String chooseHostName(String environment, String file, String jvm) {
    for (String candidate : new String[] {environment, file, jvm}) {
      if (candidate != null && !candidate.isBlank()) {
        return candidate.trim();
      }
    }
    return "unknown";
  }

  private static String readHostNameFile() {
    try {
      Path hostname = Path.of("/etc/hostname");
      return Files.isRegularFile(hostname) ? Files.readString(hostname).trim() : "";
    } catch (IOException ignored) {
      return "";
    }
  }

  private static String jvmHostName() {
    try {
      return InetAddress.getLocalHost().getHostName();
    } catch (UnknownHostException ignored) {
      return "";
    }
  }

  private static long physicalMemory() {
    java.lang.management.OperatingSystemMXBean bean = ManagementFactory.getOperatingSystemMXBean();
    if (bean instanceof com.sun.management.OperatingSystemMXBean sunBean) {
      return sunBean.getTotalMemorySize();
    }
    return 0;
  }

  private static String detectCpuModel() {
    Path cpuInfo = Path.of("/proc/cpuinfo");
    if (Files.isRegularFile(cpuInfo)) {
      try {
        return Files.readAllLines(cpuInfo, StandardCharsets.UTF_8).stream()
            .filter(line -> line.toLowerCase(Locale.ROOT).startsWith("model name"))
            .map(line -> line.substring(line.indexOf(':') + 1).trim())
            .findFirst()
            .orElse("unknown");
      } catch (IOException ignored) {
        return "unknown";
      }
    }
    if (System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac")) {
      try {
        Process process = new ProcessBuilder("sysctl", "-n", "machdep.cpu.brand_string").start();
        String value =
            new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
        return process.waitFor() == 0 && !value.isBlank() ? value : "unknown";
      } catch (IOException ignored) {
        return "unknown";
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
      }
    }
    return "unknown";
  }
}
