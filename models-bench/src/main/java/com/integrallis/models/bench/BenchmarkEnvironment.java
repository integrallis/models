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

import com.integrallis.vectors.core.PanamaConstants;
import com.integrallis.vectors.core.VectorizationProvider;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

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
    String vmName,
    String vmVersion,
    List<String> jvmArguments,
    String vectorProvider,
    boolean vectorApiEnabled,
    int preferredVectorBits,
    int effectiveVectorBits,
    Map<String, String> runtimeProperties) {

  private static final List<String> RELEVANT_PROPERTIES =
      List.of(
          "models.purejava.maxContextLength",
          "vectors.forceScalar",
          "vectors.maxBits",
          "vectors.useVectorFMA",
          "vectors.useScalarFMA",
          "vectors.forceSve",
          "jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK");

  static BenchmarkEnvironment capture() {
    boolean vectorApiEnabled = VectorizationProvider.isPanamaEnabled();
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
        System.getProperty("java.vm.name"),
        System.getProperty("java.vm.version"),
        List.copyOf(ManagementFactory.getRuntimeMXBean().getInputArguments()),
        VectorizationProvider.getProviderName(),
        vectorApiEnabled,
        PanamaConstants.PREFERRED_BITS,
        vectorApiEnabled ? Math.min(PanamaConstants.PREFERRED_BITS, PanamaConstants.MAX_BITS) : 0,
        captureRuntimeProperties());
  }

  private static Map<String, String> captureRuntimeProperties() {
    Map<String, String> properties = new LinkedHashMap<>();
    for (String property : RELEVANT_PROPERTIES) {
      properties.put(property, System.getProperty(property, "<unset>"));
    }
    return Map.copyOf(properties);
  }

  private static String hostName() {
    String host = System.getenv("HOSTNAME");
    return host == null || host.isBlank() ? "unknown" : host;
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
        String value = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
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
