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
package com.integrallis.models.bench.fusion;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.core.VectorUtil;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;

/**
 * Everything about the process and host that can change an inference result or its cost.
 *
 * @param modelsCommit {@code git rev-parse HEAD} of the working directory, or null outside git
 * @param modelsRevisionArgument the {@code --models-revision} value given on the command line
 * @param modelsDirty whether {@code git status --porcelain} reported changes; null outside git
 * @param javaVersionOutput verbatim stderr of {@code java -version} for the running JVM
 * @param jvmInputArguments JVM flags of this process
 * @param inferenceSystemProperties system properties under inference-relevant prefixes
 * @param vectorRuntime Vector API species and kernel selection reported by vectors-core
 * @param availableProcessors processors visible to the JVM
 * @param host host identity
 */
public record FusionEnvironment(
    String modelsCommit,
    String modelsRevisionArgument,
    Boolean modelsDirty,
    String javaVersionOutput,
    List<String> jvmInputArguments,
    Map<String, String> inferenceSystemProperties,
    JsonNode vectorRuntime,
    int availableProcessors,
    Host host) {

  public FusionEnvironment {
    jvmInputArguments = List.copyOf(jvmInputArguments);
    inferenceSystemProperties =
        java.util.Collections.unmodifiableMap(new TreeMap<>(inferenceSystemProperties));
  }

  static final List<String> PROPERTY_PREFIXES =
      List.of(
          "models.",
          "vectors.",
          "jdk.incubator.vector",
          "java.util.concurrent.ForkJoinPool",
          "java.vm.",
          "java.version",
          "java.vendor");

  /**
   * Host identity.
   *
   * @param hostname host name
   * @param cpuModel CPU model string
   * @param cpuFlags CPU feature flags (Linux {@code /proc/cpuinfo}, macOS {@code sysctl})
   * @param logicalCores logical processors reported by the OS
   * @param ramBytes physical memory
   * @param osName OS name
   * @param osVersion OS version
   * @param arch architecture
   */
  public record Host(
      String hostname,
      String cpuModel,
      String cpuFlags,
      int logicalCores,
      long ramBytes,
      String osName,
      String osVersion,
      String arch) {}

  static FusionEnvironment capture(String modelsRevisionArgument, ObjectMapper mapper) {
    String commit = command(List.of("git", "rev-parse", "HEAD"));
    String status = commit == null ? null : command(List.of("git", "status", "--porcelain"));
    Map<String, String> properties = new TreeMap<>();
    System.getProperties()
        .forEach(
            (key, value) -> {
              String name = String.valueOf(key);
              if (PROPERTY_PREFIXES.stream().anyMatch(name::startsWith)) {
                properties.put(name, String.valueOf(value));
              }
            });
    JsonNode vectorRuntime;
    try {
      vectorRuntime = mapper.valueToTree(VectorUtil.runtimeCapabilities());
    } catch (RuntimeException | LinkageError unavailable) {
      vectorRuntime = mapper.getNodeFactory().textNode("unavailable: " + unavailable);
    }
    String javaBinary = Path.of(System.getProperty("java.home"), "bin", "java").toString();
    return new FusionEnvironment(
        commit == null ? null : commit.strip(),
        modelsRevisionArgument,
        status == null ? null : !status.isBlank(),
        commandStderr(List.of(javaBinary, "-version")),
        ManagementFactory.getRuntimeMXBean().getInputArguments(),
        properties,
        vectorRuntime,
        Runtime.getRuntime().availableProcessors(),
        new Host(
            hostname(),
            cpuModel(),
            cpuFlags(),
            logicalCores(),
            physicalMemory(),
            System.getProperty("os.name"),
            System.getProperty("os.version"),
            System.getProperty("os.arch")));
  }

  private static String hostname() {
    try {
      String env = System.getenv("HOSTNAME");
      return env != null && !env.isBlank() ? env : InetAddress.getLocalHost().getHostName();
    } catch (IOException unknown) {
      return "unknown";
    }
  }

  private static boolean mac() {
    return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
  }

  private static String cpuInfoField(String field) {
    try {
      Path info = Path.of("/proc/cpuinfo");
      if (!Files.isRegularFile(info)) {
        return null;
      }
      for (String line : Files.readAllLines(info, StandardCharsets.UTF_8)) {
        int colon = line.indexOf(':');
        if (colon > 0 && line.substring(0, colon).strip().equalsIgnoreCase(field)) {
          return line.substring(colon + 1).strip();
        }
      }
    } catch (IOException ignored) {
      return null;
    }
    return null;
  }

  private static String cpuModel() {
    String linux = cpuInfoField("model name");
    if (linux != null) {
      return linux;
    }
    if (mac()) {
      String brand = command(List.of("sysctl", "-n", "machdep.cpu.brand_string"));
      return brand == null ? "unknown" : brand.strip();
    }
    return "unknown";
  }

  private static String cpuFlags() {
    String flags = cpuInfoField("flags");
    if (flags == null) {
      flags = cpuInfoField("Features");
    }
    if (flags != null) {
      return flags;
    }
    if (mac()) {
      String optional = command(List.of("sysctl", "hw.optional"));
      if (optional != null) {
        StringBuilder enabled = new StringBuilder();
        for (String line : optional.split("\n")) {
          String[] parts = line.split(":\\s*");
          if (parts.length == 2 && "1".equals(parts[1].strip())) {
            enabled
                .append(enabled.isEmpty() ? "" : " ")
                .append(parts[0].replace("hw.optional.", ""));
          }
        }
        return enabled.toString();
      }
    }
    return "unknown";
  }

  private static int logicalCores() {
    if (mac()) {
      String value = command(List.of("sysctl", "-n", "hw.logicalcpu"));
      if (value != null) {
        try {
          return Integer.parseInt(value.strip());
        } catch (NumberFormatException ignored) {
          return Runtime.getRuntime().availableProcessors();
        }
      }
    }
    String value = command(List.of("nproc", "--all"));
    if (value != null) {
      try {
        return Integer.parseInt(value.strip());
      } catch (NumberFormatException ignored) {
        return Runtime.getRuntime().availableProcessors();
      }
    }
    return Runtime.getRuntime().availableProcessors();
  }

  private static long physicalMemory() {
    if (ManagementFactory.getOperatingSystemMXBean()
        instanceof com.sun.management.OperatingSystemMXBean bean) {
      return bean.getTotalMemorySize();
    }
    return 0;
  }

  /** Process CPU time of all threads, in nanoseconds, or -1 when unavailable. */
  static long processCpuNanos() {
    if (ManagementFactory.getOperatingSystemMXBean()
        instanceof com.sun.management.OperatingSystemMXBean bean) {
      return bean.getProcessCpuTime();
    }
    return -1;
  }

  private static String command(List<String> command) {
    return run(command, false);
  }

  private static String commandStderr(List<String> command) {
    return run(command, true);
  }

  private static String run(List<String> command, boolean stderr) {
    try {
      Process process = new ProcessBuilder(command).redirectErrorStream(stderr).start();
      byte[] output = process.getInputStream().readAllBytes();
      if (!process.waitFor(30, TimeUnit.SECONDS) || process.exitValue() != 0) {
        return null;
      }
      return new String(output, StandardCharsets.UTF_8);
    } catch (IOException failure) {
      return null;
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return null;
    }
  }
}
