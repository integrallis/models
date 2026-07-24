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
package com.integrallis.models.backend.nativekernel;

import java.util.Locale;
import java.util.Objects;

/** Operating-system and architecture combinations distributed by the Models native backend. */
public enum NativeKernelPlatform {
  LINUX_X86_64("linux-x86_64", "libjmodels_kernels.so"),
  LINUX_AARCH64("linux-aarch64", "libjmodels_kernels.so"),
  MACOS_X86_64("macos-x86_64", "libjmodels_kernels.dylib"),
  MACOS_AARCH64("macos-aarch64", "libjmodels_kernels.dylib"),
  WINDOWS_X86_64("windows-x86_64", "jmodels_kernels.dll"),
  WINDOWS_AARCH64("windows-aarch64", "jmodels_kernels.dll");

  private static final String RESOURCE_ROOT = "META-INF/models/native/";

  private final String id;
  private final String libraryFileName;

  NativeKernelPlatform(String id, String libraryFileName) {
    this.id = id;
    this.libraryFileName = libraryFileName;
  }

  /** Returns the stable artifact and resource identifier. */
  public String id() {
    return id;
  }

  /** Returns the platform linker filename. */
  public String libraryFileName() {
    return libraryFileName;
  }

  /** Returns the classpath directory containing this platform's metadata and library. */
  public String resourceDirectory() {
    return RESOURCE_ROOT + id + "/";
  }

  /** Detects the platform hosting the current JVM. */
  public static NativeKernelPlatform current() {
    return detect(System.getProperty("os.name"), System.getProperty("os.arch"));
  }

  /** Normalizes JVM operating-system and architecture names to a distributed platform. */
  public static NativeKernelPlatform detect(String osName, String osArch) {
    Objects.requireNonNull(osName, "osName");
    Objects.requireNonNull(osArch, "osArch");
    String operatingSystem = normalizeOperatingSystem(osName);
    String architecture = normalizeArchitecture(osArch);
    if (operatingSystem == null || architecture == null) {
      throw unsupported(osName, osArch);
    }
    for (NativeKernelPlatform platform : values()) {
      if (platform.id.equals(operatingSystem + "-" + architecture)) {
        return platform;
      }
    }
    throw unsupported(osName, osArch);
  }

  private static String normalizeOperatingSystem(String osName) {
    String normalized = osName.toLowerCase(Locale.ROOT);
    if (normalized.contains("linux")) {
      return "linux";
    }
    if (normalized.contains("mac") || normalized.contains("darwin")) {
      return "macos";
    }
    if (normalized.contains("win")) {
      return "windows";
    }
    return null;
  }

  private static String normalizeArchitecture(String osArch) {
    return switch (osArch.toLowerCase(Locale.ROOT)) {
      case "amd64", "x86_64", "x64" -> "x86_64";
      case "aarch64", "arm64" -> "aarch64";
      default -> null;
    };
  }

  private static UnsupportedOperationException unsupported(String osName, String osArch) {
    return new UnsupportedOperationException(
        "Models native kernels do not support os.name=" + osName + ", os.arch=" + osArch);
  }
}
