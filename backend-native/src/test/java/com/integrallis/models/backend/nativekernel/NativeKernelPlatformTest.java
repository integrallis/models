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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class NativeKernelPlatformTest {
  @Test
  void normalizesEveryCiTarget() {
    assertPlatform("Linux", "amd64", "linux-x86_64", "libjmodels_kernels.so");
    assertPlatform("Linux", "aarch64", "linux-aarch64", "libjmodels_kernels.so");
    assertPlatform("Mac OS X", "x86_64", "macos-x86_64", "libjmodels_kernels.dylib");
    assertPlatform("Darwin", "arm64", "macos-aarch64", "libjmodels_kernels.dylib");
    assertPlatform("Windows 11", "x64", "windows-x86_64", "jmodels_kernels.dll");
    assertPlatform("Windows Server 2025", "aarch64", "windows-aarch64", "jmodels_kernels.dll");
  }

  @Test
  void rejectsUnsupportedOperatingSystemsAndArchitectures() {
    assertThatThrownBy(() -> NativeKernelPlatform.detect("FreeBSD", "amd64"))
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("FreeBSD")
        .hasMessageContaining("amd64");
    assertThatThrownBy(() -> NativeKernelPlatform.detect("Linux", "riscv64"))
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("Linux")
        .hasMessageContaining("riscv64");
  }

  private static void assertPlatform(
      String osName, String osArch, String expectedId, String expectedLibraryName) {
    NativeKernelPlatform platform = NativeKernelPlatform.detect(osName, osArch);

    assertThat(platform.id()).isEqualTo(expectedId);
    assertThat(platform.libraryFileName()).isEqualTo(expectedLibraryName);
    assertThat(platform.resourceDirectory())
        .isEqualTo("META-INF/models/native/" + expectedId + "/");
  }
}
