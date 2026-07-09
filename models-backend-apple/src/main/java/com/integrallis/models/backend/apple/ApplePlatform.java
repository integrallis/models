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
package com.integrallis.models.backend.apple;

import java.util.Locale;
import java.util.Objects;

record ApplePlatform(String osName, String osArch) {

  ApplePlatform {
    Objects.requireNonNull(osName, "osName");
    Objects.requireNonNull(osArch, "osArch");
  }

  static ApplePlatform current() {
    return new ApplePlatform(System.getProperty("os.name"), System.getProperty("os.arch"));
  }

  AppleFoundationModelsAvailability foundationModelsSupport() {
    if (!isMac()) {
      return AppleFoundationModelsAvailability.unsupported(
          "Apple Foundation Models require macOS; current OS is " + osName);
    }
    if (!isAppleSilicon()) {
      return AppleFoundationModelsAvailability.unsupported(
          "Apple Foundation Models require Apple silicon; current architecture is " + osArch);
    }
    return AppleFoundationModelsAvailability.unavailable(
        "Apple Foundation Models native bridge has not been loaded");
  }

  private boolean isMac() {
    return osName.toLowerCase(Locale.ROOT).contains("mac");
  }

  private boolean isAppleSilicon() {
    String arch = osArch.toLowerCase(Locale.ROOT);
    return arch.equals("aarch64") || arch.equals("arm64");
  }
}
