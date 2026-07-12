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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ApplePlatformTest {

  @Test
  void intelMacIsUnsupportedForAppleFoundationModels() {
    ApplePlatform platform = new ApplePlatform("Mac OS X", "x86_64");

    AppleFoundationModelsAvailability availability = platform.foundationModelsSupport();

    assertThat(availability.supported()).isFalse();
    assertThat(availability.available()).isFalse();
    assertThat(availability.reason()).contains("Apple silicon");
  }

  @Test
  void appleSiliconMacIsPotentiallySupported() {
    ApplePlatform platform = new ApplePlatform("Mac OS X", "aarch64");

    AppleFoundationModelsAvailability availability = platform.foundationModelsSupport();

    assertThat(availability.supported()).isTrue();
    assertThat(availability.available()).isFalse();
    assertThat(availability.reason()).contains("native bridge has not been loaded");
  }

  @Test
  void nonMacPlatformsAreUnsupported() {
    ApplePlatform platform = new ApplePlatform("Linux", "aarch64");

    AppleFoundationModelsAvailability availability = platform.foundationModelsSupport();

    assertThat(availability.supported()).isFalse();
    assertThat(availability.reason()).contains("macOS");
  }
}
