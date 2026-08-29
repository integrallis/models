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
package com.integrallis.models.accelerator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class AcceleratorEligibilityTest {
  private static final long MIB = 1024L * 1024L;
  private static final long GIB = 1024L * MIB;

  @Test
  void selectsAQualifiedPtxGpuWithEnoughMemoryForPrefillAndDecodePlans() {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(
                device("host CPU", "JAVA", "CPU", 16 * GIB, 4 * GIB),
                device("NVIDIA A16", "PTX", "GPU", 2 * GIB, 512 * MIB)),
            410 * MIB,
            true);

    assertThat(decision.eligible()).isTrue();
    assertThat(decision.device().name()).isEqualTo("NVIDIA A16");
    assertThat(decision.requiredBytes()).isEqualTo(1_076 * MIB);
  }

  @Test
  void fallsBackWhenTheQualifiedDeviceCannotHoldTheRetainedPlansWithHeadroom() {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(device("NVIDIA A16", "PTX", "GPU", 2 * GIB, 512 * MIB)), 900 * MIB, true);

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains("device memory");
  }

  @Test
  void leavesUnqualifiedVendorsAndCpuDevicesOnTheJavaFallback() {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(
                device("AMD Radeon", "OPENCL", "GPU", 8 * GIB, 2 * GIB),
                device("Intel Xeon", "OPENCL", "CPU", 64 * GIB, 16 * GIB)),
            410 * MIB,
            false);

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains("qualified NVIDIA GPU");
  }

  @Test
  void rejectsInvalidModelSizesWithoutInspectingDevices() {
    AcceleratorEligibility.Decision decision = AcceleratorEligibility.select(List.of(), 0, false);

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains("model size");
  }

  private static AcceleratorEligibility.DeviceCapabilities device(
      String name, String backend, String type, long memory, long allocation) {
    return new AcceleratorEligibility.DeviceCapabilities(name, backend, type, memory, allocation);
  }
}
