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
package com.integrallis.models.backend.tornado;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import org.junit.jupiter.api.Test;

class TornadoBackendStatusTest {

  @Test
  void normalizesAValidStatus() {
    TornadoBackendStatus status =
        new TornadoBackendStatus(
            true, "  NVIDIA A40  ", "  eligible  ", 1024, Duration.ofSeconds(2));

    assertThat(status.device()).isEqualTo("NVIDIA A40");
    assertThat(status.reason()).isEqualTo("eligible");
    assertThat(status.requiredDeviceBytes()).isEqualTo(1024);
    assertThat(status.readinessTime()).isEqualTo(Duration.ofSeconds(2));
  }

  @Test
  void rejectsInvalidStatusValues() {
    assertThatThrownBy(() -> new TornadoBackendStatus(false, " ", "unavailable", 0, Duration.ZERO))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("device");
    assertThatThrownBy(() -> new TornadoBackendStatus(false, "CPU", " ", 0, Duration.ZERO))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("reason");
    assertThatThrownBy(
            () -> new TornadoBackendStatus(false, "CPU", "unavailable", -1, Duration.ZERO))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("requiredDeviceBytes");
    assertThatThrownBy(
            () -> new TornadoBackendStatus(false, "CPU", "unavailable", 0, Duration.ofNanos(-1)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("readinessTime");
  }
}
