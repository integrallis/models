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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ProcessMemoryTest {

  @Test
  void parsesLinuxHighWaterMarkInKibibytes() {
    String status = "Name:\tjava\nVmPeak:\t9000000 kB\nVmHWM:\t123456 kB\nVmRSS:\t120000 kB\n";

    assertThat(ProcessMemory.parseLinuxStatus(status)).isEqualTo(123456L * 1024);
  }
}
