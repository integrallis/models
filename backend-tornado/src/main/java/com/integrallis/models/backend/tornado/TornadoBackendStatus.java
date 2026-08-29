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

import java.time.Duration;
import java.util.Objects;

/** Immutable outcome of automatic accelerator selection and readiness. */
public record TornadoBackendStatus(
    boolean accelerated,
    String device,
    String reason,
    long requiredDeviceBytes,
    Duration readinessTime) {

  public TornadoBackendStatus {
    device = requireText(device, "device");
    reason = requireText(reason, "reason");
    if (requiredDeviceBytes < 0) {
      throw new IllegalArgumentException("requiredDeviceBytes must not be negative");
    }
    readinessTime = Objects.requireNonNull(readinessTime, "readinessTime");
    if (readinessTime.isNegative()) {
      throw new IllegalArgumentException("readinessTime must not be negative");
    }
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value.strip();
  }
}
