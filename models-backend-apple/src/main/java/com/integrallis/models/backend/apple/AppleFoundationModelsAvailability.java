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

import java.util.Objects;

/**
 * Availability report for Apple Foundation Models.
 *
 * @param supported whether this JVM is running on a platform that can theoretically use Apple
 *     Foundation Models
 * @param available whether the native bridge and Apple Foundation Models runtime are currently
 *     available
 * @param reason human-readable availability detail
 */
public record AppleFoundationModelsAvailability(
    boolean supported, boolean available, String reason) {

  public AppleFoundationModelsAvailability {
    Objects.requireNonNull(reason, "reason");
    if (available && !supported) {
      throw new IllegalArgumentException("available requires supported");
    }
  }

  /** Returns an available Apple Foundation Models report. */
  public static AppleFoundationModelsAvailability availableNow() {
    return new AppleFoundationModelsAvailability(true, true, "Apple Foundation Models available");
  }

  /** Returns an unavailable report for a supported platform. */
  public static AppleFoundationModelsAvailability unavailable(String reason) {
    return new AppleFoundationModelsAvailability(true, false, reason);
  }

  /** Returns an unsupported report. */
  public static AppleFoundationModelsAvailability unsupported(String reason) {
    return new AppleFoundationModelsAvailability(false, false, reason);
  }
}
