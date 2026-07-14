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

/** Load duration accompanied by enough semantics to prevent invalid comparisons. */
public record LoadMeasurement(
    double millis,
    LoadScope scope,
    boolean coldStartVerified,
    boolean comparableAcrossBackends) {

  public LoadMeasurement {
    if (!Double.isFinite(millis) || millis < 0) {
      throw new IllegalArgumentException("millis must be finite and non-negative");
    }
  }

  static LoadMeasurement inProcessInitialization(double millis) {
    return new LoadMeasurement(millis, LoadScope.IN_PROCESS_INITIALIZATION, false, false);
  }

  static LoadMeasurement backendReported(double millis) {
    return new LoadMeasurement(millis, LoadScope.BACKEND_REPORTED_REQUEST, false, false);
  }

  static LoadMeasurement notMeasured() {
    return new LoadMeasurement(0, LoadScope.NOT_MEASURED, false, false);
  }
}
