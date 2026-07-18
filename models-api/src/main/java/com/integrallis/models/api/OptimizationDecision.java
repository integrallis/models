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
package com.integrallis.models.api;

import java.util.Map;
import java.util.Objects;

/** One inspectable optimization choice made while constructing an inference backend. */
public record OptimizationDecision(
    String id, OptimizationStatus status, String reason, Map<String, String> settings) {

  public OptimizationDecision {
    if (id == null || !id.matches("[a-z0-9][a-z0-9.-]*")) {
      throw new IllegalArgumentException("id must be a lowercase optimization identifier: " + id);
    }
    status = Objects.requireNonNull(status, "status");
    if (reason == null || reason.isBlank()) {
      throw new IllegalArgumentException("reason must not be blank");
    }
    reason = reason.trim();
    settings = Map.copyOf(Objects.requireNonNull(settings, "settings"));
  }
}
