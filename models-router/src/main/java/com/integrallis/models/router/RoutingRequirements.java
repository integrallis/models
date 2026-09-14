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
package com.integrallis.models.router;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Hard requirements for one routing decision.
 *
 * <p>Capabilities are facts declared by the candidate, such as {@code tool-calling}, {@code
 * text-generation}, or {@code text-embedding}. They are not quality tags. The caller, or a
 * separately audited policy layer, chooses the data boundary; this type deliberately does not
 * pretend to infer whether arbitrary text contains PII.
 */
public record RoutingRequirements(
    Set<String> requiredCapabilities, RoutingDataBoundary dataBoundary) {
  private static final RoutingRequirements NONE =
      new RoutingRequirements(Set.of(), RoutingDataBoundary.REMOTE_ALLOWED);

  /** Validates and defensively copies the requirements. */
  public RoutingRequirements {
    requiredCapabilities =
        Set.copyOf(Objects.requireNonNull(requiredCapabilities, "requiredCapabilities"));
    for (String capability : requiredCapabilities) {
      if (capability == null || capability.isBlank()) {
        throw new IllegalArgumentException("required capability must not be blank");
      }
    }
    dataBoundary = Objects.requireNonNull(dataBoundary, "dataBoundary");
  }

  /** Returns the backwards-compatible unconstrained requirement set. */
  public static RoutingRequirements none() {
    return NONE;
  }

  /** Starts a fluent requirement declaration. */
  public static Builder builder() {
    return new Builder();
  }

  /** Fluent builder. */
  public static final class Builder {
    private final Set<String> requiredCapabilities = new LinkedHashSet<>();
    private RoutingDataBoundary dataBoundary = RoutingDataBoundary.REMOTE_ALLOWED;

    private Builder() {}

    /** Requires one declared model capability. */
    public Builder requireCapability(String value) {
      if (value == null || value.isBlank()) {
        throw new IllegalArgumentException("required capability must not be blank");
      }
      requiredCapabilities.add(value);
      return this;
    }

    /** Sets the permitted data boundary for this request. */
    public Builder dataBoundary(RoutingDataBoundary value) {
      this.dataBoundary = Objects.requireNonNull(value, "dataBoundary");
      return this;
    }

    /** Builds immutable requirements. */
    public RoutingRequirements build() {
      return new RoutingRequirements(requiredCapabilities, dataBoundary);
    }
  }
}
