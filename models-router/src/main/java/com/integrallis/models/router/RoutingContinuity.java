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

import java.util.LinkedHashMap;
import java.util.Map;

/** Request-time facts that determine whether conversation state can safely move between models. */
public record RoutingContinuity(
    boolean activeToolLoop, boolean contextPortable, Map<String, Integer> cachedPrefixTokens) {
  private static final RoutingContinuity NONE = new RoutingContinuity(false, true, Map.of());

  /** Validates and defensively copies the evidence. */
  public RoutingContinuity {
    cachedPrefixTokens = Map.copyOf(cachedPrefixTokens);
    cachedPrefixTokens.forEach(
        (model, tokens) -> {
          if (model == null || model.isBlank()) {
            throw new IllegalArgumentException("cached-prefix model id must not be blank");
          }
          if (tokens == null || tokens < 0) {
            throw new IllegalArgumentException("cached-prefix tokens must not be negative");
          }
        });
  }

  /** Returns portable, non-tool-loop continuity with no cache evidence. */
  public static RoutingContinuity none() {
    return NONE;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Fluent builder. */
  public static final class Builder {
    private boolean activeToolLoop;
    private boolean contextPortable = true;
    private final Map<String, Integer> cachedPrefixTokens = new LinkedHashMap<>();

    private Builder() {}

    public Builder activeToolLoop(boolean value) {
      this.activeToolLoop = value;
      return this;
    }

    public Builder contextPortable(boolean value) {
      this.contextPortable = value;
      return this;
    }

    public Builder cachedPrefixTokens(String modelId, int tokens) {
      if (tokens < 0) {
        throw new IllegalArgumentException("cached-prefix tokens must not be negative");
      }
      cachedPrefixTokens.put(modelId, tokens);
      return this;
    }

    public RoutingContinuity build() {
      return new RoutingContinuity(activeToolLoop, contextPortable, cachedPrefixTokens);
    }
  }
}
