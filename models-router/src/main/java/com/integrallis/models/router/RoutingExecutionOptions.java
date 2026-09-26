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

import java.math.BigDecimal;
import java.time.Duration;
import java.util.Objects;

/** Per-call execution controls. An options instance can share a budget across many calls. */
public record RoutingExecutionOptions(
    RoutingTokenBounds tokenBounds,
    BigDecimal requestBudget,
    RoutingBudget sharedBudget,
    Duration timeout,
    RoutingCancellationToken cancellation) {
  public RoutingExecutionOptions {
    if (requestBudget != null && requestBudget.signum() < 0) {
      throw new IllegalArgumentException("requestBudget must not be negative");
    }
    if ((requestBudget != null || sharedBudget != null) && tokenBounds == null) {
      throw new IllegalArgumentException("budget enforcement requires complete token bounds");
    }
    if (timeout != null
        && (timeout.isNegative()
            || timeout.isZero()
            || timeout.compareTo(Duration.ofNanos(Long.MAX_VALUE / 2)) > 0)) {
      throw new IllegalArgumentException("timeout must be positive and less than 146 years");
    }
  }

  public static RoutingExecutionOptions unlimited() {
    return builder().build();
  }

  public boolean budgeted() {
    return requestBudget != null || sharedBudget != null;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private RoutingTokenBounds bounds;
    private BigDecimal requestBudget;
    private RoutingBudget sharedBudget;
    private Duration timeout;
    private RoutingCancellationToken cancellation;

    public Builder tokenBounds(RoutingTokenBounds value) {
      bounds = Objects.requireNonNull(value);
      return this;
    }

    public Builder requestBudget(BigDecimal value) {
      requestBudget = Objects.requireNonNull(value);
      return this;
    }

    public Builder sharedBudget(RoutingBudget value) {
      sharedBudget = Objects.requireNonNull(value);
      return this;
    }

    public Builder timeout(Duration value) {
      timeout = Objects.requireNonNull(value);
      return this;
    }

    public Builder cancellation(RoutingCancellationToken value) {
      cancellation = Objects.requireNonNull(value);
      return this;
    }

    public RoutingExecutionOptions build() {
      return new RoutingExecutionOptions(
          bounds, requestBudget, sharedBudget, timeout, cancellation);
    }
  }
}
