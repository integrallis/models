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

import java.time.Duration;
import java.util.Objects;

/** Runtime and conversation-continuity controls for adaptive routing. */
public final class AdaptiveRoutingOptions {
  private final Duration sessionIdleTimeout;
  private final int maximumSessions;
  private final int minimumTurnsBeforeSwitch;
  private final double switchMargin;
  private final double cacheAffinityWeight;
  private final double residencyWeight;
  private final double loadWeight;
  private final double feedbackWeight;
  private final int consecutiveFailuresBeforeCooldown;
  private final Duration failureCooldown;
  private final boolean toolLoopHardLock;
  private final boolean nonPortableContextHardLock;

  private AdaptiveRoutingOptions(Builder builder) {
    this.sessionIdleTimeout = builder.sessionIdleTimeout;
    this.maximumSessions = builder.maximumSessions;
    this.minimumTurnsBeforeSwitch = builder.minimumTurnsBeforeSwitch;
    this.switchMargin = builder.switchMargin;
    this.cacheAffinityWeight = builder.cacheAffinityWeight;
    this.residencyWeight = builder.residencyWeight;
    this.loadWeight = builder.loadWeight;
    this.feedbackWeight = builder.feedbackWeight;
    this.consecutiveFailuresBeforeCooldown = builder.consecutiveFailuresBeforeCooldown;
    this.failureCooldown = builder.failureCooldown;
    this.toolLoopHardLock = builder.toolLoopHardLock;
    this.nonPortableContextHardLock = builder.nonPortableContextHardLock;
  }

  /** Returns production-oriented defaults with bounded state and conservative switching. */
  public static AdaptiveRoutingOptions defaults() {
    return builder().build();
  }

  /** Starts an options builder. */
  public static Builder builder() {
    return new Builder();
  }

  public Duration sessionIdleTimeout() {
    return sessionIdleTimeout;
  }

  public int maximumSessions() {
    return maximumSessions;
  }

  public int minimumTurnsBeforeSwitch() {
    return minimumTurnsBeforeSwitch;
  }

  public double switchMargin() {
    return switchMargin;
  }

  public double cacheAffinityWeight() {
    return cacheAffinityWeight;
  }

  public double residencyWeight() {
    return residencyWeight;
  }

  public double loadWeight() {
    return loadWeight;
  }

  public double feedbackWeight() {
    return feedbackWeight;
  }

  public int consecutiveFailuresBeforeCooldown() {
    return consecutiveFailuresBeforeCooldown;
  }

  public Duration failureCooldown() {
    return failureCooldown;
  }

  public boolean toolLoopHardLock() {
    return toolLoopHardLock;
  }

  public boolean nonPortableContextHardLock() {
    return nonPortableContextHardLock;
  }

  /** Fluent builder. */
  public static final class Builder {
    private Duration sessionIdleTimeout = Duration.ofMinutes(5);
    private int maximumSessions = 1_024;
    private int minimumTurnsBeforeSwitch = 1;
    private double switchMargin = 0.05;
    private double cacheAffinityWeight = 0.20;
    private double residencyWeight = 0.05;
    private double loadWeight = 0.10;
    private double feedbackWeight = 0.20;
    private int consecutiveFailuresBeforeCooldown = 3;
    private Duration failureCooldown = Duration.ofSeconds(30);
    private boolean toolLoopHardLock = true;
    private boolean nonPortableContextHardLock = true;

    private Builder() {}

    public Builder sessionIdleTimeout(Duration value) {
      this.sessionIdleTimeout = requireDuration(value, "sessionIdleTimeout");
      return this;
    }

    public Builder maximumSessions(int value) {
      if (value < 1) {
        throw new IllegalArgumentException("maximumSessions must be positive");
      }
      this.maximumSessions = value;
      return this;
    }

    public Builder minimumTurnsBeforeSwitch(int value) {
      if (value < 0) {
        throw new IllegalArgumentException("minimumTurnsBeforeSwitch must not be negative");
      }
      this.minimumTurnsBeforeSwitch = value;
      return this;
    }

    public Builder switchMargin(double value) {
      this.switchMargin = requireWeight(value, "switchMargin");
      return this;
    }

    public Builder cacheAffinityWeight(double value) {
      this.cacheAffinityWeight = requireWeight(value, "cacheAffinityWeight");
      return this;
    }

    public Builder residencyWeight(double value) {
      this.residencyWeight = requireWeight(value, "residencyWeight");
      return this;
    }

    public Builder loadWeight(double value) {
      this.loadWeight = requireWeight(value, "loadWeight");
      return this;
    }

    public Builder feedbackWeight(double value) {
      if (!Double.isFinite(value) || value <= 0 || value > 1) {
        throw new IllegalArgumentException("feedbackWeight must be within (0, 1]");
      }
      this.feedbackWeight = value;
      return this;
    }

    public Builder consecutiveFailuresBeforeCooldown(int value) {
      if (value < 1) {
        throw new IllegalArgumentException("consecutiveFailuresBeforeCooldown must be positive");
      }
      this.consecutiveFailuresBeforeCooldown = value;
      return this;
    }

    public Builder failureCooldown(Duration value) {
      this.failureCooldown = requireDuration(value, "failureCooldown");
      return this;
    }

    public Builder toolLoopHardLock(boolean value) {
      this.toolLoopHardLock = value;
      return this;
    }

    public Builder nonPortableContextHardLock(boolean value) {
      this.nonPortableContextHardLock = value;
      return this;
    }

    public AdaptiveRoutingOptions build() {
      return new AdaptiveRoutingOptions(this);
    }

    private static Duration requireDuration(Duration value, String field) {
      Objects.requireNonNull(value, field);
      if (value.isNegative()) {
        throw new IllegalArgumentException(field + " must not be negative");
      }
      return value;
    }

    private static double requireWeight(double value, String field) {
      if (!Double.isFinite(value) || value < 0) {
        throw new IllegalArgumentException(field + " must be finite and non-negative");
      }
      return value;
    }
  }
}
