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

import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;

/** Explicit outcome and measurements from one routed model call. */
public final class RoutingFeedback {
  private final String modelId;
  private final boolean successful;
  private final String sessionId;
  private final String taskType;
  private final Long timeToFirstTokenMillis;
  private final Double tokensPerSecond;
  private final Integer promptTokens;
  private final Integer cachedInputTokens;

  private RoutingFeedback(Builder builder) {
    this.modelId = requireText(builder.modelId, "modelId");
    this.successful = builder.successful;
    this.sessionId = builder.sessionId;
    this.taskType = builder.taskType;
    this.timeToFirstTokenMillis = builder.timeToFirstTokenMillis;
    this.tokensPerSecond = builder.tokensPerSecond;
    this.promptTokens = builder.promptTokens;
    this.cachedInputTokens = builder.cachedInputTokens;
  }

  public static Builder success(String modelId) {
    return new Builder(modelId, true);
  }

  public static Builder failure(String modelId) {
    return new Builder(modelId, false);
  }

  public String modelId() {
    return modelId;
  }

  public boolean successful() {
    return successful;
  }

  public Optional<String> sessionId() {
    return Optional.ofNullable(sessionId);
  }

  public Optional<String> taskType() {
    return Optional.ofNullable(taskType);
  }

  public OptionalLong timeToFirstTokenMillis() {
    return timeToFirstTokenMillis == null
        ? OptionalLong.empty()
        : OptionalLong.of(timeToFirstTokenMillis);
  }

  public OptionalDouble tokensPerSecond() {
    return tokensPerSecond == null ? OptionalDouble.empty() : OptionalDouble.of(tokensPerSecond);
  }

  public OptionalInt promptTokens() {
    return promptTokens == null ? OptionalInt.empty() : OptionalInt.of(promptTokens);
  }

  public OptionalInt cachedInputTokens() {
    return cachedInputTokens == null ? OptionalInt.empty() : OptionalInt.of(cachedInputTokens);
  }

  private static String requireText(String value, String field) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(field + " must not be blank");
    }
    return value;
  }

  /** Fluent builder. */
  public static final class Builder {
    private final String modelId;
    private final boolean successful;
    private String sessionId;
    private String taskType;
    private Long timeToFirstTokenMillis;
    private Double tokensPerSecond;
    private Integer promptTokens;
    private Integer cachedInputTokens;

    private Builder(String modelId, boolean successful) {
      this.modelId = modelId;
      this.successful = successful;
    }

    public Builder sessionId(String value) {
      this.sessionId = requireText(value, "sessionId");
      return this;
    }

    public Builder taskType(String value) {
      this.taskType = requireText(value, "taskType");
      return this;
    }

    public Builder timeToFirstTokenMillis(long value) {
      if (value < 0) {
        throw new IllegalArgumentException("timeToFirstTokenMillis must not be negative");
      }
      this.timeToFirstTokenMillis = value;
      return this;
    }

    public Builder tokensPerSecond(double value) {
      if (!Double.isFinite(value) || value <= 0) {
        throw new IllegalArgumentException("tokensPerSecond must be finite and positive");
      }
      this.tokensPerSecond = value;
      return this;
    }

    public Builder tokenUsage(int prompt, int cached) {
      if (prompt < 0 || cached < 0 || cached > prompt) {
        throw new IllegalArgumentException(
            "token usage requires 0 <= cachedInputTokens <= promptTokens");
      }
      this.promptTokens = prompt;
      this.cachedInputTokens = cached;
      return this;
    }

    public RoutingFeedback build() {
      return new RoutingFeedback(this);
    }
  }
}
