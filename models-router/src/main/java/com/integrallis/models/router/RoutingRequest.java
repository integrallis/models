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

import java.util.Objects;
import java.util.Optional;

/**
 * One routing request.
 *
 * @param query the user's text
 * @param estimatedTokens prompt size used to exclude models whose context cannot hold it
 * @param taskTypeOverride skips classification when the caller already knows the task
 * @param sessionId pins a session to its first choice, preserving provider prompt caches
 */
public record RoutingRequest(
    String query, int estimatedTokens, String taskTypeOverride, String sessionId) {

  /** Validates the request. */
  public RoutingRequest {
    Objects.requireNonNull(query, "query");
    if (estimatedTokens < 0) {
      throw new IllegalArgumentException("estimatedTokens must not be negative");
    }
  }

  /**
   * Returns the caller-supplied task type, when present.
   *
   * @return the override
   */
  public Optional<String> taskType() {
    return Optional.ofNullable(taskTypeOverride);
  }

  /**
   * Returns the session identifier, when present.
   *
   * @return the session id
   */
  public Optional<String> session() {
    return Optional.ofNullable(sessionId);
  }

  /**
   * Starts building a request.
   *
   * @param query the user's text
   * @return a new builder
   */
  public static Builder builder(String query) {
    return new Builder(query);
  }

  /** Fluent builder. */
  public static final class Builder {
    private final String query;
    private int estimatedTokens;
    private String taskType;
    private String sessionId;

    private Builder(String query) {
      this.query = query;
    }

    /**
     * Sets the estimated prompt size.
     *
     * @param value tokens
     * @return this builder
     */
    public Builder estimatedTokens(int value) {
      this.estimatedTokens = value;
      return this;
    }

    /**
     * Skips classification.
     *
     * @param value known task type
     * @return this builder
     */
    public Builder taskType(String value) {
      this.taskType = value;
      return this;
    }

    /**
     * Pins a conversation to its first choice.
     *
     * @param value session identifier
     * @return this builder
     */
    public Builder sessionId(String value) {
      this.sessionId = value;
      return this;
    }

    /**
     * Builds the request.
     *
     * @return an immutable request
     */
    public RoutingRequest build() {
      return new RoutingRequest(query, estimatedTokens, taskType, sessionId);
    }
  }
}
