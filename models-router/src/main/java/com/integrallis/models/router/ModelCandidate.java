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

import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * One model the router may choose, local or hosted.
 *
 * <p>An application registers whatever it can actually reach: ModelJars artifacts it runs in
 * process and hosted providers for which it holds credentials. {@link ModelFleet} binds these
 * descriptors to application-owned clients without adding provider SDKs to this module.
 *
 * <p>For local artifacts the performance and quality figures can come from published ModelJars
 * qualification evidence, which is measured on a recorded host rather than advertised.
 *
 * @param id stable identifier the application maps back to a client
 * @param local whether the model runs in this process
 * @param tags folksonomy labels such as {@code code} or {@code math}
 * @param costPerMillionInputTokens prompt price, zero for local
 * @param costPerMillionOutputTokens completion price, zero for local
 * @param timeToFirstTokenMillis measured or advertised time to first token
 * @param tokensPerSecond sustained decode throughput
 * @param contextWindow maximum tokens the model accepts
 * @param quality per-task quality in [0, 1], keyed by task type
 * @param successRate share of recent calls that succeeded, in [0, 1]
 */
public record ModelCandidate(
    String id,
    boolean local,
    Set<String> tags,
    double costPerMillionInputTokens,
    double costPerMillionOutputTokens,
    long timeToFirstTokenMillis,
    double tokensPerSecond,
    int contextWindow,
    Map<String, Double> quality,
    double successRate) {

  /** Validates and defensively copies the descriptor. */
  public ModelCandidate {
    id = requireText(id);
    tags = Set.copyOf(Objects.requireNonNull(tags, "tags"));
    quality = Map.copyOf(Objects.requireNonNull(quality, "quality"));
    requireNonNegative(costPerMillionInputTokens, "costPerMillionInputTokens");
    requireNonNegative(costPerMillionOutputTokens, "costPerMillionOutputTokens");
    if (timeToFirstTokenMillis < 0) {
      throw new IllegalArgumentException("timeToFirstTokenMillis must not be negative");
    }
    if (tokensPerSecond <= 0) {
      throw new IllegalArgumentException("tokensPerSecond must be positive");
    }
    if (contextWindow < 1) {
      throw new IllegalArgumentException("contextWindow must be positive");
    }
    quality.forEach(
        (task, value) -> {
          if (!Double.isFinite(value) || value < 0 || value > 1) {
            throw new IllegalArgumentException("quality for " + task + " must be within [0, 1]");
          }
        });
    if (!Double.isFinite(successRate) || successRate < 0 || successRate > 1) {
      throw new IllegalArgumentException("successRate must be within [0, 1]");
    }
  }

  /**
   * Returns the quality recorded for one task type.
   *
   * @param taskType task classification, may be null
   * @return recorded quality, or zero when the model declares none for that task
   */
  public double qualityFor(String taskType) {
    if (taskType == null) {
      return quality.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    return quality.getOrDefault(taskType, 0.0);
  }

  /**
   * Blended price used for cost comparison.
   *
   * <p>Weighted three-to-one toward output tokens, because completions dominate spend on the chat
   * and agent workloads this routes.
   *
   * @return blended price per million tokens
   */
  public double blendedCostPerMillionTokens() {
    return (costPerMillionInputTokens + 3.0 * costPerMillionOutputTokens) / 4.0;
  }

  /**
   * Starts building a candidate.
   *
   * @param id stable model identifier
   * @return a new builder
   */
  public static Builder builder(String id) {
    return new Builder(id);
  }

  private static String requireText(String value) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("id must not be blank");
    }
    return value;
  }

  private static void requireNonNegative(double value, String field) {
    if (!Double.isFinite(value) || value < 0) {
      throw new IllegalArgumentException(field + " must be finite and non-negative");
    }
  }

  /** Fluent builder, so adding a dimension later does not break every call site. */
  public static final class Builder {
    private final String id;
    private boolean local;
    private Set<String> tags = Set.of();
    private double inputCost;
    private double outputCost;
    private long ttftMillis = 1_000;
    private double tokensPerSecond = 1.0;
    private int contextWindow = 8_192;
    private Map<String, Double> quality = Map.of();
    private double successRate = 1.0;

    private Builder(String id) {
      this.id = id;
    }

    /**
     * Marks the model as running in this process.
     *
     * @param value whether the model is local
     * @return this builder
     */
    public Builder local(boolean value) {
      this.local = value;
      return this;
    }

    /**
     * Sets folksonomy labels.
     *
     * @param value tags such as {@code code}
     * @return this builder
     */
    public Builder tags(Set<String> value) {
      // Copy here as well as in the record: the builder would otherwise hold the caller's
      // collection between this call and build(), where a mutation would still land.
      this.tags = Set.copyOf(value);
      return this;
    }

    /**
     * Sets prompt and completion pricing.
     *
     * @param input price per million prompt tokens
     * @param output price per million completion tokens
     * @return this builder
     */
    public Builder costPerMillionTokens(double input, double output) {
      this.inputCost = input;
      this.outputCost = output;
      return this;
    }

    /**
     * Sets time to first token.
     *
     * @param value milliseconds
     * @return this builder
     */
    public Builder timeToFirstTokenMillis(long value) {
      this.ttftMillis = value;
      return this;
    }

    /**
     * Sets sustained decode throughput.
     *
     * @param value tokens per second
     * @return this builder
     */
    public Builder tokensPerSecond(double value) {
      this.tokensPerSecond = value;
      return this;
    }

    /**
     * Sets the usable context window.
     *
     * @param value tokens
     * @return this builder
     */
    public Builder contextWindow(int value) {
      this.contextWindow = value;
      return this;
    }

    /**
     * Sets per-task quality.
     *
     * @param value quality in [0, 1] keyed by task type
     * @return this builder
     */
    public Builder quality(Map<String, Double> value) {
      this.quality = Map.copyOf(value);
      return this;
    }

    /**
     * Sets the observed success rate.
     *
     * @param value share of recent calls that succeeded
     * @return this builder
     */
    public Builder successRate(double value) {
      this.successRate = value;
      return this;
    }

    /**
     * Builds the candidate.
     *
     * @return an immutable descriptor
     */
    public ModelCandidate build() {
      return new ModelCandidate(
          id,
          local,
          tags,
          inputCost,
          outputCost,
          ttftMillis,
          tokensPerSecond,
          contextWindow,
          quality,
          successRate);
    }
  }
}
