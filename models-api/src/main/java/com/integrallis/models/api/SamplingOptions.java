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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable configuration for token sampling strategies.
 *
 * <p>Floating-point controls are finite. A zero temperature selects deterministic greedy sampling;
 * {@code seed} is nullable and absent when the caller accepts a runtime-selected seed.
 */
public record SamplingOptions(
    float temperature,
    float topP,
    int topK,
    int maxTokens,
    Long seed,
    float repetitionPenalty,
    List<String> stopSequences) {

  public SamplingOptions {
    stopSequences = List.copyOf(Objects.requireNonNull(stopSequences, "stopSequences"));
    if (!Float.isFinite(temperature) || temperature < 0) {
      throw new IllegalArgumentException(
          "temperature must be finite and >= 0, got: " + temperature);
    }
    if (!Float.isFinite(topP) || topP <= 0 || topP > 1.0f) {
      throw new IllegalArgumentException("topP must be finite and in (0, 1], got: " + topP);
    }
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0, got: " + topK);
    }
    if (maxTokens <= 0) {
      throw new IllegalArgumentException("maxTokens must be > 0, got: " + maxTokens);
    }
    if (!Float.isFinite(repetitionPenalty) || repetitionPenalty < 1.0f) {
      throw new IllegalArgumentException(
          "repetitionPenalty must be finite and >= 1.0, got: " + repetitionPenalty);
    }
    for (String stopSequence : stopSequences) {
      if (stopSequence == null || stopSequence.isEmpty()) {
        throw new IllegalArgumentException("stop sequence must not be null or empty");
      }
    }
  }

  /** Returns a new builder with default values. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link SamplingOptions}. */
  public static final class Builder {
    private Float temperature;
    private Float topP;
    private Integer topK;
    private Integer maxTokens;
    private Long seed;
    private Float repetitionPenalty;
    private final List<String> stopSequences = new ArrayList<>();

    Builder() {}

    public Builder temperature(float temperature) {
      this.temperature = temperature;
      return this;
    }

    public Builder topP(float topP) {
      this.topP = topP;
      return this;
    }

    public Builder topK(int topK) {
      this.topK = topK;
      return this;
    }

    public Builder maxTokens(int maxTokens) {
      this.maxTokens = maxTokens;
      return this;
    }

    public Builder seed(long seed) {
      this.seed = seed;
      return this;
    }

    public Builder repetitionPenalty(float repetitionPenalty) {
      this.repetitionPenalty = repetitionPenalty;
      return this;
    }

    public Builder stopSequence(String stopSequence) {
      stopSequences.add(stopSequence);
      return this;
    }

    public Builder stopSequences(List<String> stopSequences) {
      this.stopSequences.clear();
      this.stopSequences.addAll(Objects.requireNonNull(stopSequences, "stopSequences"));
      return this;
    }

    public SamplingOptions build() {
      return new SamplingOptions(
          temperature != null ? temperature : 1.0f,
          topP != null ? topP : 0.9f,
          topK != null ? topK : 40,
          maxTokens != null ? maxTokens : 256,
          seed,
          repetitionPenalty != null ? repetitionPenalty : 1.0f,
          stopSequences);
    }
  }
}
