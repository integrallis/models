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
package com.integrallis.models.runtime;

/** Configuration for model-free n-gram speculative generation. */
public record SpeculativeGenerationOptions(
    boolean enabled,
    int ngramSize,
    int minimumDraftTokens,
    int maximumDraftTokens,
    int historyWindow,
    int adaptationWindow,
    float minimumAcceptanceRate,
    int cooldownTokens) {

  public SpeculativeGenerationOptions {
    if (ngramSize < 2) {
      throw new IllegalArgumentException("ngramSize must be >= 2: " + ngramSize);
    }
    if (minimumDraftTokens < 1) {
      throw new IllegalArgumentException("minimumDraftTokens must be >= 1: " + minimumDraftTokens);
    }
    if (maximumDraftTokens < minimumDraftTokens) {
      throw new IllegalArgumentException(
          "maximumDraftTokens must be >= minimumDraftTokens: "
              + maximumDraftTokens
              + " < "
              + minimumDraftTokens);
    }
    if (historyWindow < ngramSize + minimumDraftTokens) {
      throw new IllegalArgumentException(
          "historyWindow must hold an n-gram and minimum draft: " + historyWindow);
    }
    if (adaptationWindow < 1) {
      throw new IllegalArgumentException("adaptationWindow must be >= 1: " + adaptationWindow);
    }
    if (minimumAcceptanceRate < 0.0f || minimumAcceptanceRate > 1.0f) {
      throw new IllegalArgumentException(
          "minimumAcceptanceRate must be in [0, 1]: " + minimumAcceptanceRate);
    }
    if (cooldownTokens < 1) {
      throw new IllegalArgumentException("cooldownTokens must be >= 1: " + cooldownTokens);
    }
  }

  /** Returns disabled speculation using otherwise valid defaults. */
  public static SpeculativeGenerationOptions disabled() {
    return new Builder().enabled(false).build();
  }

  /** Returns an enabled n-gram speculation builder. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder with defaults chosen for a maximum target-verification batch of eight tokens. */
  public static final class Builder {
    private boolean enabled = true;
    private int ngramSize = 4;
    private int minimumDraftTokens = 3;
    private int maximumDraftTokens = 7;
    private int historyWindow = 2048;
    private int adaptationWindow = 1;
    private float minimumAcceptanceRate = 0.8f;
    private int cooldownTokens = 256;

    private Builder() {}

    public Builder enabled(boolean enabled) {
      this.enabled = enabled;
      return this;
    }

    public Builder ngramSize(int ngramSize) {
      this.ngramSize = ngramSize;
      return this;
    }

    public Builder minimumDraftTokens(int minimumDraftTokens) {
      this.minimumDraftTokens = minimumDraftTokens;
      return this;
    }

    public Builder maximumDraftTokens(int maximumDraftTokens) {
      this.maximumDraftTokens = maximumDraftTokens;
      return this;
    }

    public Builder historyWindow(int historyWindow) {
      this.historyWindow = historyWindow;
      return this;
    }

    public Builder adaptationWindow(int adaptationWindow) {
      this.adaptationWindow = adaptationWindow;
      return this;
    }

    public Builder minimumAcceptanceRate(float minimumAcceptanceRate) {
      this.minimumAcceptanceRate = minimumAcceptanceRate;
      return this;
    }

    public Builder cooldownTokens(int cooldownTokens) {
      this.cooldownTokens = cooldownTokens;
      return this;
    }

    public SpeculativeGenerationOptions build() {
      return new SpeculativeGenerationOptions(
          enabled,
          ngramSize,
          minimumDraftTokens,
          maximumDraftTokens,
          historyWindow,
          adaptationWindow,
          minimumAcceptanceRate,
          cooldownTokens);
    }
  }
}
