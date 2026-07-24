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
package com.integrallis.models.rag;

import java.util.Map;
import java.util.Objects;

/** Immutable hosted-API pricing snapshot embedded in a benchmark report. */
public record HostedApiPricing(
    String provider,
    String model,
    double inputUsdPerMillionTokens,
    double cacheReadUsdPerMillionTokens,
    double cacheWriteUsdPerMillionTokens,
    double outputUsdPerMillionTokens,
    String effectiveDate,
    String sourceUrl) {
  private static final Map<String, HostedApiPricing> PROFILES =
      Map.of(
          key("openai", "gpt-5.4-nano-2026-03-17"),
          new HostedApiPricing(
              "openai",
              "gpt-5.4-nano-2026-03-17",
              0.20,
              0.02,
              0.20,
              1.25,
              "2026-07-23",
              "https://developers.openai.com/api/docs/models/gpt-5.4-nano"),
          key("anthropic", "claude-haiku-4-5-20251001"),
          new HostedApiPricing(
              "anthropic",
              "claude-haiku-4-5-20251001",
              1.00,
              0.10,
              1.25,
              5.00,
              "2026-07-23",
              "https://platform.claude.com/docs/en/about-claude/pricing"),
          key("deepseek", "deepseek-v4-flash"),
          new HostedApiPricing(
              "deepseek",
              "deepseek-v4-flash",
              0.14,
              0.0028,
              0.14,
              0.28,
              "2026-07-23",
              "https://api-docs.deepseek.com/quick_start/pricing/"));

  public HostedApiPricing {
    provider = requireText(provider, "provider");
    model = requireText(model, "model");
    requireNonNegative(inputUsdPerMillionTokens, "inputUsdPerMillionTokens");
    requireNonNegative(cacheReadUsdPerMillionTokens, "cacheReadUsdPerMillionTokens");
    requireNonNegative(cacheWriteUsdPerMillionTokens, "cacheWriteUsdPerMillionTokens");
    requireNonNegative(outputUsdPerMillionTokens, "outputUsdPerMillionTokens");
    effectiveDate = requireText(effectiveDate, "effectiveDate");
    sourceUrl = requireText(sourceUrl, "sourceUrl");
  }

  /** Returns the pinned pricing snapshot for a certified hosted comparison model. */
  public static HostedApiPricing forModel(String provider, String model) {
    HostedApiPricing pricing = PROFILES.get(key(provider, model));
    if (pricing == null) {
      throw new IllegalArgumentException(
          "no pinned hosted pricing for provider/model: " + provider + "/" + model);
    }
    return pricing;
  }

  /** Estimates the API charge for normalized token usage. */
  public double estimateUsd(
      int inputTokens, int cacheReadInputTokens, int cacheWriteInputTokens, int outputTokens) {
    if (inputTokens < 0
        || cacheReadInputTokens < 0
        || cacheWriteInputTokens < 0
        || outputTokens < 0) {
      throw new IllegalArgumentException("token counts must be non-negative");
    }
    int uncachedInputTokens = inputTokens - cacheReadInputTokens - cacheWriteInputTokens;
    if (uncachedInputTokens < 0) {
      throw new IllegalArgumentException("cache token counts must not exceed total input tokens");
    }
    double million = 1_000_000.0;
    return (uncachedInputTokens * inputUsdPerMillionTokens
            + cacheReadInputTokens * cacheReadUsdPerMillionTokens
            + cacheWriteInputTokens * cacheWriteUsdPerMillionTokens
            + outputTokens * outputUsdPerMillionTokens)
        / million;
  }

  private static String key(String provider, String model) {
    return requireText(provider, "provider") + '\u0000' + requireText(model, "model");
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value.trim();
  }

  private static void requireNonNegative(double value, String name) {
    if (!Double.isFinite(value) || value < 0) {
      throw new IllegalArgumentException(name + " must be finite and non-negative");
    }
  }
}
