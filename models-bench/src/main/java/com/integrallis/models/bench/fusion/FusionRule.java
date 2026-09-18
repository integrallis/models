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
package com.integrallis.models.bench.fusion;

import java.util.Locale;

/** Token-level combination rules under test. */
public enum FusionRule {
  /** Product of experts: {@code s = Σ wᵢ·log_softmax(logitsᵢ)}. */
  POE,
  /** Mixture of experts: {@code s = log Σ wᵢ·softmax(logitsᵢ)}. */
  MIXTURE,
  /** The published article's rule: {@code s = Σ wᵢ·logitsᵢ} on raw logits. */
  ARTICLE;

  /** Lower-case identifier used on the command line and in reports. */
  public String id() {
    return name().toLowerCase(Locale.ROOT);
  }

  /** Parses {@code poe}, {@code mixture} or {@code article}. */
  public static FusionRule parse(String value) {
    for (FusionRule rule : values()) {
      if (rule.id().equals(value)) {
        return rule;
      }
    }
    throw new IllegalArgumentException("fusion rule must be poe, mixture or article: " + value);
  }
}
