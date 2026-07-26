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

import com.integrallis.models.api.SamplingOptions;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Reproducible local-generation controls shared by Models, Ollama, and llama.cpp runs. */
public record RagSamplingProfile(
    double temperature,
    double topP,
    int topK,
    long seed,
    double repetitionPenalty,
    List<String> stopSequences) {

  public RagSamplingProfile {
    stopSequences = List.copyOf(Objects.requireNonNull(stopSequences, "stopSequences"));
    if (!Double.isFinite(temperature) || temperature < 0) {
      throw new IllegalArgumentException("temperature must be finite and non-negative");
    }
    if (!Double.isFinite(topP) || topP <= 0 || topP > 1) {
      throw new IllegalArgumentException("topP must be finite and in (0, 1]");
    }
    if (topK < 1) {
      throw new IllegalArgumentException("topK must be positive");
    }
    if (seed < 0) {
      throw new IllegalArgumentException("seed must be non-negative");
    }
    if (!Double.isFinite(repetitionPenalty) || repetitionPenalty <= 0) {
      throw new IllegalArgumentException("repetitionPenalty must be finite and positive");
    }
    for (String stopSequence : stopSequences) {
      if (stopSequence == null || stopSequence.isEmpty()) {
        throw new IllegalArgumentException("stop sequence must not be null or empty");
      }
    }
  }

  public RagSamplingProfile(
      double temperature, double topP, int topK, long seed, double repetitionPenalty) {
    this(temperature, topP, topK, seed, repetitionPenalty, List.of());
  }

  /** Greedy profile retained as the benchmark default. */
  public static RagSamplingProfile deterministic() {
    return new RagSamplingProfile(0, 1, 1, 42, 1, List.of());
  }

  /** Converts this profile to the Models runtime representation. */
  public SamplingOptions toSamplingOptions(int maxTokens) {
    return SamplingOptions.builder()
        .temperature((float) temperature)
        .topP((float) topP)
        .topK(topK)
        .seed(seed)
        .repetitionPenalty((float) repetitionPenalty)
        .stopSequences(stopSequences)
        .maxTokens(maxTokens)
        .build();
  }

  /** Canonical report values used by the cross-backend comparability gate. */
  public Map<String, String> controls() {
    Map<String, String> controls = new LinkedHashMap<>();
    controls.put("temperature", Double.toString(temperature));
    controls.put("topK", Integer.toString(topK));
    controls.put("topP", Double.toString(topP));
    controls.put("seed", Long.toString(seed));
    controls.put("repetitionPenalty", Double.toString(repetitionPenalty));
    if (!stopSequences.isEmpty()) {
      controls.put("stopSequences", canonicalStopSequences());
    }
    return Map.copyOf(controls);
  }

  private String canonicalStopSequences() {
    return stopSequences.stream()
        .map(RagSamplingProfile::escape)
        .reduce((a, b) -> a + "," + b)
        .orElse("");
  }

  private static String escape(String value) {
    StringBuilder escaped = new StringBuilder(value.length());
    for (int index = 0; index < value.length(); index++) {
      char character = value.charAt(index);
      switch (character) {
        case '\\' -> escaped.append("\\\\");
        case '\n' -> escaped.append("\\n");
        case '\r' -> escaped.append("\\r");
        case '\t' -> escaped.append("\\t");
        case ',' -> escaped.append("\\,");
        default -> escaped.append(character);
      }
    }
    return escaped.toString();
  }
}
