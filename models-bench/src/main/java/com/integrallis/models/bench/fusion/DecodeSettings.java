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

import com.integrallis.models.api.SamplingOptions;

/**
 * Decoding controls shared by every arm.
 *
 * @param temperature 0 for greedy argmax; otherwise the public {@code Sampler} is used
 * @param topK 0 for the whole vocabulary
 * @param topP nucleus threshold in (0, 1]
 * @param seed sampler seed
 * @param maxTokens generated-token cap; reaching it marks the output truncated
 * @param tokenLogEvery keep a per-token log entry every N steps; 0 keeps none
 */
public record DecodeSettings(
    float temperature, int topK, float topP, long seed, int maxTokens, int tokenLogEvery) {

  public DecodeSettings {
    if (!Float.isFinite(temperature) || temperature < 0) {
      throw new IllegalArgumentException("temperature must be finite and >= 0");
    }
    if (topK < 0) {
      throw new IllegalArgumentException("top-k must be >= 0");
    }
    if (!(topP > 0 && topP <= 1)) {
      throw new IllegalArgumentException("top-p must be in (0, 1]");
    }
    if (maxTokens < 1) {
      throw new IllegalArgumentException("max tokens must be positive");
    }
    if (tokenLogEvery < 0) {
      throw new IllegalArgumentException("token-log-every must be >= 0");
    }
  }

  DecodeSettings withSeed(long newSeed) {
    return new DecodeSettings(temperature, topK, topP, newSeed, maxTokens, tokenLogEvery);
  }

  boolean greedy() {
    return temperature == 0f;
  }

  SamplingOptions samplingOptions(int vocabulary) {
    return SamplingOptions.builder()
        .temperature(temperature)
        .topK(topK == 0 ? vocabulary : Math.min(topK, vocabulary))
        .topP(topP)
        .seed(seed)
        .repetitionPenalty(1.0f)
        .maxTokens(maxTokens)
        .build();
  }
}
