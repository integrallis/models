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
package com.integrallis.models.bench;

import java.net.URI;
import java.nio.file.Path;

/** Fully resolved benchmark command configuration. */
public record BenchmarkConfiguration(
    String backend,
    String modelId,
    String model,
    Path artifact,
    URI endpoint,
    String prompt,
    int maxTokens,
    int warmups,
    int iterations,
    int contextLength,
    String backendVersion,
    int threads,
    long backendPid,
    double loadMillis,
    Path output) {

  public BenchmarkConfiguration {
    if (backend == null || backend.isBlank()) {
      throw new IllegalArgumentException("backend must not be blank");
    }
    if (modelId == null || modelId.isBlank()) {
      throw new IllegalArgumentException("modelId must not be blank");
    }
    if (model == null || model.isBlank()) {
      throw new IllegalArgumentException("model must not be blank");
    }
    if (prompt == null || prompt.isBlank()) {
      throw new IllegalArgumentException("prompt must not be blank");
    }
    if (backendVersion == null || backendVersion.isBlank()) {
      throw new IllegalArgumentException("backendVersion must not be blank");
    }
    if (maxTokens <= 0
        || warmups < 0
        || iterations <= 0
        || contextLength <= 0
        || threads <= 0
        || backendPid < 0
        || !Double.isFinite(loadMillis)
        || loadMillis < 0) {
      throw new IllegalArgumentException(
          "token, warmup, iteration, and context values are invalid");
    }
  }
}
