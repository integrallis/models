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

import com.integrallis.models.runtime.SpeculativeGenerationOptions;
import java.util.List;

/** Versioned, machine-readable benchmark evidence. */
public record BenchmarkReport(
    int schemaVersion,
    String createdAt,
    String backend,
    String backendVersion,
    String modelId,
    String model,
    String artifactSha256,
    long artifactSizeBytes,
    BenchmarkRun run,
    BenchmarkEnvironment environment,
    SpeculativeGenerationOptions speculativeOptions,
    PerformanceSummary summary,
    PerformanceTier performanceTier,
    List<TrialMeasurement> trials) {

  public static final int CURRENT_SCHEMA_VERSION = 4;

  public BenchmarkReport {
    if (speculativeOptions == null) {
      speculativeOptions = SpeculativeGenerationOptions.disabled();
    }
    trials = List.copyOf(trials);
  }

  @Override
  public List<TrialMeasurement> trials() {
    return List.copyOf(trials);
  }
}
