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

import com.integrallis.models.api.BackendDiagnostics;
import java.util.List;
import java.util.Objects;

/** Reproducible machine-readable RAG benchmark report. */
public record RagBenchmarkReport(
    int schemaVersion,
    String generatedAt,
    String framework,
    String backend,
    String backendVersion,
    String modelId,
    String model,
    String artifactSha256,
    long artifactSizeBytes,
    RagBenchmarkSettings settings,
    RagBenchmarkEnvironment environment,
    BackendDiagnostics backendDiagnostics,
    RagBenchmarkSummary summary,
    RagPerformanceTier performanceTier,
    List<RagRun> runs,
    List<RagBenchmarkFailure> failures) {
  public static final int CURRENT_SCHEMA_VERSION = 3;

  public RagBenchmarkReport {
    backendDiagnostics = Objects.requireNonNull(backendDiagnostics, "backendDiagnostics");
    runs = List.copyOf(runs);
    failures = List.copyOf(failures);
  }
}
