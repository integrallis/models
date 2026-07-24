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

import java.util.List;
import java.util.Map;

/** Auditable result of applying absolute and same-host relative production gates. */
public record RagProductionQualification(
    String modelId,
    String candidateBackend,
    String artifactSha256,
    RagPerformanceTier absoluteTier,
    RagQualificationVerdict verdict,
    boolean qualified,
    List<String> qualifyingComparators,
    List<RagComparatorAssessment> comparisons,
    Map<String, String> exclusions) {
  public RagProductionQualification {
    qualifyingComparators = List.copyOf(qualifyingComparators);
    comparisons = List.copyOf(comparisons);
    exclusions = Map.copyOf(exclusions);
  }
}
