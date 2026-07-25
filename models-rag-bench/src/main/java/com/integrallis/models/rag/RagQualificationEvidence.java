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
import java.util.Objects;

/** File-hashed inputs and policy result for one reproducible model qualification. */
public record RagQualificationEvidence(
    int schemaVersion,
    String policyId,
    RagQualificationReportReference candidate,
    List<RagQualificationReportReference> comparators,
    RagProductionQualification qualification) {
  public static final int CURRENT_SCHEMA_VERSION = 1;

  public RagQualificationEvidence {
    if (schemaVersion != CURRENT_SCHEMA_VERSION) {
      throw new IllegalArgumentException(
          "schemaVersion must be " + CURRENT_SCHEMA_VERSION + ": " + schemaVersion);
    }
    Objects.requireNonNull(policyId, "policyId");
    Objects.requireNonNull(candidate, "candidate");
    comparators = List.copyOf(Objects.requireNonNull(comparators, "comparators"));
    Objects.requireNonNull(qualification, "qualification");
  }
}
