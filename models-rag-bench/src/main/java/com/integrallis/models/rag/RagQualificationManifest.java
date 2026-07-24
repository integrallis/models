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

/** Machine-generated launch manifest whose entries are derived from immutable raw reports. */
public record RagQualificationManifest(
    int schemaVersion,
    String generatedAt,
    String policyVersion,
    String modelsRevision,
    int targetQualifiedModels,
    int qualifiedModels,
    int rejectedModels,
    List<RagQualificationManifestEntry> entries) {
  public static final int CURRENT_SCHEMA_VERSION = 1;
  public static final String POLICY_VERSION = GroundedAnswerPolicy.POLICY_ID;

  public RagQualificationManifest {
    entries = List.copyOf(entries);
  }

  /** Fails the launch gate when fewer distinct models qualified than requested. */
  public void requireTarget() {
    if (qualifiedModels < targetQualifiedModels) {
      throw new IllegalStateException(
          qualifiedModels + " distinct models qualified; target is " + targetQualifiedModels);
    }
  }
}
