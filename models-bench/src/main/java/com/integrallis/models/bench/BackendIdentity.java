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

import java.util.Map;

/** Runtime identity and artifact attestation for one inference backend. */
public record BackendIdentity(
    String name,
    String version,
    String build,
    String modelReference,
    String runtimeArtifact,
    String runtimeArtifactSha256,
    boolean artifactVerified,
    Map<String, String> settings) {

  public BackendIdentity {
    settings = settings == null ? Map.of() : Map.copyOf(settings);
  }
}
