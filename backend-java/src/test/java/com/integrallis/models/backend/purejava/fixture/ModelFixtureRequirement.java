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
package com.integrallis.models.backend.purejava.fixture;

import java.util.Objects;

/** Immutable selector for one pinned real-model test fixture. */
public final class ModelFixtureRequirement {
  private final String sourceId;
  private final String versionRange;
  private final String variant;
  private final String backend;
  private final String capability;

  private ModelFixtureRequirement(
      String sourceId, String versionRange, String variant, String backend, String capability) {
    this.sourceId = requireText(sourceId, "sourceId");
    this.versionRange = versionRange;
    this.variant = variant;
    this.backend = backend;
    this.capability = capability;
  }

  public static ModelFixtureRequirement of(String sourceId) {
    return new ModelFixtureRequirement(sourceId, null, null, null, null);
  }

  public ModelFixtureRequirement version(String range) {
    return new ModelFixtureRequirement(
        sourceId, requireText(range, "range"), variant, backend, capability);
  }

  public ModelFixtureRequirement variant(String value) {
    return new ModelFixtureRequirement(
        sourceId, versionRange, requireText(value, "variant"), backend, capability);
  }

  public ModelFixtureRequirement backend(String value) {
    return new ModelFixtureRequirement(
        sourceId, versionRange, variant, requireText(value, "backend"), capability);
  }

  public ModelFixtureRequirement capability(String value) {
    return new ModelFixtureRequirement(
        sourceId, versionRange, variant, backend, requireText(value, "capability"));
  }

  boolean matches(ModelFixtureDescriptor descriptor) {
    Objects.requireNonNull(descriptor, "descriptor");
    return sourceId.equals(descriptor.sourceId())
        && matches(variant, descriptor.variant())
        && matches(backend, descriptor.backend())
        && (capability == null || descriptor.capabilities().contains(capability))
        && (versionRange == null || versionMatches(versionRange, descriptor.modelVersion()));
  }

  private static boolean matches(String expected, String actual) {
    return expected == null || expected.equals(actual);
  }

  private static boolean versionMatches(String range, String version) {
    if (!(range.startsWith("[") || range.startsWith("("))
        || !(range.endsWith("]") || range.endsWith(")"))
        || !range.contains(",")) {
      return range.equals(version);
    }
    String[] bounds = range.substring(1, range.length() - 1).split(",", -1);
    if (bounds.length != 2) {
      throw new IllegalArgumentException("Unsupported fixture version range: " + range);
    }
    int lower = bounds[0].isBlank() ? 1 : compareVersions(version, bounds[0]);
    int upper = bounds[1].isBlank() ? -1 : compareVersions(version, bounds[1]);
    boolean lowerMatches = range.startsWith("[") ? lower >= 0 : lower > 0;
    boolean upperMatches = range.endsWith("]") ? upper <= 0 : upper < 0;
    return lowerMatches && upperMatches;
  }

  private static int compareVersions(String left, String right) {
    String[] leftParts = left.split("[.+-]", 4);
    String[] rightParts = right.split("[.+-]", 4);
    int length = Math.max(leftParts.length, rightParts.length);
    for (int index = 0; index < length; index++) {
      int leftPart = index < leftParts.length ? Integer.parseInt(leftParts[index]) : 0;
      int rightPart = index < rightParts.length ? Integer.parseInt(rightParts[index]) : 0;
      int comparison = Integer.compare(leftPart, rightPart);
      if (comparison != 0) {
        return comparison;
      }
    }
    return 0;
  }

  private static String requireText(String value, String name) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value.trim();
  }
}
