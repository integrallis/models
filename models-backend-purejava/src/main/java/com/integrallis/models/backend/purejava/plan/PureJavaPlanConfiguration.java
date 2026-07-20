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
package com.integrallis.models.backend.purejava.plan;

/** Explicit deployment overrides consumed once when a backend is loaded. */
public record PureJavaPlanConfiguration(
    boolean groupedProjections,
    boolean mixedKProjections,
    int prefillBatchSize,
    boolean finalLayerPrefillPruning) {

  public static final String GROUPED_PROJECTIONS_PROPERTY = "models.purejava.groupedProjections";
  public static final String MIXED_K_PROJECTIONS_PROPERTY = "models.purejava.mixedKProjections";
  public static final String PREFILL_BATCH_SIZE_PROPERTY = "models.purejava.prefillBatchSize";
  public static final String FINAL_LAYER_PREFILL_PRUNING_PROPERTY =
      "models.purejava.finalLayerPrefillPruning";
  public static final int DEFAULT_PREFILL_BATCH_SIZE = 32;

  public PureJavaPlanConfiguration {
    if (prefillBatchSize < 1) {
      throw new IllegalArgumentException(
          PREFILL_BATCH_SIZE_PROPERTY + " must be >= 1: " + prefillBatchSize);
    }
  }

  /** Returns the stable default policy. */
  public static PureJavaPlanConfiguration defaults() {
    return new PureJavaPlanConfiguration(true, true, DEFAULT_PREFILL_BATCH_SIZE, true);
  }

  /** Reads deployment overrides without running a performance probe. */
  public static PureJavaPlanConfiguration fromSystemProperties() {
    return new PureJavaPlanConfiguration(
        groupedProjections(System.getProperty(GROUPED_PROJECTIONS_PROPERTY)),
        mixedKProjections(System.getProperty(MIXED_K_PROJECTIONS_PROPERTY)),
        prefillBatchSize(System.getProperty(PREFILL_BATCH_SIZE_PROPERTY)),
        finalLayerPrefillPruning(System.getProperty(FINAL_LAYER_PREFILL_PRUNING_PROPERTY)));
  }

  static boolean groupedProjections(String configured) {
    return booleanProperty(GROUPED_PROJECTIONS_PROPERTY, configured);
  }

  static boolean mixedKProjections(String configured) {
    return booleanProperty(MIXED_K_PROJECTIONS_PROPERTY, configured);
  }

  static boolean finalLayerPrefillPruning(String configured) {
    return booleanProperty(FINAL_LAYER_PREFILL_PRUNING_PROPERTY, configured);
  }

  private static boolean booleanProperty(String property, String configured) {
    if (configured == null || configured.equalsIgnoreCase("true")) {
      return true;
    }
    if (configured.equalsIgnoreCase("false")) {
      return false;
    }
    throw new IllegalArgumentException(property + " must be true or false: " + configured);
  }

  static int prefillBatchSize(String configured) {
    if (configured == null) {
      return DEFAULT_PREFILL_BATCH_SIZE;
    }
    try {
      int batchSize = Integer.parseInt(configured);
      if (batchSize >= 1) {
        return batchSize;
      }
    } catch (NumberFormatException ignored) {
      // Report one stable configuration error below.
    }
    throw new IllegalArgumentException(
        PREFILL_BATCH_SIZE_PROPERTY + " must be a positive integer: " + configured);
  }
}
