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

import com.integrallis.vectors.core.GgufQ4Kernel;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/** Validated load-time settings selected from deployment overrides and measured recommendations. */
public record PureJavaPlanConfiguration(
    boolean groupedProjections,
    boolean mixedKProjections,
    GgufQ4Kernel q4Kernel,
    int prefillBatchSize,
    boolean finalLayerPrefillPruning,
    boolean finalLayerKvOnlyPrefill,
    boolean batchedAttentionScores,
    boolean batchedAttentionValues) {

  public static final String GROUPED_PROJECTIONS_PROPERTY = "models.purejava.groupedProjections";
  public static final String MIXED_K_PROJECTIONS_PROPERTY = "models.purejava.mixedKProjections";
  public static final String Q4_KERNEL_PROPERTY = "models.purejava.q4Kernel";
  public static final String PREFILL_BATCH_SIZE_PROPERTY = "models.purejava.prefillBatchSize";
  public static final String FINAL_LAYER_PREFILL_PRUNING_PROPERTY =
      "models.purejava.finalLayerPrefillPruning";
  public static final String FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY =
      "models.purejava.finalLayerKvOnlyPrefill";
  public static final String BATCHED_ATTENTION_SCORES_PROPERTY =
      "models.purejava.batchedAttentionScores";
  public static final String BATCHED_ATTENTION_VALUES_PROPERTY =
      "models.purejava.batchedAttentionValues";
  public static final int DEFAULT_PREFILL_BATCH_SIZE = 32;
  private static final String PROPERTY_PREFIX = "models.purejava.";
  private static final Set<String> SUPPORTED_SETTINGS =
      Set.of(
          GROUPED_PROJECTIONS_PROPERTY,
          MIXED_K_PROJECTIONS_PROPERTY,
          Q4_KERNEL_PROPERTY,
          PREFILL_BATCH_SIZE_PROPERTY,
          FINAL_LAYER_PREFILL_PRUNING_PROPERTY,
          FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY,
          BATCHED_ATTENTION_SCORES_PROPERTY,
          BATCHED_ATTENTION_VALUES_PROPERTY);

  public PureJavaPlanConfiguration {
    q4Kernel = Objects.requireNonNull(q4Kernel, "q4Kernel");
    if (prefillBatchSize < 1) {
      throw new IllegalArgumentException(
          PREFILL_BATCH_SIZE_PROPERTY + " must be >= 1: " + prefillBatchSize);
    }
  }

  /** Returns the stable default policy. */
  public static PureJavaPlanConfiguration defaults() {
    return new PureJavaPlanConfiguration(
        true, true, GgufQ4Kernel.WIDENED, DEFAULT_PREFILL_BATCH_SIZE, true, true, false, false);
  }

  /** Reads deployment overrides without running a performance probe. */
  public static PureJavaPlanConfiguration fromSystemProperties(
      Map<String, String> recommendations) {
    Properties systemProperties = System.getProperties();
    Map<String, String> deployment = new LinkedHashMap<>();
    synchronized (systemProperties) {
      SUPPORTED_SETTINGS.stream()
          .sorted()
          .forEach(
              property -> {
                String value = systemProperties.getProperty(property);
                if (value != null) {
                  deployment.put(property, value);
                }
              });
    }
    return from(Map.copyOf(deployment), recommendations);
  }

  static PureJavaPlanConfiguration from(
      Map<String, String> deployment, Map<String, String> recommendations) {
    Objects.requireNonNull(deployment, "deployment");
    Objects.requireNonNull(recommendations, "recommendations");
    validateSettings(deployment, "deployment setting");
    validateSettings(recommendations, "recommendation");
    return new PureJavaPlanConfiguration(
        groupedProjections(configured(GROUPED_PROJECTIONS_PROPERTY, deployment, recommendations)),
        mixedKProjections(configured(MIXED_K_PROJECTIONS_PROPERTY, deployment, recommendations)),
        q4Kernel(configured(Q4_KERNEL_PROPERTY, deployment, recommendations)),
        prefillBatchSize(configured(PREFILL_BATCH_SIZE_PROPERTY, deployment, recommendations)),
        finalLayerPrefillPruning(
            configured(FINAL_LAYER_PREFILL_PRUNING_PROPERTY, deployment, recommendations)),
        finalLayerKvOnlyPrefill(
            configured(FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY, deployment, recommendations)),
        batchedAttentionScores(
            configured(BATCHED_ATTENTION_SCORES_PROPERTY, deployment, recommendations)),
        batchedAttentionValues(
            configured(BATCHED_ATTENTION_VALUES_PROPERTY, deployment, recommendations)));
  }

  private static void validateSettings(Map<String, String> settings, String source) {
    settings.keySet().stream()
        .filter(key -> key.startsWith(PROPERTY_PREFIX))
        .filter(key -> !SUPPORTED_SETTINGS.contains(key))
        .findFirst()
        .ifPresent(
            key -> {
              throw new IllegalArgumentException("Unsupported pure-Java " + source + ": " + key);
            });
  }

  private static String configured(
      String property, Map<String, String> deployment, Map<String, String> recommendations) {
    String configured = deployment.get(property);
    return configured != null ? configured : recommendations.get(property);
  }

  static boolean groupedProjections(String configured) {
    return booleanProperty(GROUPED_PROJECTIONS_PROPERTY, configured);
  }

  static boolean mixedKProjections(String configured) {
    return booleanProperty(MIXED_K_PROJECTIONS_PROPERTY, configured);
  }

  static GgufQ4Kernel q4Kernel(String configured) {
    if (configured == null) {
      return GgufQ4Kernel.WIDENED;
    }
    return switch (configured.trim().toLowerCase(Locale.ROOT)) {
      case "widened" -> GgufQ4Kernel.WIDENED;
      case "short-pairwise" -> GgufQ4Kernel.SHORT_PAIRWISE;
      default ->
          throw new IllegalArgumentException(
              Q4_KERNEL_PROPERTY + " must be widened or short-pairwise: " + configured);
    };
  }

  static boolean finalLayerPrefillPruning(String configured) {
    return booleanProperty(FINAL_LAYER_PREFILL_PRUNING_PROPERTY, configured);
  }

  static boolean finalLayerKvOnlyPrefill(String configured) {
    return booleanProperty(FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY, configured);
  }

  static boolean batchedAttentionScores(String configured) {
    return configured != null && booleanProperty(BATCHED_ATTENTION_SCORES_PROPERTY, configured);
  }

  static boolean batchedAttentionValues(String configured) {
    return configured != null && booleanProperty(BATCHED_ATTENTION_VALUES_PROPERTY, configured);
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
