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
import com.integrallis.vectors.core.GgufQ6BatchedKernel;
import com.integrallis.vectors.core.GgufQ8BlockMajorKernel;
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
    GgufQ6BatchedKernel q6BatchedKernel,
    int prefillBatchSize,
    boolean finalLayerPrefillPruning,
    boolean finalLayerKvOnlyPrefill,
    boolean batchedAttentionScores,
    boolean batchedAttentionValues,
    boolean fusedGroupedAttention,
    boolean stagedQuantizedFfn,
    boolean stagedQuantizedLayer,
    boolean blockMajorQ8Activations,
    GgufQ8BlockMajorKernel q8BlockMajorKernel,
    boolean parallelQ8FfnPreparation,
    int maxContextLength) {

  public static final String GROUPED_PROJECTIONS_PROPERTY = "models.purejava.groupedProjections";
  public static final String MIXED_K_PROJECTIONS_PROPERTY = "models.purejava.mixedKProjections";
  public static final String Q4_KERNEL_PROPERTY = "models.purejava.q4Kernel";
  public static final String Q6_BATCHED_KERNEL_PROPERTY = "models.purejava.q6BatchedKernel";
  public static final String PREFILL_BATCH_SIZE_PROPERTY = "models.purejava.prefillBatchSize";

  /**
   * The context a session is sized for. Absent, a session is sized for the model's own maximum,
   * which for a long-context model is tens of gigabytes of key and value cache that a short-prompt
   * workload never reaches. A ModelJar that knows its prompts are short recommends a bound here.
   */
  public static final String MAX_CONTEXT_LENGTH_PROPERTY = "models.purejava.maxContextLength";

  /** Sentinel meaning "size the session for the model's own maximum". */
  public static final int MODEL_MAXIMUM_CONTEXT = 0;

  public static final String FINAL_LAYER_PREFILL_PRUNING_PROPERTY =
      "models.purejava.finalLayerPrefillPruning";
  public static final String FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY =
      "models.purejava.finalLayerKvOnlyPrefill";
  public static final String BATCHED_ATTENTION_SCORES_PROPERTY =
      "models.purejava.batchedAttentionScores";
  public static final String BATCHED_ATTENTION_VALUES_PROPERTY =
      "models.purejava.batchedAttentionValues";
  public static final String FUSED_GROUPED_ATTENTION_PROPERTY =
      "models.purejava.fusedGroupedAttention";
  public static final String STAGED_QUANTIZED_FFN_PROPERTY = "models.purejava.stagedQuantizedFfn";
  public static final String STAGED_QUANTIZED_LAYER_PROPERTY =
      "models.purejava.stagedQuantizedLayer";
  public static final String BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY =
      "models.purejava.blockMajorQ8Activations";
  public static final String Q8_BLOCK_MAJOR_KERNEL_PROPERTY = "models.purejava.q8BlockMajorKernel";
  public static final String PARALLEL_Q8_FFN_PREPARATION_PROPERTY =
      "models.purejava.parallelQ8FfnPreparation";
  public static final int DEFAULT_PREFILL_BATCH_SIZE = 32;
  private static final String PROPERTY_PREFIX = "models.purejava.";
  private static final Set<String> SUPPORTED_SETTINGS =
      Set.of(
          GROUPED_PROJECTIONS_PROPERTY,
          MIXED_K_PROJECTIONS_PROPERTY,
          Q4_KERNEL_PROPERTY,
          Q6_BATCHED_KERNEL_PROPERTY,
          PREFILL_BATCH_SIZE_PROPERTY,
          MAX_CONTEXT_LENGTH_PROPERTY,
          FINAL_LAYER_PREFILL_PRUNING_PROPERTY,
          FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY,
          BATCHED_ATTENTION_SCORES_PROPERTY,
          BATCHED_ATTENTION_VALUES_PROPERTY,
          FUSED_GROUPED_ATTENTION_PROPERTY,
          STAGED_QUANTIZED_FFN_PROPERTY,
          STAGED_QUANTIZED_LAYER_PROPERTY,
          BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY,
          Q8_BLOCK_MAJOR_KERNEL_PROPERTY,
          PARALLEL_Q8_FFN_PREPARATION_PROPERTY);

  public PureJavaPlanConfiguration {
    q4Kernel = Objects.requireNonNull(q4Kernel, "q4Kernel");
    q6BatchedKernel = Objects.requireNonNull(q6BatchedKernel, "q6BatchedKernel");
    q8BlockMajorKernel = Objects.requireNonNull(q8BlockMajorKernel, "q8BlockMajorKernel");
    if (prefillBatchSize < 1) {
      throw new IllegalArgumentException(
          PREFILL_BATCH_SIZE_PROPERTY + " must be >= 1: " + prefillBatchSize);
    }
  }

  /** Returns the stable default policy. */
  public static PureJavaPlanConfiguration defaults() {
    // Delegates to the resolvers rather than repeating their values. This used to be a second,
    // hand-maintained copy of every default, and the two drifted the moment one moved: six settings
    // became enabled-when-unset in the resolvers on 2026-10-08 while this method still returned the
    // old disabled values, so `defaults()` and `from(Map.of(), Map.of())` disagreed. One definition
    // cannot drift from itself.
    return from(Map.of(), Map.of());
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
        q6BatchedKernel(configured(Q6_BATCHED_KERNEL_PROPERTY, deployment, recommendations)),
        prefillBatchSize(configured(PREFILL_BATCH_SIZE_PROPERTY, deployment, recommendations)),
        finalLayerPrefillPruning(
            configured(FINAL_LAYER_PREFILL_PRUNING_PROPERTY, deployment, recommendations)),
        finalLayerKvOnlyPrefill(
            configured(FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY, deployment, recommendations)),
        batchedAttentionScores(
            configured(BATCHED_ATTENTION_SCORES_PROPERTY, deployment, recommendations)),
        batchedAttentionValues(
            configured(BATCHED_ATTENTION_VALUES_PROPERTY, deployment, recommendations)),
        fusedGroupedAttention(
            configured(FUSED_GROUPED_ATTENTION_PROPERTY, deployment, recommendations)),
        stagedQuantizedFfn(configured(STAGED_QUANTIZED_FFN_PROPERTY, deployment, recommendations)),
        stagedQuantizedLayer(
            configured(STAGED_QUANTIZED_LAYER_PROPERTY, deployment, recommendations)),
        blockMajorQ8Activations(
            configured(BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY, deployment, recommendations)),
        q8BlockMajorKernel(configured(Q8_BLOCK_MAJOR_KERNEL_PROPERTY, deployment, recommendations)),
        parallelQ8FfnPreparation(
            configured(PARALLEL_Q8_FFN_PREPARATION_PROPERTY, deployment, recommendations)),
        maxContextLength(configured(MAX_CONTEXT_LENGTH_PROPERTY, deployment, recommendations)));
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
      case "unsigned-pairwise" -> GgufQ4Kernel.UNSIGNED_PAIRWISE;
      default ->
          throw new IllegalArgumentException(
              Q4_KERNEL_PROPERTY
                  + " must be widened, short-pairwise, or unsigned-pairwise: "
                  + configured);
    };
  }

  static GgufQ6BatchedKernel q6BatchedKernel(String configured) {
    if (configured == null) {
      return GgufQ6BatchedKernel.ONE_QUERY_BLOCK;
    }
    return switch (configured.trim().toLowerCase(Locale.ROOT)) {
      case "one-query-block" -> GgufQ6BatchedKernel.ONE_QUERY_BLOCK;
      case "two-query-block" -> GgufQ6BatchedKernel.TWO_QUERY_BLOCK;
      default ->
          throw new IllegalArgumentException(
              Q6_BATCHED_KERNEL_PROPERTY
                  + " must be one-query-block or two-query-block: "
                  + configured);
    };
  }

  static boolean finalLayerPrefillPruning(String configured) {
    return booleanProperty(FINAL_LAYER_PREFILL_PRUNING_PROPERTY, configured);
  }

  static boolean finalLayerKvOnlyPrefill(String configured) {
    return booleanProperty(FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY, configured);
  }

  /**
   * Reads {@value #BATCHED_ATTENTION_SCORES_PROPERTY}.
   *
   * <p><b>Unset means enabled.</b> It used to mean disabled, which cost users a faster path for
   * nothing: measured 2026-10-08 with {@code :backend-java:defaultOffParityTest} and {@code
   * :backend-native:nativeAttentionParityTest}, enabling it generates <b>byte-identical tokens</b>
   * on Q4_0, Q8_0 and Q4_K artifacts, on both the pure-Java and the rust-ffm path. Every arm
   * asserted the planner reported the optimization ENABLED first, so these are measurements and not
   * silent no-ops.
   *
   * @param configured the raw property value, or null when unset
   * @return whether to use the faster route
   */
  static boolean batchedAttentionScores(String configured) {
    return booleanProperty(BATCHED_ATTENTION_SCORES_PROPERTY, configured);
  }

  /**
   * Reads {@value #BATCHED_ATTENTION_VALUES_PROPERTY}.
   *
   * <p><b>Unset means enabled.</b> It used to mean disabled, which cost users a faster path for
   * nothing: measured 2026-10-08 with {@code :backend-java:defaultOffParityTest} and {@code
   * :backend-native:nativeAttentionParityTest}, enabling it generates <b>byte-identical tokens</b>
   * on Q4_0, Q8_0 and Q4_K artifacts, on both the pure-Java and the rust-ffm path. Every arm
   * asserted the planner reported the optimization ENABLED first, so these are measurements and not
   * silent no-ops.
   *
   * @param configured the raw property value, or null when unset
   * @return whether to use the faster route
   */
  static boolean batchedAttentionValues(String configured) {
    return booleanProperty(BATCHED_ATTENTION_VALUES_PROPERTY, configured);
  }

  /**
   * Reads {@value #FUSED_GROUPED_ATTENTION_PROPERTY}.
   *
   * <p><b>Unset means disabled, and that default is deliberate.</b> The fused path reads each
   * cached K and V row once per group rather than once per query head, which is a real decode win,
   * and it agrees with the head-by-head loop to within 1.4e-6 relative on the attention output. But
   * greedy decoding is a discrete argmax, so a perturbation that small still flips a token whenever
   * the top two candidates sit inside it. Measured, not assumed: swapping only the attention kernel
   * under Granite on the RAG workload changed <b>2 of 9</b> generated answers from identical
   * prompts, at unchanged correctness -- a comma became a semicolon, and one clause was reworded.
   *
   * <p>Qualification records are claims about the bytes a model produced, and a published one
   * cannot be retracted. So this stays off until a configuration is adopted as a new measurement
   * epoch, with every pinned greedy oracle re-run and the tier band re-derived on it. Turning it on
   * is a choice to leave that epoch, which is why it has to be asked for by name.
   *
   * <p><b>Re-measured 2026-10-08</b> by {@code :backend-java:defaultOffParityTest}, which checked
   * every default-off setting the same way: this is the only one that moved. Enabling it generated
   * byte-identical tokens on a Q8_0 and a Q4_K artifact but <b>diverged at token 14</b> on
   * qwen2.5-coder-0.5b Q4_0. Intermittent is exactly what the argmax argument above predicts, and
   * it is why this one cannot follow the other six to enabled-by-default: the other six measured
   * identical on every artifact and every backend, so they were adopted.
   *
   * @param configured the raw property value, or null when unset
   * @return whether to fuse grouped-query attention
   */
  static boolean fusedGroupedAttention(String configured) {
    return configured != null && booleanProperty(FUSED_GROUPED_ATTENTION_PROPERTY, configured);
  }

  /**
   * Reads {@value #STAGED_QUANTIZED_FFN_PROPERTY}.
   *
   * <p><b>Unset means enabled.</b> It used to mean disabled, which cost users a faster path for
   * nothing: measured 2026-10-08 with {@code :backend-java:defaultOffParityTest} and {@code
   * :backend-native:nativeAttentionParityTest}, enabling it generates <b>byte-identical tokens</b>
   * on the Q4_0 and Q8_0 artifacts that support it; the rust-ffm path still forces it false through
   * RustGgufBatchedMatrixKernel, because the shim already owns those projections. Every arm
   * asserted the planner reported the optimization ENABLED first, so these are measurements and not
   * silent no-ops.
   *
   * @param configured the raw property value, or null when unset
   * @return whether to use the faster route
   */
  static boolean stagedQuantizedFfn(String configured) {
    return booleanProperty(STAGED_QUANTIZED_FFN_PROPERTY, configured);
  }

  /**
   * Reads {@value #STAGED_QUANTIZED_LAYER_PROPERTY}.
   *
   * <p><b>Unset means enabled.</b> It used to mean disabled, which cost users a faster path for
   * nothing: measured 2026-10-08 with {@code :backend-java:defaultOffParityTest} and {@code
   * :backend-native:nativeAttentionParityTest}, enabling it generates <b>byte-identical tokens</b>
   * on the Q4_0 and Q8_0 artifacts that support it; the rust-ffm path still forces it false through
   * RustGgufBatchedMatrixKernel, because the shim already owns those projections. Every arm
   * asserted the planner reported the optimization ENABLED first, so these are measurements and not
   * silent no-ops.
   *
   * @param configured the raw property value, or null when unset
   * @return whether to use the faster route
   */
  static boolean stagedQuantizedLayer(String configured) {
    return booleanProperty(STAGED_QUANTIZED_LAYER_PROPERTY, configured);
  }

  /**
   * Reads {@value #BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY}.
   *
   * <p><b>Unset means enabled.</b> It used to mean disabled, which cost users a faster path for
   * nothing: measured 2026-10-08 with {@code :backend-java:defaultOffParityTest} and {@code
   * :backend-native:nativeAttentionParityTest}, enabling it generates <b>byte-identical tokens</b>
   * on the Q8_0 artifact that supports it, measured with its prerequisites (a retained staged plan
   * and batched prefill) supplied, since it reports DISABLED without them. Every arm asserted the
   * planner reported the optimization ENABLED first, so these are measurements and not silent
   * no-ops.
   *
   * @param configured the raw property value, or null when unset
   * @return whether to use the faster route
   */
  static boolean blockMajorQ8Activations(String configured) {
    return booleanProperty(BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY, configured);
  }

  static GgufQ8BlockMajorKernel q8BlockMajorKernel(String configured) {
    if (configured == null) {
      return GgufQ8BlockMajorKernel.SCATTERED;
    }
    return switch (configured.trim().toLowerCase(Locale.ROOT)) {
      case "scattered" -> GgufQ8BlockMajorKernel.SCATTERED;
      case "row-accumulated" -> GgufQ8BlockMajorKernel.ROW_ACCUMULATED;
      case "float-lane-accumulated" -> GgufQ8BlockMajorKernel.FLOAT_LANE_ACCUMULATED;
      default ->
          throw new IllegalArgumentException(
              Q8_BLOCK_MAJOR_KERNEL_PROPERTY
                  + " must be scattered, row-accumulated, or float-lane-accumulated: "
                  + configured);
    };
  }

  /**
   * Reads {@value #PARALLEL_Q8_FFN_PREPARATION_PROPERTY}.
   *
   * <p><b>Unset means enabled.</b> It used to mean disabled, which cost users a faster path for
   * nothing: measured 2026-10-08 with {@code :backend-java:defaultOffParityTest} and {@code
   * :backend-native:nativeAttentionParityTest}, enabling it generates <b>byte-identical tokens</b>
   * on the Q8_0 artifact that supports it, measured with its prerequisites (the staged layer plan
   * and block-major activations) supplied, since it reports DISABLED without them. Every arm
   * asserted the planner reported the optimization ENABLED first, so these are measurements and not
   * silent no-ops.
   *
   * @param configured the raw property value, or null when unset
   * @return whether to use the faster route
   */
  static boolean parallelQ8FfnPreparation(String configured) {
    return booleanProperty(PARALLEL_Q8_FFN_PREPARATION_PROPERTY, configured);
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

  static int maxContextLength(String configured) {
    if (configured == null) {
      return MODEL_MAXIMUM_CONTEXT;
    }
    int value;
    try {
      value = Integer.parseInt(configured.trim());
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException(
          MAX_CONTEXT_LENGTH_PROPERTY + " must be a positive integer: " + configured, failure);
    }
    if (value <= 0) {
      throw new IllegalArgumentException(
          MAX_CONTEXT_LENGTH_PROPERTY + " must be a positive integer: " + configured);
    }
    return value;
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
