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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Selects a deterministic plan from model topology, runtime identity, and explicit overrides. */
public final class ExecutionPlanner {

  public static final String PLAN_VERSION = "pure-java-v4";

  private ExecutionPlanner() {}

  public static PureJavaExecutionPlan plan(
      RuntimeFingerprint runtime, ModelTopology topology, PureJavaPlanConfiguration configuration) {
    Objects.requireNonNull(runtime, "runtime");
    Objects.requireNonNull(topology, "topology");
    Objects.requireNonNull(configuration, "configuration");

    List<OptimizationDecision> decisions = new ArrayList<>();
    boolean grouped = groupedProjections(topology, configuration, decisions);
    boolean mixedK = mixedKProjections(topology, configuration, grouped, decisions);
    int prefillBatchSize = batchedPrefill(topology, configuration, decisions);
    boolean finalLayerPrefillPruning = finalLayerPrefillPruning(topology, configuration, decisions);
    boolean finalLayerKvOnlyPrefill =
        finalLayerKvOnlyPrefill(topology, configuration, finalLayerPrefillPruning, decisions);
    decisions.add(
        new OptimizationDecision(
            "mapped-model-weights",
            OptimizationStatus.ENABLED,
            "GGUF tensors remain mapped for model-scale streaming locality",
            Map.of("storage", "memory-segment")));
    decisions.add(vectorFma(runtime));
    decisions.add(persistentExecutor(runtime));

    Map<String, String> environment = new LinkedHashMap<>(runtime.asEnvironment());
    environment.put("model-architecture", topology.architecture());
    environment.put("projection-layers", Integer.toString(topology.layers().size()));
    BackendDiagnostics diagnostics =
        new BackendDiagnostics("pure-java", PLAN_VERSION, environment, decisions);
    return new PureJavaExecutionPlan(
        runtime,
        topology,
        grouped,
        mixedK,
        prefillBatchSize,
        finalLayerPrefillPruning,
        finalLayerKvOnlyPrefill,
        diagnostics);
  }

  private static boolean finalLayerKvOnlyPrefill(
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      boolean finalLayerPrefillPruning,
      List<OptimizationDecision> decisions) {
    Map<String, String> settings =
        Map.of("unused-rows", "full-attention", "formats", topology.finalLayerAttentionFormats());
    if (!topology.supportsFinalLayerKvOnlyPrefill()) {
      decisions.add(
          new OptimizationDecision(
              "final-layer-kv-only-prefill",
              OptimizationStatus.UNSUPPORTED,
              "the final-layer attention layout has no controlled exactness and performance gate",
              settings));
      return false;
    }
    if (!configuration.finalLayerKvOnlyPrefill()) {
      decisions.add(
          new OptimizationDecision(
              "final-layer-kv-only-prefill",
              OptimizationStatus.DISABLED,
              "disabled by models.purejava.finalLayerKvOnlyPrefill",
              settings));
      return false;
    }
    if (!finalLayerPrefillPruning) {
      decisions.add(
          new OptimizationDecision(
              "final-layer-kv-only-prefill",
              OptimizationStatus.DISABLED,
              "final-layer prefill pruning is disabled",
              settings));
      return false;
    }
    decisions.add(
        new OptimizationDecision(
            "final-layer-kv-only-prefill",
            OptimizationStatus.ENABLED,
            "unused final-layer prompt rows compute only the persistent K/V state",
            Map.of("unused-rows", "kv-only", "formats", topology.finalLayerAttentionFormats())));
    return true;
  }

  private static boolean finalLayerPrefillPruning(
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      List<OptimizationDecision> decisions) {
    if (!configuration.finalLayerPrefillPruning()) {
      decisions.add(
          new OptimizationDecision(
              "final-layer-prefill-pruning",
              OptimizationStatus.DISABLED,
              "disabled by models.purejava.finalLayerPrefillPruning",
              Map.of("output-rows", "all")));
      return false;
    }
    if (!topology.supportsFinalLayerPrefillPruning()) {
      decisions.add(
          new OptimizationDecision(
              "final-layer-prefill-pruning",
              OptimizationStatus.UNSUPPORTED,
              "the final-layer FFN layout has no controlled exactness and performance gate",
              Map.of("output-rows", "all", "formats", topology.finalLayerFfnFormats())));
      return false;
    }
    decisions.add(
        new OptimizationDecision(
            "final-layer-prefill-pruning",
            OptimizationStatus.ENABLED,
            "the validated final FFN runs only for requested prompt output rows",
            Map.of("output-rows", "requested", "formats", topology.finalLayerFfnFormats())));
    return true;
  }

  private static boolean groupedProjections(
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      List<OptimizationDecision> decisions) {
    Map<String, String> settings =
        Map.of("qkv", topology.qkvMode(), "gate-up", topology.gateUpMode());
    if (!topology.hasGroupedProjection()) {
      decisions.add(
          new OptimizationDecision(
              "grouped-projections",
              OptimizationStatus.UNSUPPORTED,
              "loaded projection tensor types have no retained grouped kernel",
              settings));
      return false;
    }
    if (!configuration.groupedProjections()) {
      decisions.add(
          new OptimizationDecision(
              "grouped-projections",
              OptimizationStatus.DISABLED,
              "disabled by models.purejava.groupedProjections",
              settings));
      return false;
    }
    decisions.add(
        new OptimizationDecision(
            "grouped-projections",
            OptimizationStatus.ENABLED,
            "loaded tensor topology has a retained grouped projection kernel",
            settings));
    return true;
  }

  private static int batchedPrefill(
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      List<OptimizationDecision> decisions) {
    if (!topology.supportsBatchedPrefill()) {
      decisions.add(
          new OptimizationDecision(
              "batched-prefill",
              OptimizationStatus.UNSUPPORTED,
              "at least one loaded projection tensor lacks a retained batched kernel",
              Map.of("batch-size", "1")));
      return 1;
    }
    if (configuration.prefillBatchSize() == 1) {
      decisions.add(
          new OptimizationDecision(
              "batched-prefill",
              OptimizationStatus.DISABLED,
              "disabled by models.purejava.prefillBatchSize",
              Map.of("batch-size", "1")));
      return 1;
    }
    decisions.add(
        new OptimizationDecision(
            "batched-prefill",
            OptimizationStatus.ENABLED,
            "all loaded projection tensors have retained batched kernels",
            Map.of("batch-size", Integer.toString(configuration.prefillBatchSize()))));
    return configuration.prefillBatchSize();
  }

  private static boolean mixedKProjections(
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      boolean groupedProjections,
      List<OptimizationDecision> decisions) {
    int eligibleLayers = topology.mixedKProjectionLayers();
    Map<String, String> settings =
        Map.of("eligible-layers", Integer.toString(eligibleLayers), "formats", "Q4_K,Q4_K,Q6_K");
    if (eligibleLayers == 0) {
      decisions.add(
          new OptimizationDecision(
              "mixed-k-projections",
              OptimizationStatus.UNSUPPORTED,
              "loaded tensor topology has no Q4_K/Q4_K/Q6_K projection group",
              settings));
      return false;
    }
    if (!groupedProjections || !configuration.mixedKProjections()) {
      decisions.add(
          new OptimizationDecision(
              "mixed-k-projections",
              OptimizationStatus.DISABLED,
              !groupedProjections
                  ? "grouped projections are disabled"
                  : "disabled by models.purejava.mixedKProjections",
              settings));
      return false;
    }
    decisions.add(
        new OptimizationDecision(
            "mixed-k-projections",
            OptimizationStatus.ENABLED,
            "eligible projections share one Q8_K quantization and row dispatch",
            settings));
    return true;
  }

  private static OptimizationDecision vectorFma(RuntimeFingerprint runtime) {
    OptimizationStatus status =
        runtime.fastVectorFma() ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED;
    String reason =
        runtime.fastVectorFma()
            ? "vectors-core selected deterministic vector FMA for this platform"
            : "vectors-core selected multiply/add for this platform";
    return new OptimizationDecision(
        "vector-fma",
        status,
        reason,
        Map.of("fast-scalar-fma", Boolean.toString(runtime.fastScalarFma())));
  }

  private static OptimizationDecision persistentExecutor(RuntimeFingerprint runtime) {
    boolean enabled = "persistent".equals(runtime.ggufExecutor());
    return new OptimizationDecision(
        "persistent-row-executor",
        enabled ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED,
        enabled
            ? "vectors-core selected reusable GGUF row workers"
            : "vectors-core selected a different GGUF executor",
        Map.of(
            "executor",
            runtime.ggufExecutor(),
            "threads",
            Integer.toString(runtime.ggufThreads()),
            "chunks-per-thread",
            Integer.toString(runtime.ggufChunksPerThread())));
  }
}
