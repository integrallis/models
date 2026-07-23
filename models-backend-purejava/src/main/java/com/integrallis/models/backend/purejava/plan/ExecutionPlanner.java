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
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Selects a deterministic plan from model topology, runtime identity, and explicit overrides. */
public final class ExecutionPlanner {

  public static final String PLAN_VERSION = "pure-java-v12";

  private ExecutionPlanner() {}

  public static PureJavaExecutionPlan plan(
      RuntimeFingerprint runtime, ModelTopology topology, PureJavaPlanConfiguration configuration) {
    Objects.requireNonNull(runtime, "runtime");
    Objects.requireNonNull(topology, "topology");
    Objects.requireNonNull(configuration, "configuration");

    List<OptimizationDecision> decisions = new ArrayList<>();
    boolean grouped = groupedProjections(topology, configuration, decisions);
    boolean mixedK = mixedKProjections(topology, configuration, grouped, decisions);
    GgufQ4Kernel q4Kernel = q4Kernel(runtime, topology, configuration, decisions);
    int prefillBatchSize = batchedPrefill(topology, configuration, decisions);
    boolean finalLayerPrefillPruning = finalLayerPrefillPruning(topology, configuration, decisions);
    boolean finalLayerKvOnlyPrefill =
        finalLayerKvOnlyPrefill(topology, configuration, finalLayerPrefillPruning, decisions);
    boolean batchedAttentionScores = batchedAttentionScores(configuration, decisions);
    boolean batchedAttentionValues = batchedAttentionValues(configuration, decisions);
    boolean stagedQuantizedFfn =
        stagedQuantizedFfn(runtime, topology, configuration, prefillBatchSize, decisions);
    boolean stagedQuantizedLayer =
        stagedQuantizedLayer(runtime, topology, configuration, prefillBatchSize, decisions);
    boolean blockMajorQ8Activations =
        blockMajorQ8Activations(
            topology,
            configuration,
            prefillBatchSize,
            stagedQuantizedFfn,
            stagedQuantizedLayer,
            decisions);
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
        q4Kernel,
        prefillBatchSize,
        finalLayerPrefillPruning,
        finalLayerKvOnlyPrefill,
        batchedAttentionScores,
        batchedAttentionValues,
        stagedQuantizedFfn,
        stagedQuantizedLayer,
        blockMajorQ8Activations,
        diagnostics);
  }

  private static boolean blockMajorQ8Activations(
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      int prefillBatchSize,
      boolean stagedQuantizedFfn,
      boolean stagedQuantizedLayer,
      List<OptimizationDecision> decisions) {
    boolean requested = configuration.blockMajorQ8Activations();
    boolean staged = stagedQuantizedFfn || stagedQuantizedLayer;
    boolean q8Eligible = topology.hasStagedQ8Projection(stagedQuantizedLayer);
    boolean enabled = requested && staged && q8Eligible && prefillBatchSize > 1;
    OptimizationStatus status;
    String reason;
    if (enabled) {
      status = OptimizationStatus.ENABLED;
      reason = "staged Q8 projections traverse each activation block sequentially";
    } else if (!q8Eligible) {
      status = OptimizationStatus.UNSUPPORTED;
      reason = "loaded staged projection topology has no Q8_0 consumer";
    } else if (!requested) {
      status = OptimizationStatus.DISABLED;
      reason = "disabled by models.purejava.blockMajorQ8Activations";
    } else if (!staged) {
      status = OptimizationStatus.DISABLED;
      reason = "block-major Q8 activations require a retained staged plan";
    } else {
      status = OptimizationStatus.DISABLED;
      reason = "block-major Q8 activations require batched prefill";
    }
    decisions.add(
        new OptimizationDecision(
            "block-major-q8-activations",
            status,
            reason,
            Map.of(
                "layout",
                "block-major-bytes",
                "prefill-batch-size",
                Integer.toString(prefillBatchSize))));
    return enabled;
  }

  private static boolean stagedQuantizedLayer(
      RuntimeFingerprint runtime,
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      int prefillBatchSize,
      List<OptimizationDecision> decisions) {
    int eligibleLayers = topology.stagedQuantizedLayerLayers();
    boolean requested = configuration.stagedQuantizedLayer();
    boolean enabled =
        requested
            && eligibleLayers > 0
            && prefillBatchSize > 1
            && runtime.processors() > 1
            && runtime.ggufParallel()
            && "persistent".equals(runtime.ggufExecutor());
    OptimizationStatus status;
    String reason;
    if (enabled) {
      status = OptimizationStatus.ENABLED;
      reason =
          "eligible Q4_0/Q8_0 attention and FFN work shares one retained seven-stage publication";
    } else if (eligibleLayers == 0) {
      status = OptimizationStatus.UNSUPPORTED;
      reason = "loaded tensor topology has no supported Q4_0/Q8_0 output-and-FFN layer";
    } else if (!requested) {
      status = OptimizationStatus.DISABLED;
      reason = "disabled by models.purejava.stagedQuantizedLayer";
    } else if (prefillBatchSize == 1) {
      status = OptimizationStatus.DISABLED;
      reason = "staged quantized layer requires batched prefill";
    } else if (!runtime.ggufParallel()) {
      status = OptimizationStatus.DISABLED;
      reason = "staged quantized layer requires parallel GGUF execution";
    } else if (!"persistent".equals(runtime.ggufExecutor())) {
      status = OptimizationStatus.DISABLED;
      reason = "staged quantized layer requires the persistent GGUF executor";
    } else {
      status = OptimizationStatus.DISABLED;
      reason = "staged quantized layer requires more than one processor";
    }
    decisions.add(
        new OptimizationDecision(
            "staged-quantized-layer",
            status,
            reason,
            Map.of(
                "eligible-layers",
                Integer.toString(eligibleLayers),
                "executor",
                runtime.ggufExecutor(),
                "parallel",
                Boolean.toString(runtime.ggufParallel()),
                "stages",
                "7")));
    return enabled;
  }

  private static boolean stagedQuantizedFfn(
      RuntimeFingerprint runtime,
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      int prefillBatchSize,
      List<OptimizationDecision> decisions) {
    int eligibleLayers = topology.stagedQuantizedFfnLayers();
    boolean requested = configuration.stagedQuantizedFfn();
    boolean enabled =
        requested
            && eligibleLayers > 0
            && prefillBatchSize > 1
            && runtime.processors() > 1
            && runtime.ggufParallel()
            && "persistent".equals(runtime.ggufExecutor());
    OptimizationStatus status;
    String reason;
    if (enabled) {
      status = OptimizationStatus.ENABLED;
      reason = "eligible Q4_0/Q8_0 FFN layers share one retained two-stage row publication";
    } else if (eligibleLayers == 0) {
      status = OptimizationStatus.UNSUPPORTED;
      reason = "loaded tensor topology has no supported Q4_0/Q8_0 FFN layer";
    } else if (!requested) {
      status = OptimizationStatus.DISABLED;
      reason = "disabled by models.purejava.stagedQuantizedFfn";
    } else if (prefillBatchSize == 1) {
      status = OptimizationStatus.DISABLED;
      reason = "staged quantized FFN requires batched prefill";
    } else if (!runtime.ggufParallel()) {
      status = OptimizationStatus.DISABLED;
      reason = "staged quantized FFN requires parallel GGUF execution";
    } else if (!"persistent".equals(runtime.ggufExecutor())) {
      status = OptimizationStatus.DISABLED;
      reason = "staged quantized FFN requires the persistent GGUF executor";
    } else {
      status = OptimizationStatus.DISABLED;
      reason = "staged quantized FFN requires more than one processor";
    }
    decisions.add(
        new OptimizationDecision(
            "staged-quantized-ffn",
            status,
            reason,
            Map.of(
                "eligible-layers",
                Integer.toString(eligibleLayers),
                "executor",
                runtime.ggufExecutor(),
                "parallel",
                Boolean.toString(runtime.ggufParallel()),
                "stages",
                "2")));
    return enabled;
  }

  private static boolean batchedAttentionScores(
      PureJavaPlanConfiguration configuration, List<OptimizationDecision> decisions) {
    boolean enabled = configuration.batchedAttentionScores();
    decisions.add(
        new OptimizationDecision(
            "batched-attention-scores",
            enabled ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED,
            enabled
                ? "the model profile selected exact two-row key scoring"
                : "disabled by models.purejava.batchedAttentionScores",
            Map.of("rows-per-group", enabled ? "2" : "1")));
    return enabled;
  }

  private static boolean batchedAttentionValues(
      PureJavaPlanConfiguration configuration, List<OptimizationDecision> decisions) {
    boolean enabled = configuration.batchedAttentionValues();
    decisions.add(
        new OptimizationDecision(
            "batched-attention-values",
            enabled ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED,
            enabled
                ? "the model profile selected four-row value accumulation"
                : "disabled by models.purejava.batchedAttentionValues",
            Map.of("rows-per-group", enabled ? "4" : "1")));
    return enabled;
  }

  private static GgufQ4Kernel q4Kernel(
      RuntimeFingerprint runtime,
      ModelTopology topology,
      PureJavaPlanConfiguration configuration,
      List<OptimizationDecision> decisions) {
    GgufQ4Kernel requested = configuration.q4Kernel();
    Map<String, String> settings =
        Map.of(
            "requested",
            kernelName(requested),
            "active-vector-bits",
            Integer.toString(runtime.vectorBits()),
            "short-supported",
            Boolean.toString(runtime.q4ShortPairwiseSupported()),
            "unsigned-supported",
            Boolean.toString(runtime.q4UnsignedPairwiseSupported()),
            "compiler",
            runtime.compiler(),
            "java-feature",
            Integer.toString(Runtime.Version.parse(runtime.javaVersion()).feature()),
            "os",
            runtime.osName(),
            "architecture",
            runtime.architecture());
    if (!topology.uses(GgufTensorType.Q4_0)) {
      decisions.add(
          new OptimizationDecision(
              "q4-kernel",
              OptimizationStatus.UNSUPPORTED,
              "the loaded model has no Q4_0 projection tensors",
              settings));
      return GgufQ4Kernel.WIDENED;
    }
    if (requested == GgufQ4Kernel.WIDENED) {
      decisions.add(
          new OptimizationDecision(
              "q4-kernel",
              OptimizationStatus.DISABLED,
              "the stable widened Q4 kernel was selected",
              settings));
      return requested;
    }
    if (requested == GgufQ4Kernel.SHORT_PAIRWISE && !runtime.q4ShortPairwiseSupported()) {
      decisions.add(
          new OptimizationDecision(
              "q4-kernel",
              OptimizationStatus.UNSUPPORTED,
              "the active Vector API shape cannot execute the measured pairwise kernel",
              settings));
      return GgufQ4Kernel.WIDENED;
    }
    if (requested == GgufQ4Kernel.UNSIGNED_PAIRWISE && !runtime.q4UnsignedPairwiseSupported()) {
      decisions.add(
          new OptimizationDecision(
              "q4-kernel",
              OptimizationStatus.UNSUPPORTED,
              "the active Vector API shape cannot execute the unsigned pairwise kernel",
              settings));
      return GgufQ4Kernel.WIDENED;
    }
    decisions.add(
        new OptimizationDecision(
            "q4-kernel",
            OptimizationStatus.ENABLED,
            "the model-scoped execution plan selected the measured pairwise Q4 kernel",
            settings));
    return requested;
  }

  private static String kernelName(GgufQ4Kernel kernel) {
    return kernel.name().toLowerCase(java.util.Locale.ROOT).replace('_', '-');
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
