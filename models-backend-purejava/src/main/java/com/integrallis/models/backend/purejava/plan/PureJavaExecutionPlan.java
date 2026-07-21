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
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.util.Objects;

/** Immutable execution choices selected for one loaded pure-Java model. */
public record PureJavaExecutionPlan(
    RuntimeFingerprint runtime,
    ModelTopology topology,
    boolean groupedProjections,
    boolean mixedKProjections,
    GgufQ4Kernel q4Kernel,
    int prefillBatchSize,
    boolean finalLayerPrefillPruning,
    boolean finalLayerKvOnlyPrefill,
    boolean batchedAttentionScores,
    boolean batchedAttentionValues,
    BackendDiagnostics diagnostics) {

  public PureJavaExecutionPlan {
    runtime = Objects.requireNonNull(runtime, "runtime");
    topology = Objects.requireNonNull(topology, "topology");
    q4Kernel = Objects.requireNonNull(q4Kernel, "q4Kernel");
    if (prefillBatchSize < 1) {
      throw new IllegalArgumentException("prefillBatchSize must be >= 1");
    }
    if (groupedProjections && !topology.hasGroupedProjection()) {
      throw new IllegalArgumentException(
          "grouped projections contradict the execution plan topology");
    }
    if (mixedKProjections && (!groupedProjections || topology.mixedKProjectionLayers() == 0)) {
      throw new IllegalArgumentException(
          "mixed K-quant projections contradict the execution plan topology");
    }
    if (q4Kernel == GgufQ4Kernel.SHORT_PAIRWISE
        && (!runtime.q4ShortPairwiseSupported() || !topology.uses(GgufTensorType.Q4_0))) {
      throw new IllegalArgumentException(
          "short-pairwise Q4 kernel contradicts the execution plan runtime or topology");
    }
    if (prefillBatchSize > 1 && !topology.supportsBatchedPrefill()) {
      throw new IllegalArgumentException("batched prefill contradicts the execution plan topology");
    }
    if (finalLayerPrefillPruning && !topology.supportsFinalLayerPrefillPruning()) {
      throw new IllegalArgumentException(
          "final-layer prefill pruning contradicts the execution plan topology");
    }
    if (finalLayerKvOnlyPrefill
        && (!finalLayerPrefillPruning || !topology.supportsFinalLayerKvOnlyPrefill())) {
      throw new IllegalArgumentException(
          "final-layer K/V-only prefill contradicts the execution plan topology");
    }
    diagnostics = Objects.requireNonNull(diagnostics, "diagnostics");
  }
}
