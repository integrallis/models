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
import com.integrallis.vectors.core.GgufQ8BlockMajorKernel;
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
    boolean stagedQuantizedFfn,
    boolean stagedQuantizedLayer,
    boolean blockMajorQ8Activations,
    GgufQ8BlockMajorKernel q8BlockMajorKernel,
    boolean parallelQ8FfnPreparation,
    BackendDiagnostics diagnostics) {

  public PureJavaExecutionPlan {
    runtime = Objects.requireNonNull(runtime, "runtime");
    topology = Objects.requireNonNull(topology, "topology");
    q4Kernel = Objects.requireNonNull(q4Kernel, "q4Kernel");
    q8BlockMajorKernel = Objects.requireNonNull(q8BlockMajorKernel, "q8BlockMajorKernel");
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
    if (q4Kernel != GgufQ4Kernel.WIDENED && !topology.uses(GgufTensorType.Q4_0)) {
      throw new IllegalArgumentException(
          "optimized Q4 kernel contradicts the execution plan topology");
    }
    if (q4Kernel == GgufQ4Kernel.SHORT_PAIRWISE && !runtime.q4ShortPairwiseSupported()) {
      throw new IllegalArgumentException(
          "short-pairwise Q4 kernel contradicts the execution plan runtime");
    }
    if (q4Kernel == GgufQ4Kernel.UNSIGNED_PAIRWISE && !runtime.q4UnsignedPairwiseSupported()) {
      throw new IllegalArgumentException(
          "unsigned-pairwise Q4 kernel contradicts the execution plan runtime");
    }
    if (prefillBatchSize > 1 && !topology.supportsBatchedPrefill()) {
      throw new IllegalArgumentException("batched prefill contradicts the execution plan topology");
    }
    if (stagedQuantizedFfn
        && (prefillBatchSize < 2
            || runtime.processors() < 2
            || !"persistent".equals(runtime.ggufExecutor())
            || topology.stagedQuantizedFfnLayers() == 0)) {
      throw new IllegalArgumentException(
          "staged quantized FFN contradicts the execution plan topology or runtime");
    }
    if (stagedQuantizedLayer
        && (prefillBatchSize < 2
            || runtime.processors() < 2
            || !runtime.ggufParallel()
            || !"persistent".equals(runtime.ggufExecutor())
            || topology.stagedQuantizedLayerLayers() == 0)) {
      throw new IllegalArgumentException(
          "staged quantized layer contradicts the execution plan topology or runtime");
    }
    if (blockMajorQ8Activations
        && (prefillBatchSize < 2
            || (!stagedQuantizedFfn && !stagedQuantizedLayer)
            || !topology.hasStagedQ8Projection(stagedQuantizedLayer))) {
      throw new IllegalArgumentException(
          "block-major Q8 activations contradict the execution plan topology");
    }
    if (q8BlockMajorKernel != GgufQ8BlockMajorKernel.SCATTERED
        && (!blockMajorQ8Activations
            || !"graal-jvmci".equals(runtime.compiler())
            || runtime.vectorBits() < 256
            || (q8BlockMajorKernel == GgufQ8BlockMajorKernel.FLOAT_LANE_ACCUMULATED
                && runtime.vectorBits() != 256))) {
      throw new IllegalArgumentException(
          "specialized Q8 kernel contradicts the execution plan topology or runtime");
    }
    if (parallelQ8FfnPreparation
        && (!stagedQuantizedLayer
            || !blockMajorQ8Activations
            || !topology.supportsParallelQ8FfnPreparation())) {
      throw new IllegalArgumentException(
          "parallel Q8 FFN preparation contradicts the execution plan topology");
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
