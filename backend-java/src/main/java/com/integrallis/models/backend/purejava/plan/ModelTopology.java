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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Tensor topology relevant to deterministic execution planning. */
public record ModelTopology(
    String architecture,
    int queryRows,
    int keyRows,
    int valueRows,
    List<LayerTopology> layers,
    boolean threadShareableProjectionWeights) {

  private static final Thread ACCESS_PROBE = Thread.ofPlatform().unstarted(() -> {});

  /** Projection tensor types for one transformer layer. */
  public record LayerTopology(
      GgufTensorType query,
      GgufTensorType key,
      GgufTensorType value,
      GgufTensorType attentionOutput,
      GgufTensorType gate,
      GgufTensorType up,
      GgufTensorType down,
      List<GgufTensorType> auxiliaryProjections) {

    public LayerTopology(
        GgufTensorType query,
        GgufTensorType key,
        GgufTensorType value,
        GgufTensorType attentionOutput,
        GgufTensorType gate,
        GgufTensorType up,
        GgufTensorType down) {
      this(query, key, value, attentionOutput, gate, up, down, List.of());
    }

    public LayerTopology {
      Objects.requireNonNull(query, "query");
      Objects.requireNonNull(key, "key");
      Objects.requireNonNull(value, "value");
      Objects.requireNonNull(attentionOutput, "attentionOutput");
      Objects.requireNonNull(gate, "gate");
      Objects.requireNonNull(up, "up");
      Objects.requireNonNull(down, "down");
      auxiliaryProjections = List.copyOf(Objects.requireNonNull(auxiliaryProjections));
    }

    boolean supportsBatchedPrefill() {
      return TensorOps.supportsBatchedMatmul(query)
          && TensorOps.supportsBatchedMatmul(key)
          && TensorOps.supportsBatchedMatmul(value)
          && TensorOps.supportsBatchedMatmul(attentionOutput)
          && TensorOps.supportsBatchedMatmul(gate)
          && TensorOps.supportsBatchedMatmul(up)
          && TensorOps.supportsBatchedMatmul(down)
          && auxiliaryProjections.stream().allMatch(TensorOps::supportsBatchedMatmul);
    }

    boolean groupsGateUp() {
      return gate == up && TensorOps.supportsGroupedMatmul(gate);
    }

    boolean groupsGateUp(GgufBatchedMatrixKernel kernel) {
      return groupsGateUp() || kernel.supportsDual(gate, up);
    }

    boolean groupsMixedKQkv() {
      return query == GgufTensorType.Q4_K
          && key == GgufTensorType.Q4_K
          && value == GgufTensorType.Q6_K;
    }

    boolean supportsStagedQuantizedFfn() {
      return supportsStagedProjection(gate)
          && supportsStagedProjection(up)
          && supportsStagedProjection(down);
    }

    boolean supportsStagedQuantizedLayer() {
      return supportsStagedProjection(attentionOutput) && supportsStagedQuantizedFfn();
    }

    boolean supportsParallelQ8FfnPreparation() {
      return supportsStagedQuantizedLayer()
          && gate == GgufTensorType.Q8_0
          && up == GgufTensorType.Q8_0;
    }

    private static boolean supportsStagedProjection(GgufTensorType type) {
      return type == GgufTensorType.Q4_0 || type == GgufTensorType.Q8_0;
    }

    String qkvMode() {
      if (TensorOps.supportsGroupedTripleMatmul(query, key, value)) {
        return "grouped";
      }
      if ((query == key && TensorOps.supportsGroupedTripleMatmul(query))
          || (query == value && TensorOps.supportsGroupedTripleMatmul(query))
          || (key == value && TensorOps.supportsGroupedTripleMatmul(key))) {
        return "partial";
      }
      return "independent";
    }

    String qkvMode(GgufBatchedMatrixKernel kernel) {
      return kernel.supportsTriple(query, key, value) ? "grouped" : qkvMode();
    }
  }

  public ModelTopology {
    if (architecture == null || architecture.isBlank()) {
      throw new IllegalArgumentException("architecture must not be blank");
    }
    architecture = architecture.trim().toLowerCase(java.util.Locale.ROOT);
    if (queryRows <= 0 || keyRows <= 0 || valueRows <= 0) {
      throw new IllegalArgumentException("projection row counts must be > 0");
    }
    layers = List.copyOf(Objects.requireNonNull(layers, "layers"));
    if (layers.isEmpty()) {
      throw new IllegalArgumentException("layers must not be empty");
    }
  }

  /** Builds planning topology from the tensors actually loaded from GGUF. */
  public static ModelTopology from(String architecture, LlamaConfig config, LlamaWeights weights) {
    List<LayerTopology> layers = new ArrayList<>(config.numLayers());
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights value = weights.layer(layer);
      layers.add(
          new LayerTopology(
              value.wqType(),
              value.wkType(),
              value.wvType(),
              value.woType(),
              value.ffnGateType(),
              value.ffnUpType(),
              value.ffnDownType()));
    }
    return new ModelTopology(
        architecture,
        config.queryDim(),
        config.keyDim(),
        config.valueDim(),
        layers,
        threadShareableProjectionWeights(config, weights));
  }

  /** Builds a conservative plan topology for a mapped non-GGUF architecture. */
  public static ModelTopology mappedArchitecture(
      String architecture, int queryRows, int keyRows, int valueRows, int layerCount) {
    if (layerCount <= 0) {
      throw new IllegalArgumentException("layerCount must be > 0");
    }
    LayerTopology neutral =
        new LayerTopology(
            GgufTensorType.F32,
            GgufTensorType.F32,
            GgufTensorType.F32,
            GgufTensorType.F32,
            GgufTensorType.F32,
            GgufTensorType.F32,
            GgufTensorType.F32);
    return new ModelTopology(
        architecture,
        queryRows,
        keyRows,
        valueRows,
        java.util.Collections.nCopies(layerCount, neutral),
        true);
  }

  boolean supportsBatchedPrefill() {
    return supportsLlamaProjectionRouting()
        && layers.stream().allMatch(LayerTopology::supportsBatchedPrefill);
  }

  boolean hasGroupedProjection() {
    return supportsLlamaProjectionRouting()
        && layers.stream()
            .anyMatch(layer -> layer.groupsGateUp() || !"independent".equals(layer.qkvMode()));
  }

  boolean hasGroupedProjection(GgufBatchedMatrixKernel kernel) {
    return hasGroupedProjection() || hasInjectedGroupedProjection(kernel);
  }

  /** Returns whether an injected kernel can group at least one loaded projection set. */
  public boolean hasInjectedGroupedProjection(GgufBatchedMatrixKernel kernel) {
    Objects.requireNonNull(kernel, "kernel");
    return supportsLlamaProjectionRouting()
        && layers.stream()
            .anyMatch(
                layer ->
                    kernel.supportsDual(layer.gate(), layer.up())
                        || kernel.supportsTriple(layer.query(), layer.key(), layer.value()));
  }

  int mixedKProjectionLayers() {
    if (!supportsLlamaProjectionRouting()) {
      return 0;
    }
    return Math.toIntExact(layers.stream().filter(LayerTopology::groupsMixedKQkv).count());
  }

  int stagedQuantizedFfnLayers() {
    if (!supportsStandardLlamaLayerSemantics() || !threadShareableProjectionWeights) {
      return 0;
    }
    return Math.toIntExact(
        layers.stream().filter(LayerTopology::supportsStagedQuantizedFfn).count());
  }

  int stagedQuantizedLayerLayers() {
    if (!supportsStandardLlamaLayerSemantics() || !threadShareableProjectionWeights) {
      return 0;
    }
    return Math.toIntExact(
        layers.stream().filter(LayerTopology::supportsStagedQuantizedLayer).count());
  }

  int parallelQ8FfnPreparationLayers() {
    if (!supportsStandardLlamaLayerSemantics() || !threadShareableProjectionWeights) {
      return 0;
    }
    return Math.toIntExact(
        layers.stream().filter(LayerTopology::supportsParallelQ8FfnPreparation).count());
  }

  boolean supportsParallelQ8FfnPreparation() {
    int stagedLayers = stagedQuantizedLayerLayers();
    return stagedLayers > 0 && parallelQ8FfnPreparationLayers() == stagedLayers;
  }

  boolean hasStagedQ8Projection(boolean includeAttentionOutput) {
    if (!supportsStandardLlamaLayerSemantics() || !threadShareableProjectionWeights) {
      return false;
    }
    return layers.stream()
        .anyMatch(
            layer ->
                (includeAttentionOutput
                        ? layer.supportsStagedQuantizedLayer()
                        : layer.supportsStagedQuantizedFfn())
                    && (layer.gate() == GgufTensorType.Q8_0
                        || layer.up() == GgufTensorType.Q8_0
                        || layer.down() == GgufTensorType.Q8_0
                        || (includeAttentionOutput
                            && layer.attentionOutput() == GgufTensorType.Q8_0)));
  }

  private static boolean threadShareableProjectionWeights(
      LlamaConfig config, LlamaWeights weights) {
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights value = weights.layer(layer);
      if (!threadShareable(value.wo(), value.ffnGate(), value.ffnUp(), value.ffnDown())) {
        return false;
      }
    }
    return true;
  }

  private static boolean threadShareable(MemorySegment... segments) {
    for (MemorySegment segment : segments) {
      if (!segment.isAccessibleBy(ACCESS_PROBE)) {
        return false;
      }
    }
    return true;
  }

  boolean uses(GgufTensorType type) {
    Objects.requireNonNull(type, "type");
    return layers.stream()
        .anyMatch(
            layer ->
                layer.query() == type
                    || layer.key() == type
                    || layer.value() == type
                    || layer.attentionOutput() == type
                    || layer.gate() == type
                    || layer.up() == type
                    || layer.down() == type);
  }

  boolean supportsFinalLayerPrefillPruning() {
    if (!supportsStandardLlamaLayerSemantics()) {
      return false;
    }
    LayerTopology finalLayer = layers.getLast();
    GgufTensorType type = finalLayer.gate();
    return type == finalLayer.up()
        && type == finalLayer.down()
        && supportsFinalLayerPromptPruning(type);
  }

  boolean supportsFinalLayerKvOnlyPrefill() {
    if (!supportsStandardLlamaLayerSemantics()) {
      return false;
    }
    LayerTopology finalLayer = layers.getLast();
    GgufTensorType type = finalLayer.query();
    return supportsFinalLayerPrefillPruning()
        && type == finalLayer.gate()
        && type == finalLayer.key()
        && type == finalLayer.value()
        && type == finalLayer.attentionOutput()
        && supportsFinalLayerPromptPruning(type);
  }

  private static boolean supportsFinalLayerPromptPruning(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K;
  }

  private boolean supportsStandardLlamaLayerSemantics() {
    return !"gemma3".equals(architecture)
        && !"qwen35".equals(architecture)
        && supportsLlamaProjectionRouting();
  }

  boolean supportsBatchedAttentionKernels() {
    return supportsLlamaProjectionRouting();
  }

  private boolean supportsLlamaProjectionRouting() {
    return !"gemma4".equals(architecture) && !"needle2".equals(architecture);
  }

  String finalLayerFfnFormats() {
    LayerTopology finalLayer = layers.getLast();
    return finalLayer.gate() + "," + finalLayer.up() + "," + finalLayer.down();
  }

  String finalLayerAttentionFormats() {
    LayerTopology finalLayer = layers.getLast();
    return finalLayer.query()
        + ","
        + finalLayer.key()
        + ","
        + finalLayer.value()
        + ","
        + finalLayer.attentionOutput();
  }

  String qkvMode() {
    boolean anyGrouped = layers.stream().anyMatch(layer -> !"independent".equals(layer.qkvMode()));
    boolean allGrouped = layers.stream().allMatch(layer -> "grouped".equals(layer.qkvMode()));
    if (allGrouped) {
      return "grouped";
    }
    return anyGrouped ? "partial" : "independent";
  }

  String qkvMode(GgufBatchedMatrixKernel kernel) {
    Objects.requireNonNull(kernel, "kernel");
    boolean anyGrouped =
        layers.stream().anyMatch(layer -> !"independent".equals(layer.qkvMode(kernel)));
    boolean allGrouped = layers.stream().allMatch(layer -> "grouped".equals(layer.qkvMode(kernel)));
    if (allGrouped) {
      return "grouped";
    }
    return anyGrouped ? "partial" : "independent";
  }

  String gateUpMode() {
    long grouped = layers.stream().filter(LayerTopology::groupsGateUp).count();
    if (grouped == layers.size()) {
      return "grouped";
    }
    return grouped > 0 ? "partial" : "independent";
  }

  String gateUpMode(GgufBatchedMatrixKernel kernel) {
    Objects.requireNonNull(kernel, "kernel");
    long grouped = layers.stream().filter(layer -> layer.groupsGateUp(kernel)).count();
    if (grouped == layers.size()) {
      return "grouped";
    }
    return grouped > 0 ? "partial" : "independent";
  }
}
