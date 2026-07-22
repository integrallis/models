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
      GgufTensorType down) {

    public LayerTopology {
      Objects.requireNonNull(query, "query");
      Objects.requireNonNull(key, "key");
      Objects.requireNonNull(value, "value");
      Objects.requireNonNull(attentionOutput, "attentionOutput");
      Objects.requireNonNull(gate, "gate");
      Objects.requireNonNull(up, "up");
      Objects.requireNonNull(down, "down");
    }

    boolean supportsBatchedPrefill() {
      return TensorOps.supportsBatchedMatmul(query)
          && TensorOps.supportsBatchedMatmul(key)
          && TensorOps.supportsBatchedMatmul(value)
          && TensorOps.supportsBatchedMatmul(attentionOutput)
          && TensorOps.supportsBatchedMatmul(gate)
          && TensorOps.supportsBatchedMatmul(up)
          && TensorOps.supportsBatchedMatmul(down);
    }

    boolean groupsGateUp() {
      return gate == up && TensorOps.supportsGroupedMatmul(gate);
    }

    boolean groupsMixedKQkv() {
      return query == GgufTensorType.Q4_K
          && key == GgufTensorType.Q4_K
          && value == GgufTensorType.Q6_K;
    }

    boolean supportsStagedQ4Ffn() {
      return gate == GgufTensorType.Q4_0
          && up == GgufTensorType.Q4_0
          && down == GgufTensorType.Q4_0;
    }

    boolean supportsStagedQ4Layer() {
      return attentionOutput == GgufTensorType.Q4_0 && supportsStagedQ4Ffn();
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

  boolean supportsBatchedPrefill() {
    return layers.stream().allMatch(LayerTopology::supportsBatchedPrefill);
  }

  boolean hasGroupedProjection() {
    return layers.stream()
        .anyMatch(layer -> layer.groupsGateUp() || !"independent".equals(layer.qkvMode()));
  }

  int mixedKProjectionLayers() {
    return Math.toIntExact(layers.stream().filter(LayerTopology::groupsMixedKQkv).count());
  }

  int stagedQ4FfnLayers() {
    if (!threadShareableProjectionWeights) {
      return 0;
    }
    return Math.toIntExact(layers.stream().filter(LayerTopology::supportsStagedQ4Ffn).count());
  }

  int stagedQ4LayerLayers() {
    if (!threadShareableProjectionWeights) {
      return 0;
    }
    return Math.toIntExact(layers.stream().filter(LayerTopology::supportsStagedQ4Layer).count());
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
    LayerTopology finalLayer = layers.getLast();
    GgufTensorType type = finalLayer.gate();
    return type == finalLayer.up()
        && type == finalLayer.down()
        && (type == GgufTensorType.Q4_0 || type == GgufTensorType.Q8_0);
  }

  boolean supportsFinalLayerKvOnlyPrefill() {
    LayerTopology finalLayer = layers.getLast();
    GgufTensorType type = finalLayer.query();
    return supportsFinalLayerPrefillPruning()
        && type == finalLayer.gate()
        && type == finalLayer.key()
        && type == finalLayer.value()
        && type == finalLayer.attentionOutput()
        && (type == GgufTensorType.Q4_0 || type == GgufTensorType.Q8_0);
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

  String gateUpMode() {
    long grouped = layers.stream().filter(LayerTopology::groupsGateUp).count();
    if (grouped == layers.size()) {
      return "grouped";
    }
    return grouped > 0 ? "partial" : "independent";
  }
}
