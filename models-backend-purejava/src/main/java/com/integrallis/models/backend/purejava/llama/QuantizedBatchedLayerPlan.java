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
package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.GgufQ8_0Batch;
import com.integrallis.vectors.core.GgufStagePlan;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/** Reusable retained Q4_0 and Q8_0 batched schedules owned by the transformer backend. */
final class QuantizedBatchedLayerPlan {

  @FunctionalInterface
  interface AttentionStage {
    void execute(
        LlamaWeights.LayerWeights layer,
        int layerIndex,
        int startPosition,
        int fromBatch,
        int toBatch);
  }

  private static final int Q8_BLOCK_SIZE = 32;

  private final int batchCapacity;
  private final int embeddingDim;
  private final int attentionOutputDim;
  private final int hiddenDim;
  private final float rmsNormEps;
  private final GgufQ4Kernel q4Kernel;
  private final float[] residual;
  private final float[] normalizedInput;
  private final float[] attentionOutput;
  private final float[] attentionProjected;
  private final float[] gate;
  private final float[] up;
  private final float[] activated;
  private final float[] projected;
  private final float[] laneScratch;
  private final AttentionStage attentionPreparation;
  private final AttentionStage attentionComputation;
  private final GgufQ8_0Batch attentionActivation;
  private final GgufQ8_0Batch inputActivation;
  private final GgufQ8_0Batch outputActivation;
  private final GgufStagePlan ffnStagePlan;
  private final GgufStagePlan layerStagePlan;

  private LlamaWeights.LayerWeights activeLayer;
  private int activeLayerIndex;
  private int activeStartPosition;
  private int activeBatchSize;

  QuantizedBatchedLayerPlan(
      int batchCapacity,
      int embeddingDim,
      int attentionOutputDim,
      int hiddenDim,
      float rmsNormEps,
      GgufQ4Kernel q4Kernel,
      float[] residual,
      float[] normalizedInput,
      float[] attentionOutput,
      float[] attentionProjected,
      float[] gate,
      float[] up,
      float[] activated,
      float[] projected,
      float[] laneScratch,
      AttentionStage attentionPreparation,
      AttentionStage attentionComputation) {
    if (batchCapacity < 2) {
      throw new IllegalArgumentException("batchCapacity must be at least two");
    }
    if (embeddingDim % Q8_BLOCK_SIZE != 0
        || attentionOutputDim % Q8_BLOCK_SIZE != 0
        || hiddenDim % Q8_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException("quantized layer dimensions must be multiples of 32");
    }
    this.batchCapacity = batchCapacity;
    this.embeddingDim = embeddingDim;
    this.attentionOutputDim = attentionOutputDim;
    this.hiddenDim = hiddenDim;
    this.rmsNormEps = rmsNormEps;
    this.q4Kernel = Objects.requireNonNull(q4Kernel, "q4Kernel");
    this.residual = Objects.requireNonNull(residual, "residual");
    this.normalizedInput = Objects.requireNonNull(normalizedInput, "normalizedInput");
    this.attentionOutput = Objects.requireNonNull(attentionOutput, "attentionOutput");
    this.attentionProjected = Objects.requireNonNull(attentionProjected, "attentionProjected");
    this.gate = Objects.requireNonNull(gate, "gate");
    this.up = Objects.requireNonNull(up, "up");
    this.activated = Objects.requireNonNull(activated, "activated");
    this.projected = Objects.requireNonNull(projected, "projected");
    this.laneScratch = Objects.requireNonNull(laneScratch, "laneScratch");
    this.attentionPreparation =
        Objects.requireNonNull(attentionPreparation, "attentionPreparation");
    this.attentionComputation =
        Objects.requireNonNull(attentionComputation, "attentionComputation");
    this.attentionActivation = GgufQ8_0Batch.allocate(batchCapacity, attentionOutputDim);
    this.inputActivation = GgufQ8_0Batch.allocate(batchCapacity, embeddingDim);
    this.outputActivation = GgufQ8_0Batch.allocate(batchCapacity, hiddenDim);
    this.ffnStagePlan =
        GgufStagePlan.of(
            GgufStagePlan.stage(hiddenDim / Q8_BLOCK_SIZE, this::projectGateUpAndActivate),
            GgufStagePlan.stage(embeddingDim, this::projectDown));
    this.layerStagePlan =
        GgufStagePlan.of(
            GgufStagePlan.stage(batchCapacity, this::prepareAttention),
            GgufStagePlan.stage(batchCapacity, this::computeAttention),
            GgufStagePlan.stage(attentionOutputDim / Q8_BLOCK_SIZE, this::quantizeAttentionOutput),
            GgufStagePlan.stage(embeddingDim, this::projectAttentionOutput),
            GgufStagePlan.stage(1, this::prepareFfnInput),
            GgufStagePlan.stage(hiddenDim / Q8_BLOCK_SIZE, this::projectGateUpAndActivate),
            GgufStagePlan.stage(embeddingDim, this::projectDown));
  }

  boolean supportsFfn(LlamaWeights.LayerWeights layer) {
    Objects.requireNonNull(layer, "layer");
    return supportsProjection(layer.ffnGateType())
        && supportsProjection(layer.ffnUpType())
        && supportsProjection(layer.ffnDownType());
  }

  boolean supportsLayer(LlamaWeights.LayerWeights layer) {
    return supportsFfn(layer) && supportsProjection(layer.woType());
  }

  void executeFfn(LlamaWeights.LayerWeights layer, int batchSize) {
    Objects.requireNonNull(layer, "layer");
    if (!supportsFfn(layer)) {
      throw new IllegalArgumentException(
          "staged quantized FFN requires Q4_0 or Q8_0 gate, up, and down weights");
    }
    validateBatchSize(batchSize);
    quantize(inputActivation, normalizedInput, batchSize, layer.ffnGateType(), layer.ffnUpType());
    execute(layer, batchSize, ffnStagePlan);
  }

  void executeLayer(
      LlamaWeights.LayerWeights layer, int layerIndex, int startPosition, int batchSize) {
    Objects.requireNonNull(layer, "layer");
    if (!supportsLayer(layer)) {
      throw new IllegalArgumentException(
          "staged quantized layer requires Q4_0 or Q8_0 output, gate, up, and down weights");
    }
    validateBatchSize(batchSize);
    if (layerIndex < 0) {
      throw new IllegalArgumentException("layerIndex must be non-negative: " + layerIndex);
    }
    if (startPosition < 0) {
      throw new IllegalArgumentException("startPosition must be non-negative: " + startPosition);
    }
    activeLayerIndex = layerIndex;
    activeStartPosition = startPosition;
    try {
      execute(layer, batchSize, layerStagePlan);
    } finally {
      activeLayerIndex = 0;
      activeStartPosition = 0;
    }
  }

  int layerStageCount() {
    return layerStagePlan.stageCount();
  }

  private void validateBatchSize(int batchSize) {
    if (batchSize < 2 || batchSize > batchCapacity) {
      throw new IllegalArgumentException(
          "batchSize must be between 2 and " + batchCapacity + ": " + batchSize);
    }
  }

  private void execute(
      LlamaWeights.LayerWeights layer, int batchSize, GgufStagePlan selectedStagePlan) {
    activeLayer = layer;
    activeBatchSize = batchSize;
    try {
      selectedStagePlan.execute();
    } finally {
      activeLayer = null;
      activeBatchSize = 0;
    }
  }

  private void projectAttentionOutput(int fromRow, int toRow) {
    batchedMatmulRows(
        activeLayer.wo(),
        activeLayer.woType(),
        activeBatchSize,
        embeddingDim,
        attentionOutputDim,
        fromRow,
        toRow,
        attentionProjected,
        attentionActivation);
  }

  private void prepareAttention(int fromBatch, int toBatch) {
    int activeToBatch = Math.min(toBatch, activeBatchSize);
    if (fromBatch < activeToBatch) {
      attentionPreparation.execute(
          activeLayer, activeLayerIndex, activeStartPosition, fromBatch, activeToBatch);
    }
  }

  private void computeAttention(int fromBatch, int toBatch) {
    int activeToBatch = Math.min(toBatch, activeBatchSize);
    if (fromBatch < activeToBatch) {
      attentionComputation.execute(
          activeLayer, activeLayerIndex, activeStartPosition, fromBatch, activeToBatch);
    }
  }

  private void quantizeAttentionOutput(int fromBlock, int toBlock) {
    quantizeBlockRange(
        attentionActivation,
        attentionOutput,
        activeBatchSize,
        fromBlock,
        toBlock,
        activeLayer.woType());
  }

  private void prepareFfnInput(int ignoredFrom, int ignoredTo) {
    int activeLength = activeBatchSize * embeddingDim;
    for (int index = 0; index < activeLength; index++) {
      residual[index] += attentionProjected[index];
    }
    float[] ffnNorm = activeLayer.ffnNorm();
    for (int batch = 0; batch < activeBatchSize; batch++) {
      int offset = batch * embeddingDim;
      TensorOps.rmsNorm(
          normalizedInput, offset, residual, offset, ffnNorm, embeddingDim, rmsNormEps);
    }
    quantize(
        inputActivation,
        normalizedInput,
        activeBatchSize,
        activeLayer.ffnGateType(),
        activeLayer.ffnUpType());
  }

  private void projectGateUpAndActivate(int fromBlock, int toBlock) {
    int fromRow = fromBlock * Q8_BLOCK_SIZE;
    int toRow = toBlock * Q8_BLOCK_SIZE;
    LlamaWeights.LayerWeights layer = activeLayer;
    batchedMatmulRows(
        layer.ffnGate(),
        layer.ffnGateType(),
        activeBatchSize,
        hiddenDim,
        embeddingDim,
        fromRow,
        toRow,
        gate,
        inputActivation);
    batchedMatmulRows(
        layer.ffnUp(),
        layer.ffnUpType(),
        activeBatchSize,
        hiddenDim,
        embeddingDim,
        fromRow,
        toRow,
        up,
        inputActivation);

    int length = toRow - fromRow;
    for (int batch = 0; batch < activeBatchSize; batch++) {
      int rowOffset = batch * hiddenDim + fromRow;
      TensorOps.swiGlu(activated, rowOffset, gate, rowOffset, up, rowOffset, length);
    }
    quantizeBlockRange(
        outputActivation,
        activated,
        activeBatchSize,
        fromBlock,
        toBlock,
        activeLayer.ffnDownType());
  }

  private void projectDown(int fromRow, int toRow) {
    batchedMatmulRows(
        activeLayer.ffnDown(),
        activeLayer.ffnDownType(),
        activeBatchSize,
        embeddingDim,
        hiddenDim,
        fromRow,
        toRow,
        projected,
        outputActivation);
  }

  private void batchedMatmulRows(
      MemorySegment weight,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols,
      int fromRow,
      int toRow,
      float[] out,
      GgufQ8_0Batch activation) {
    switch (type) {
      case Q4_0 ->
          VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
              weight,
              batchSize,
              rows,
              cols,
              fromRow,
              toRow,
              out,
              activation,
              laneScratch,
              q4Kernel);
      case Q8_0 ->
          VectorUtil.ggufQ8_0Q8_0BatchedMatmulRows(
              weight, batchSize, rows, cols, fromRow, toRow, out, activation);
      default -> throw new IllegalArgumentException("unsupported staged weight type: " + type);
    }
  }

  private void quantize(
      GgufQ8_0Batch activation,
      float[] values,
      int batchSize,
      GgufTensorType firstConsumerType,
      GgufTensorType secondConsumerType) {
    if (firstConsumerType == GgufTensorType.Q4_0 || secondConsumerType == GgufTensorType.Q4_0) {
      activation.quantizeForQ4(values, batchSize, q4Kernel);
    } else {
      activation.quantize(values, batchSize);
    }
  }

  private void quantizeBlockRange(
      GgufQ8_0Batch activation,
      float[] values,
      int batchSize,
      int fromBlock,
      int toBlock,
      GgufTensorType consumerType) {
    if (consumerType == GgufTensorType.Q4_0) {
      activation.quantizeBlockRangeForQ4(values, batchSize, fromBlock, toBlock, q4Kernel);
    } else {
      activation.quantizeBlockRange(values, batchSize, fromBlock, toBlock);
    }
  }

  private static boolean supportsProjection(GgufTensorType type) {
    return type == GgufTensorType.Q4_0 || type == GgufTensorType.Q8_0;
  }
}
