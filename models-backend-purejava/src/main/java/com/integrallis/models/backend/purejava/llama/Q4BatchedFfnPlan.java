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
import java.util.Objects;

/** Reusable two-stage Q4_0 batched FFN schedule owned by the transformer backend. */
final class Q4BatchedFfnPlan {

  private static final int Q8_BLOCK_SIZE = 32;

  private final int batchCapacity;
  private final int embeddingDim;
  private final int hiddenDim;
  private final GgufQ4Kernel q4Kernel;
  private final float[] gate;
  private final float[] up;
  private final float[] activated;
  private final float[] projected;
  private final float[] laneScratch;
  private final GgufQ8_0Batch inputActivation;
  private final GgufQ8_0Batch outputActivation;
  private final GgufStagePlan stagePlan;

  private LlamaWeights.LayerWeights activeLayer;
  private int activeBatchSize;

  Q4BatchedFfnPlan(
      int batchCapacity,
      int embeddingDim,
      int hiddenDim,
      GgufQ4Kernel q4Kernel,
      float[] gate,
      float[] up,
      float[] activated,
      float[] projected,
      float[] laneScratch) {
    if (batchCapacity < 2) {
      throw new IllegalArgumentException("batchCapacity must be at least two");
    }
    if (embeddingDim % Q8_BLOCK_SIZE != 0 || hiddenDim % Q8_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException("Q4 FFN dimensions must be multiples of 32");
    }
    this.batchCapacity = batchCapacity;
    this.embeddingDim = embeddingDim;
    this.hiddenDim = hiddenDim;
    this.q4Kernel = Objects.requireNonNull(q4Kernel, "q4Kernel");
    this.gate = Objects.requireNonNull(gate, "gate");
    this.up = Objects.requireNonNull(up, "up");
    this.activated = Objects.requireNonNull(activated, "activated");
    this.projected = Objects.requireNonNull(projected, "projected");
    this.laneScratch = Objects.requireNonNull(laneScratch, "laneScratch");
    this.inputActivation = GgufQ8_0Batch.allocate(batchCapacity, embeddingDim);
    this.outputActivation = GgufQ8_0Batch.allocate(batchCapacity, hiddenDim);
    this.stagePlan =
        GgufStagePlan.of(
            GgufStagePlan.stage(hiddenDim / Q8_BLOCK_SIZE, this::projectGateUpAndActivate),
            GgufStagePlan.stage(embeddingDim, this::projectDown));
  }

  boolean supports(LlamaWeights.LayerWeights layer) {
    Objects.requireNonNull(layer, "layer");
    return layer.ffnGateType() == GgufTensorType.Q4_0
        && layer.ffnUpType() == GgufTensorType.Q4_0
        && layer.ffnDownType() == GgufTensorType.Q4_0;
  }

  void execute(LlamaWeights.LayerWeights layer, float[] normalizedInput, int batchSize) {
    Objects.requireNonNull(layer, "layer");
    if (!supports(layer)) {
      throw new IllegalArgumentException("staged Q4 FFN requires Q4_0 gate, up, and down weights");
    }
    if (batchSize < 2 || batchSize > batchCapacity) {
      throw new IllegalArgumentException(
          "batchSize must be between 2 and " + batchCapacity + ": " + batchSize);
    }
    inputActivation.quantizeForQ4(normalizedInput, batchSize, q4Kernel);
    activeLayer = layer;
    activeBatchSize = batchSize;
    try {
      stagePlan.execute();
    } finally {
      activeLayer = null;
      activeBatchSize = 0;
    }
  }

  private void projectGateUpAndActivate(int fromBlock, int toBlock) {
    int fromRow = fromBlock * Q8_BLOCK_SIZE;
    int toRow = toBlock * Q8_BLOCK_SIZE;
    LlamaWeights.LayerWeights layer = activeLayer;
    VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
        layer.ffnGate(),
        activeBatchSize,
        hiddenDim,
        embeddingDim,
        fromRow,
        toRow,
        gate,
        inputActivation,
        laneScratch,
        q4Kernel);
    VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
        layer.ffnUp(),
        activeBatchSize,
        hiddenDim,
        embeddingDim,
        fromRow,
        toRow,
        up,
        inputActivation,
        laneScratch,
        q4Kernel);

    int length = toRow - fromRow;
    for (int batch = 0; batch < activeBatchSize; batch++) {
      int rowOffset = batch * hiddenDim + fromRow;
      TensorOps.swiGlu(activated, rowOffset, gate, rowOffset, up, rowOffset, length);
    }
    outputActivation.quantizeBlockRangeForQ4(
        activated, activeBatchSize, fromBlock, toBlock, q4Kernel);
  }

  private void projectDown(int fromRow, int toRow) {
    VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
        activeLayer.ffnDown(),
        activeBatchSize,
        embeddingDim,
        hiddenDim,
        fromRow,
        toRow,
        projected,
        outputActivation,
        laneScratch,
        q4Kernel);
  }
}
