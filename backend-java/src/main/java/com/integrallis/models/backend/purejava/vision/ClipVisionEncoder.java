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
package com.integrallis.models.backend.purejava.vision;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.DenseProjectionKernels;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * Pure-Java CLIP vision encoder.
 *
 * <p>Loads weights from a CLIP mmproj GGUF and runs a full ViT forward pass, producing vision
 * feature tokens. A trained projector is required before decoder injection.
 *
 * <p>The forward pass reuses existing kernels:
 *
 * <ul>
 *   <li>{@link com.integrallis.models.backend.purejava.ops.TensorOps#ggufBatchedMatmul} for linear
 *       projections
 *   <li>{@link com.integrallis.models.backend.purejava.ops.TensorOps#layerNorm} for pre-layer norms
 *   <li>{@link com.integrallis.models.backend.purejava.ops.TensorOps#softmax} for attention scores
 *   <li>{@link com.integrallis.models.backend.purejava.ops.TensorOps#geluErf} for FFN activation
 *   <li>{@link com.integrallis.vectors.core.VectorUtil#batchDotProductExact} and {@link
 *       VectorUtil#addWeightedRowsInPlace} for attention
 * </ul>
 *
 * <p>Quantized weight types supported: F32, F16, BF16, Q4_0, Q5_0, Q8_0 and K-quants through the
 * standard GGUF dequantization path.
 */
public final class ClipVisionEncoder {

  private final java.util.Map<
          ClipVisionWeights.Matrix,
          com.integrallis.models.backend.purejava.spi.DenseProjectionKernels.Projection>
      projections = new java.util.IdentityHashMap<>();
  private final ClipVisionConfig config;
  private final ClipVisionWeights weights;
  private final GgufBatchedMatrixKernel matrixKernel;

  // Working buffers, sized for one image at a time
  private final DenseProjectionKernels denseKernels;
  private final DenseProjectionKernels.Projection patchProjection;
  private final float[] patchPixels;
  private final float[] patchOutput;
  private final float[] hidden;
  private final float[] normalized;
  private final float[] queryKeyValues;
  private final float[] queries;
  private final float[] keys;
  private final float[] values;
  private final float[] attention;
  private final float[] attentionProjected;
  private final float[] feedForward;
  private final float[] feedForwardProjected;
  private final float[] scores;

  public ClipVisionEncoder(
      ClipVisionConfig config, ClipVisionWeights weights, GgufBatchedMatrixKernel matrixKernel) {
    this(config, weights, matrixKernel, null);
  }

  ClipVisionEncoder(
      ClipVisionConfig config,
      ClipVisionWeights weights,
      GgufBatchedMatrixKernel matrixKernel,
      com.integrallis.models.backend.purejava.spi.DenseProjectionKernels denseKernels) {
    this.config = config;
    this.weights = weights;
    this.matrixKernel = matrixKernel != null ? matrixKernel : GgufBatchedMatrixKernel.none();

    for (int i = 0; i < config.numLayers(); i++) {
      var layer = weights.layer(i);
      for (var matrix :
          new ClipVisionWeights.Matrix[] {
            layer.attnQkv(), layer.attnOut(), layer.ffnUp(), layer.ffnDown()
          }) {
        projections.put(matrix, VisionProjection.prepare(matrix, denseKernels));
      }
    }
    int tokens = config.numTokens();
    int dim = config.hiddenSize();
    int hiddenDim = config.intermediateSize();

    long batchElementsLong = Math.multiplyExact((long) tokens, dim);
    if (batchElementsLong > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "tokens * dim exceeds Integer.MAX_VALUE: " + tokens + " * " + dim);
    }
    int batchElements = (int) batchElementsLong;

    this.denseKernels = denseKernels;
    this.patchProjection = VisionProjection.prepare(weights.patchEmbed(), denseKernels);
    this.patchPixels = new float[(tokens - 1) * 3 * config.patchSize() * config.patchSize()];
    this.patchOutput = new float[(tokens - 1) * dim];
    this.hidden = new float[batchElements];
    this.normalized = new float[batchElements];
    this.queryKeyValues = new float[Math.multiplyExact(batchElements, 3)];
    this.queries = new float[batchElements];
    this.keys = new float[batchElements];
    this.values = new float[batchElements];
    this.attention = new float[batchElements];
    this.attentionProjected = new float[batchElements];
    this.feedForward = new float[Math.multiplyExact(tokens, hiddenDim)];
    this.feedForwardProjected = new float[batchElements];
    this.scores = new float[Math.multiplyExact(tokens, tokens)];
  }

  /**
   * Encodes one preprocessed image and returns vision feature tokens.
   *
   * @param chw preprocessed image in CHW layout, length = 3 * imageSize * imageSize
   * @return vision tokens, shape [numTokens][hiddenSize], stored row-major in a flat array
   */
  public float[] encode(float[] chw) {
    int tokens = config.numTokens();
    int dim = config.hiddenSize();
    int patchSize = config.patchSize();
    int imageSize = config.imageSize();
    int channels = 3;
    int expectedPixels = Math.multiplyExact(channels, Math.multiplyExact(imageSize, imageSize));
    if (chw.length != expectedPixels) {
      throw new IllegalArgumentException(
          "Expected "
              + expectedPixels
              + " pixels (3 * "
              + imageSize
              + " * "
              + imageSize
              + ") but got "
              + chw.length);
    }

    for (float value : chw) {
      if (!Float.isFinite(value))
        throw new IllegalArgumentException("Image contains nonfinite values");
    }
    Arrays.fill(hidden, 0);
    // Patch embedding: writes patches to hidden[dim..], leaving hidden[0..dim] for CLS
    patchEmbed(chw, imageSize, patchSize, channels, dim);

    // Prepend class token
    float[] classEmb = weights.classEmb();
    if (classEmb != null) {
      System.arraycopy(classEmb, 0, hidden, 0, dim);
    }

    // Add position embeddings (covers all tokens including CLS)
    addPositionEmbeddings(tokens, dim);

    // Pre-transformer layer norm
    if (weights.preLnWeight() != null) {
      normalizeRows(hidden, hidden, tokens, weights.preLnWeight(), weights.preLnBias());
    }

    // Transformer blocks
    for (int layer = 0; layer < config.numLayers(); layer++) {
      executeLayer(weights.layer(layer), tokens, dim);
    }

    // Post-transformer layer norm
    if (weights.postLnWeight() != null) {
      normalizeRows(hidden, hidden, tokens, weights.postLnWeight(), weights.postLnBias());
    }

    // Return a copy so callers can't mutate internal state
    return Arrays.copyOf(hidden, tokens * dim);
  }

  /** Patch embedding: each patch becomes a vector via linear projection. */
  private void patchEmbed(float[] chw, int imageSize, int patchSize, int channels, int dim) {
    int patchesPerSide = imageSize / patchSize;
    int patchPixels = channels * patchSize * patchSize;
    for (int py = 0; py < patchesPerSide; py++) {
      for (int px = 0; px < patchesPerSide; px++) {
        int patchIndex = py * patchesPerSide + px;
        for (int c = 0; c < channels; c++) {
          for (int y = 0; y < patchSize; y++) {
            int src = c * imageSize * imageSize + (py * patchSize + y) * imageSize + px * patchSize;
            int dst = patchIndex * patchPixels + c * patchSize * patchSize + y * patchSize;
            System.arraycopy(chw, src, this.patchPixels, dst, patchSize);
          }
        }
      }
    }
    int count = patchesPerSide * patchesPerSide;
    patchProjection.project(patchOutput, this.patchPixels, count);
    addBias(patchOutput, count, dim, weights.patchEmbedBias());
    System.arraycopy(patchOutput, 0, hidden, dim, count * dim);
  }

  /** Add learned position embeddings to each token. */
  private void addPositionEmbeddings(int tokens, int dim) {
    float[] posEmbed = weights.positionEmbedF32();
    for (int t = 0; t < tokens; t++) {
      int pos = t;
      int offset = pos * dim;
      for (int h = 0; h < dim; h++) {
        hidden[t * dim + h] += posEmbed[offset + h];
      }
    }
  }

  /** One transformer block: pre-norm attention + residual, pre-norm FFN + residual. */
  private void executeLayer(ClipVisionWeights.Layer layer, int tokens, int dim) {
    int batchElements = Math.multiplyExact(tokens, dim);

    // 1. Pre-attention layer norm
    normalizeRows(normalized, hidden, tokens, layer.ln1Weight(), layer.ln1Bias());

    // 2. QKV projection (fused)
    projectQ8_0(
        queryKeyValues, normalized, tokens, layer.attnQkvQ8_0(), dim * 3, dim, layer.attnQkv());
    addBias(queryKeyValues, tokens, dim * 3, layer.attnQkvBias());

    // 3. Split into Q, K, V
    splitQueryKeyValue(tokens, dim);

    // 4. Self-attention
    attend(tokens, dim);

    // 5. Output projection + residual
    projectQ8_0(
        attentionProjected, attention, tokens, layer.attnOutQ8_0(), dim, dim, layer.attnOut());
    addBias(attentionProjected, tokens, dim, layer.attnOutBias());
    for (int i = 0; i < batchElements; i++) {
      attentionProjected[i] += hidden[i];
    }

    // 6. Pre-FFN layer norm
    normalizeRows(normalized, attentionProjected, tokens, layer.ln2Weight(), layer.ln2Bias());

    // 7. FFN: up projection
    int hiddenDim = config.intermediateSize();
    projectQ8_0(feedForward, normalized, tokens, layer.ffnUpQ8_0(), hiddenDim, dim, layer.ffnUp());
    addBias(feedForward, tokens, hiddenDim, layer.ffnUpBias());

    // 8. GELU activation
    if (config.quickGelu()) {
      for (int i = 0; i < tokens * hiddenDim; i++) {
        feedForward[i] = (float) (feedForward[i] / (1.0 + Math.exp(-1.702 * feedForward[i])));
      }
    } else {
      TensorOps.gelu(feedForward, 0, feedForward, 0, tokens * hiddenDim);
    }

    // 9. FFN down projection + residual
    projectQ8_0(
        feedForwardProjected,
        feedForward,
        tokens,
        layer.ffnDownQ8_0(),
        dim,
        hiddenDim,
        layer.ffnDown());
    addBias(feedForwardProjected, tokens, dim, layer.ffnDownBias());
    for (int i = 0; i < batchElements; i++) {
      feedForwardProjected[i] += attentionProjected[i];
    }

    // Result becomes hidden for next layer
    System.arraycopy(feedForwardProjected, 0, hidden, 0, batchElements);
  }

  /** Multi-head self-attention (identical to BertForwardPass.attend but without rotary). */
  private void attend(int tokens, int dim) {
    if (denseKernels != null
        && denseKernels.attention(attention, queries, keys, values, tokens, dim, config.numHeads()))
      return;
    int numHeads = config.numHeads();
    int headDim = config.headDim();
    float scale = (float) (1.0 / Math.sqrt(headDim));
    Arrays.fill(attention, 0, tokens * dim, 0.0f);

    com.integrallis.vectors.core.GgufStagePlan.of(
            com.integrallis.vectors.core.GgufStagePlan.stage(
                tokens,
                (from, to) -> {
                  for (int position = from; position < to; position++) {
                    int scoreOffset = position * tokens;
                    for (int head = 0; head < numHeads; head++) {
                      int headOffset = head * headDim;
                      VectorUtil.batchDotProductExact(
                          queries,
                          position * dim + headOffset,
                          keys,
                          headOffset,
                          dim,
                          tokens,
                          headDim,
                          scores,
                          scoreOffset);
                      for (int keyPos = 0; keyPos < tokens; keyPos++) {
                        scores[scoreOffset + keyPos] *= scale;
                      }
                      TensorOps.softmax(scores, scoreOffset, tokens);
                      VectorUtil.addWeightedRowsInPlace(
                          attention,
                          position * dim + headOffset,
                          values,
                          headOffset,
                          dim,
                          scores,
                          scoreOffset,
                          tokens,
                          headDim);
                    }
                  }
                }))
        .execute();
  }

  private void splitQueryKeyValue(int tokens, int dim) {
    for (int t = 0; t < tokens; t++) {
      int src = t * dim * 3;
      int dst = t * dim;
      System.arraycopy(queryKeyValues, src, queries, dst, dim);
      System.arraycopy(queryKeyValues, src + dim, keys, dst, dim);
      System.arraycopy(queryKeyValues, src + dim * 2, values, dst, dim);
    }
  }

  private void normalizeRows(float[] out, float[] in, int rows, float[] weight, float[] bias) {
    int dim = config.hiddenSize();
    for (int r = 0; r < rows; r++) {
      int offset = r * dim;
      TensorOps.layerNorm(out, offset, in, offset, weight, bias, dim, config.layerNormEps());
    }
  }

  /**
   * Projects using Q8_0 quantized weights through the native kernel when available, otherwise falls
   * back to the original weight matrix.
   */
  private void projectQ8_0(
      float[] output,
      float[] input,
      int sequenceLength,
      MemorySegment q8_0Weights,
      int rows,
      int columns,
      ClipVisionWeights.Matrix fallbackMatrix) {
    if (q8_0Weights != null
        && matrixKernel.isEligible(GgufTensorType.Q8_0, sequenceLength, rows, columns)) {
      matrixKernel.multiply(
          output, input, q8_0Weights, GgufTensorType.Q8_0, sequenceLength, rows, columns);
      return;
    }
    // Fallback: use original matrix (F32/BF16/F16 path)
    project(output, input, sequenceLength, fallbackMatrix);
  }

  private void project(
      float[] output, float[] input, int sequenceLength, ClipVisionWeights.Matrix matrix) {
    if (matrixKernel.isEligible(matrix.type(), sequenceLength, matrix.rows(), matrix.columns())) {
      matrixKernel.multiply(
          output,
          input,
          matrix.data(),
          matrix.type(),
          sequenceLength,
          matrix.rows(),
          matrix.columns());
    } else {
      projections.get(matrix).project(output, input, sequenceLength);
    }
  }

  private static void addBias(float[] values, int rows, int columns, float[] bias) {
    for (int row = 0; row < rows; row++) {
      int offset = row * columns;
      for (int col = 0; col < columns; col++) {
        values[offset + col] += bias[col];
      }
    }
  }
}
