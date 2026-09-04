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
package com.integrallis.models.backend.purejava.soprano;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Objects;

/** Java-native ConvNeXt and inverse-STFT decoder for Soprano acoustic features. */
final class SopranoVocoder {

  private static final float LAYER_NORM_EPSILON = 1.0e-6f;

  private final SopranoConfig config;
  private final SopranoVocoderWeights weights;

  SopranoVocoder(SopranoConfig config, SopranoVocoderWeights weights) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    if (weights.layerCount() != config.decoderLayers()) {
      throw new IllegalArgumentException("Soprano vocoder layer count does not match its config");
    }
  }

  /** Decodes frame-major transformer features into normalized 32 kHz mono PCM samples. */
  synchronized float[] decode(float[] features, int frames) {
    Objects.requireNonNull(features, "features");
    if (frames <= 0 || features.length != Math.multiplyExact(frames, config.hiddenSize())) {
      throw new IllegalArgumentException(
          "Soprano features do not match frame count and hidden size");
    }

    int outputFrames = Math.addExact(Math.multiplyExact(config.upscale(), frames - 1), 1);
    int dim = config.decoderDim();
    int intermediate = config.decoderIntermediateSize();
    float[] upsampled =
        SopranoVocoderMath.interpolateAligned(
            features, frames, config.hiddenSize(), config.upscale());
    float[] hidden = new float[Math.multiplyExact(outputFrames, dim)];
    projectFloat(hidden, upsampled, outputFrames, weights.embedding(), dim, config.hiddenSize());
    addBias(hidden, outputFrames, dim, weights.embeddingBias());
    normalizeRows(
        hidden, hidden, outputFrames, dim, weights.inputNormWeight(), weights.inputNormBias());

    Scratch scratch = new Scratch(outputFrames, Math.max(dim, intermediate), maxQ40Rows());
    float[] normalized = new float[hidden.length];
    float[] up = new float[Math.multiplyExact(outputFrames, intermediate)];
    float[] down = new float[hidden.length];
    for (int layer = 0; layer < weights.layerCount(); layer++) {
      SopranoVocoderWeights.Block block = weights.layer(layer);
      float[] convolved =
          SopranoVocoderMath.depthwiseConv1d(
              hidden,
              outputFrames,
              dim,
              block.depthwiseWeight(),
              block.depthwiseBias(),
              config.decoderKernelSize());
      normalizeRows(normalized, convolved, outputFrames, dim, block.normWeight(), block.normBias());
      project(up, normalized, outputFrames, block.pointwiseUp(), scratch);
      addBias(up, outputFrames, intermediate, block.pointwiseUpBias());
      TensorOps.geluErf(up, 0, up, 0, up.length);
      project(down, up, outputFrames, block.pointwiseDown(), scratch);
      addBias(down, outputFrames, dim, block.pointwiseDownBias());
      for (int frame = 0; frame < outputFrames; frame++) {
        int offset = frame * dim;
        for (int channel = 0; channel < dim; channel++) {
          hidden[offset + channel] += down[offset + channel] * block.gamma()[channel];
        }
      }
    }

    normalizeRows(
        normalized, hidden, outputFrames, dim, weights.finalNormWeight(), weights.finalNormBias());
    float[] head = new float[Math.multiplyExact(outputFrames, config.fftSize() + 2)];
    project(head, normalized, outputFrames, weights.head(), scratch);
    addBias(head, outputFrames, config.fftSize() + 2, weights.headBias());
    return SopranoIstft.decode(
        head, outputFrames, config.fftSize(), config.hopLength(), weights.window());
  }

  private int maxQ40Rows() {
    int rows = weights.head().type() == GgufTensorType.Q4_0 ? weights.head().rows() : 0;
    for (int layer = 0; layer < weights.layerCount(); layer++) {
      SopranoVocoderWeights.Block block = weights.layer(layer);
      if (block.pointwiseUp().type() == GgufTensorType.Q4_0) {
        rows = Math.max(rows, block.pointwiseUp().rows());
      }
      if (block.pointwiseDown().type() == GgufTensorType.Q4_0) {
        rows = Math.max(rows, block.pointwiseDown().rows());
      }
    }
    return rows;
  }

  private static void project(
      float[] output,
      float[] input,
      int frames,
      SopranoVocoderWeights.Matrix matrix,
      Scratch scratch) {
    TensorOps.ggufBatchedMatmul(
        output,
        input,
        matrix.data(),
        matrix.type(),
        frames,
        matrix.rows(),
        matrix.columns(),
        scratch.quantizedActivations,
        scratch.quantizedActivationScales,
        scratch.quantizedActivationZeroPointCorrections,
        scratch.quantizedActivationSums,
        scratch.q4LaneScratch,
        GgufQ4Kernel.WIDENED);
  }

  private static void projectFloat(
      float[] output, float[] input, int frames, float[] matrix, int rows, int columns) {
    for (int frame = 0; frame < frames; frame++) {
      VectorUtil.batchDotProductExact(
          input, frame * columns, matrix, 0, columns, rows, columns, output, frame * rows);
    }
  }

  private static void normalizeRows(
      float[] output, float[] input, int frames, int columns, float[] scale, float[] bias) {
    for (int frame = 0; frame < frames; frame++) {
      int offset = frame * columns;
      TensorOps.layerNorm(output, offset, input, offset, scale, bias, columns, LAYER_NORM_EPSILON);
    }
  }

  private static void addBias(float[] values, int frames, int columns, float[] bias) {
    for (int frame = 0; frame < frames; frame++) {
      int offset = frame * columns;
      for (int column = 0; column < columns; column++) {
        values[offset + column] += bias[column];
      }
    }
  }

  private static final class Scratch {
    private final byte[] quantizedActivations;
    private final float[] quantizedActivationScales;
    private final int[] quantizedActivationZeroPointCorrections;
    private final short[] quantizedActivationSums;
    private final float[] q4LaneScratch;

    private Scratch(int frames, int widestActivation, int q4Rows) {
      quantizedActivations = new byte[Math.multiplyExact(frames, widestActivation)];
      quantizedActivationScales =
          new float[Math.multiplyExact(frames, (widestActivation + 31) / 32)];
      quantizedActivationZeroPointCorrections =
          new int[Math.multiplyExact(frames, (widestActivation + 3) / 4)];
      quantizedActivationSums = new short[Math.multiplyExact(frames, (widestActivation + 15) / 16)];
      q4LaneScratch =
          q4Rows == 0
              ? new float[0]
              : new float[Math.multiplyExact(Math.multiplyExact(frames, q4Rows), 8)];
    }
  }
}
