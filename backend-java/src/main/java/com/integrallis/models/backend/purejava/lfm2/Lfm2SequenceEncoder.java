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
package com.integrallis.models.backend.purejava.lfm2;

import com.integrallis.models.backend.purejava.SequenceEncoder;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.util.Arrays;
import java.util.Objects;

/**
 * LFM2 as a bidirectional encoder, for the files that publish {@code attention.causal = false}.
 *
 * <p>LFM2.5-Embedding-350M is an {@code lfm2} file in every respect except the mask: same hybrid of
 * attention layers and gated short convolutions, same tied embedding table, same {@code
 * token_embd_norm} as the final norm. What differs is that it declares {@code lfm2.attention.causal
 * = false} and {@code lfm2.pooling_type = 2}, and llama.cpp reads the first of those generically
 * for every architecture ({@code llama-model.cpp:1069}), so the reference runs the same graph with
 * two operators changed:
 *
 * <ul>
 *   <li><b>attention spans the whole sequence</b> rather than the past, which is why this is a
 *       sequence encoder and not a batched form of {@link Lfm2ForwardPass}: a token's output
 *       depends on tokens after it, so a layer cannot be finished for position <i>p</i> before
 *       position <i>p+1</i> has been through the layer below. The pass is layer-major over the
 *       whole sequence, where the decoder is token-major over layers.
 *   <li><b>the short convolution becomes a centred window</b> with symmetric zero padding instead
 *       of a shift register over the past -- see {@link Lfm2ShortConv#applyCentered}.
 * </ul>
 *
 * <p>Running the decoder on one of these files does not fail. It produced worst-probe cosine 0.03
 * against llama.cpp on all seven published artifacts, F16 and BF16 included, which is the shape of
 * this class's reason for existing: a causal pass over a bidirectional model is a plausible vector
 * that is not the model's.
 */
public final class Lfm2SequenceEncoder implements SequenceEncoder {

  private final Lfm2Config config;
  private final Lfm2Weights weights;
  private final RotaryTable rotary;
  private final int maxKeyDim;
  private final int maxQ40ProjectionRows;
  private final float[] tokenEmbedding;
  private final double[] pooledSum;

  private float[] hidden = new float[0];
  private float[] normalized = new float[0];
  private float[] queries = new float[0];
  private float[] keys = new float[0];
  private float[] values = new float[0];
  private float[] attention = new float[0];
  private float[] projected = new float[0];
  private float[] convProjected = new float[0];
  private float[] convGated = new float[0];
  private float[] gate = new float[0];
  private float[] up = new float[0];
  private float[] feedForward = new float[0];
  private float[] scores = new float[0];
  private byte[] quantizedActivations = new byte[0];
  private float[] quantizedActivationScales = new float[0];
  private int[] quantizedActivationZeroPointCorrections = new int[0];
  private short[] quantizedActivationSums = new short[0];
  private float[] q4LaneScratch = new float[0];
  private int capacity;

  /** Loads a bidirectional LFM2 encoder from an already parsed GGUF. */
  public static Lfm2SequenceEncoder fromGgufFile(GgufFile file, Lfm2Config config) {
    Objects.requireNonNull(config, "config");
    if (!config.encodesWholeSequence()) {
      throw new IllegalArgumentException(
          "this file declares causal attention and is a decoder; load Lfm2ForwardPass instead");
    }
    return new Lfm2SequenceEncoder(config, Lfm2Weights.fromGgufFile(file, config));
  }

  Lfm2SequenceEncoder(Lfm2Config config, Lfm2Weights weights) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    this.rotary = new RotaryTable(config.headDim(), config.ropeTheta(), 1.0f);
    int widestKeyDim = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesAttention(layer)) {
        widestKeyDim = Math.max(widestKeyDim, config.keyDim(layer));
      }
    }
    this.maxKeyDim = widestKeyDim;
    this.maxQ40ProjectionRows = weights.maxQ40ProjectionRows();
    this.tokenEmbedding = new float[config.embeddingDim()];
    this.pooledSum = new double[config.embeddingDim()];
  }

  @Override
  public synchronized float[] encode(int[] tokens) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (tokens.length > config.contextLength()) {
      throw new IllegalArgumentException(
          "sequence of "
              + tokens.length
              + " tokens exceeds the model's context length of "
              + config.contextLength());
    }
    int sequenceLength = tokens.length;
    int dim = config.embeddingDim();
    ensureCapacity(sequenceLength);
    for (int position = 0; position < sequenceLength; position++) {
      weights.embedToken(tokens[position], tokenEmbedding);
      System.arraycopy(tokenEmbedding, 0, hidden, position * dim, dim);
    }
    for (int layer = 0; layer < config.numLayers(); layer++) {
      executeLayer(layer, sequenceLength);
    }
    // The final norm is token_embd_norm despite the name; see Lfm2Weights. llama.cpp assigns
    // res->t_embd to exactly this activation and only then applies the vocabulary projection, so
    // the embedding is the normalized state and not the state before it.
    normalizeRows(normalized, hidden, sequenceLength, weights.outputNorm());
    return pool(sequenceLength);
  }

  private void executeLayer(int layer, int sequenceLength) {
    int dim = config.embeddingDim();
    int hiddenDim = config.hiddenDim();
    Lfm2Weights.LayerWeights layerWeights = weights.layer(layer);
    normalizeRows(normalized, hidden, sequenceLength, layerWeights.attentionNorm());
    // Both mixers leave their result in `projected`, so the residual is the same either way.
    if (config.usesAttention(layer)) {
      attend(layer, layerWeights, sequenceLength);
    } else {
      project(convProjected, normalized, sequenceLength, layerWeights.shortConvInProjection());
      Lfm2ShortConv.applyCentered(
          convProjected,
          sequenceLength,
          dim,
          layerWeights.shortConvKernel(),
          config.shortConvCache(),
          convGated);
      project(projected, convGated, sequenceLength, layerWeights.shortConvOutProjection());
    }
    add(hidden, projected, sequenceLength * dim);

    normalizeRows(normalized, hidden, sequenceLength, layerWeights.ffnNorm());
    project(gate, normalized, sequenceLength, layerWeights.gateProjection());
    project(up, normalized, sequenceLength, layerWeights.upProjection());
    TensorOps.swiGlu(feedForward, gate, up, sequenceLength * hiddenDim);
    project(projected, feedForward, sequenceLength, layerWeights.downProjection());
    add(hidden, projected, sequenceLength * dim);
  }

  private void attend(int layer, Lfm2Weights.LayerWeights layerWeights, int sequenceLength) {
    int headDim = config.headDim();
    int numHeads = config.numHeads();
    int numKvHeads = config.numKvHeads(layer);
    int keyDim = config.keyDim(layer);
    int queryDim = config.queryDim();

    project(queries, normalized, sequenceLength, layerWeights.queryProjection());
    project(keys, normalized, sequenceLength, layerWeights.keyProjection());
    project(values, normalized, sequenceLength, layerWeights.valueProjection());

    // Per-head norms first, then rope: the reference's order, and the same order the decoder uses.
    for (int position = 0; position < sequenceLength; position++) {
      int queryBase = position * queryDim;
      int keyBase = position * keyDim;
      for (int head = 0; head < numHeads; head++) {
        int offset = queryBase + head * headDim;
        TensorOps.rmsNorm(
            queries,
            offset,
            queries,
            offset,
            layerWeights.queryNorm(),
            headDim,
            config.rmsNormEpsilon());
      }
      for (int head = 0; head < numKvHeads; head++) {
        int offset = keyBase + head * headDim;
        TensorOps.rmsNorm(
            keys, offset, keys, offset, layerWeights.keyNorm(), headDim, config.rmsNormEpsilon());
      }
      rotary.prepare(position);
      for (int head = 0; head < numHeads; head++) {
        rotary.apply(queries, queryBase + head * headDim, true);
      }
      for (int head = 0; head < numKvHeads; head++) {
        rotary.apply(keys, keyBase + head * headDim, true);
      }
    }

    int groupSize = numHeads / numKvHeads;
    float scale = (float) (1.0 / Math.sqrt(headDim));
    Arrays.fill(attention, 0, sequenceLength * queryDim, 0.0f);
    for (int position = 0; position < sequenceLength; position++) {
      for (int head = 0; head < numHeads; head++) {
        int kvHead = head / groupSize;
        int queryOffset = position * queryDim + head * headDim;
        // The whole sequence, not the past: this is the one line the causal pass cannot express.
        for (int keyPosition = 0; keyPosition < sequenceLength; keyPosition++) {
          int keyOffset = keyPosition * keyDim + kvHead * headDim;
          float dot = 0.0f;
          for (int index = 0; index < headDim; index++) {
            dot += queries[queryOffset + index] * keys[keyOffset + index];
          }
          scores[keyPosition] = dot * scale;
        }
        TensorOps.softmax(scores, 0, sequenceLength);
        for (int keyPosition = 0; keyPosition < sequenceLength; keyPosition++) {
          float weight = scores[keyPosition];
          int valueOffset = keyPosition * keyDim + kvHead * headDim;
          for (int index = 0; index < headDim; index++) {
            attention[queryOffset + index] += weight * values[valueOffset + index];
          }
        }
      }
    }
    project(projected, attention, sequenceLength, layerWeights.attentionOutputProjection());
  }

  private float[] pool(int sequenceLength) {
    int dim = config.embeddingDim();
    float[] pooled = new float[dim];
    switch (config.pooling()) {
      case CLS -> System.arraycopy(normalized, 0, pooled, 0, dim);
      case LAST -> System.arraycopy(normalized, (sequenceLength - 1) * dim, pooled, 0, dim);
      case MEAN -> {
        Arrays.fill(pooledSum, 0.0);
        for (int position = 0; position < sequenceLength; position++) {
          int offset = position * dim;
          for (int index = 0; index < dim; index++) {
            pooledSum[index] += normalized[offset + index];
          }
        }
        for (int index = 0; index < dim; index++) {
          pooled[index] = (float) (pooledSum[index] / sequenceLength);
        }
      }
      // RANK is a reranker's scalar score through a classifier head, and this file carries no
      // such head -- its tensor list is the base model's. Returning the CLS vector under a rank
      // pooling would be a different model's answer.
      default ->
          throw new UnsupportedOperationException(
              "LFM2 pooling " + config.pooling() + " is not supported for embeddings");
    }
    return pooled;
  }

  private void normalizeRows(
      float[] output, float[] input, int sequenceLength, float[] normWeight) {
    int dim = config.embeddingDim();
    for (int position = 0; position < sequenceLength; position++) {
      int offset = position * dim;
      TensorOps.rmsNorm(output, offset, input, offset, normWeight, dim, config.rmsNormEpsilon());
    }
  }

  private void project(
      float[] output, float[] input, int sequenceLength, Lfm2Weights.Matrix matrix) {
    TensorOps.ggufBatchedMatmul(
        output,
        input,
        matrix.data(),
        matrix.type(),
        sequenceLength,
        matrix.rows(),
        matrix.columns(),
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        GgufQ4Kernel.WIDENED);
  }

  private static void add(float[] target, float[] addend, int count) {
    for (int index = 0; index < count; index++) {
      target[index] += addend[index];
    }
  }

  private void ensureCapacity(int sequenceLength) {
    if (capacity >= sequenceLength) {
      return;
    }
    int dim = config.embeddingDim();
    int hiddenDim = config.hiddenDim();
    int queryDim = config.queryDim();
    int stateElements = Math.multiplyExact(sequenceLength, dim);
    hidden = new float[stateElements];
    normalized = new float[stateElements];
    projected = new float[stateElements];
    convGated = new float[stateElements];
    convProjected = new float[Math.multiplyExact(stateElements, 3)];
    queries = new float[Math.multiplyExact(sequenceLength, queryDim)];
    attention = new float[Math.multiplyExact(sequenceLength, queryDim)];
    keys = new float[Math.multiplyExact(sequenceLength, maxKeyDim)];
    values = new float[Math.multiplyExact(sequenceLength, maxKeyDim)];
    gate = new float[Math.multiplyExact(sequenceLength, hiddenDim)];
    up = new float[Math.multiplyExact(sequenceLength, hiddenDim)];
    feedForward = new float[Math.multiplyExact(sequenceLength, hiddenDim)];
    scores = new float[sequenceLength];

    // The convolution's in_proj is the widest activation any projection consumes at 3 * dim, and
    // the feed-forward's down projection consumes hiddenDim; sizing to the smaller of the two
    // would overflow the scratch on whichever layer kind came second.
    int widestActivation = Math.max(Math.max(dim, hiddenDim), Math.max(queryDim, 3 * dim));
    quantizedActivations = new byte[Math.multiplyExact(sequenceLength, widestActivation)];
    quantizedActivationScales =
        new float[Math.multiplyExact(sequenceLength, (widestActivation + 31) / 32)];
    quantizedActivationZeroPointCorrections =
        new int[Math.multiplyExact(sequenceLength, (widestActivation + 3) / 4)];
    quantizedActivationSums =
        new short[Math.multiplyExact(sequenceLength, (widestActivation + 15) / 16)];
    q4LaneScratch =
        maxQ40ProjectionRows == 0
            ? new float[0]
            : new float
                [Math.multiplyExact(Math.multiplyExact(sequenceLength, maxQ40ProjectionRows), 8)];
    capacity = sequenceLength;
  }
}
