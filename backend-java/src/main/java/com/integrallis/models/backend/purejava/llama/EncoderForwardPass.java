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
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Whole-sequence forward pass for encoder models, whose positions attend in both directions.
 *
 * <p>Bidirectional attention is not a mask change bolted onto the decoder pass — it inverts the
 * loop nesting. A causal model can walk tokens one at a time because position <i>p</i> only ever
 * needs keys from positions it has already visited, which is what makes a KV cache work at all.
 * Here every position needs every other position's key at the same layer, so layer <i>L</i> cannot
 * start until layer <i>L−1</i> has produced its output for the entire sequence. The sequence is
 * therefore the unit of work, and there is no cache to carry between calls — llama.cpp likewise
 * builds no KV cache for these architectures.
 *
 * <p>Two things follow that callers should expect. Cost is quadratic in sequence length rather than
 * linear, bounded by the sliding window where the architecture uses one. And there is no
 * incremental mode: appending a token changes every position's output, so an encoder can only
 * re-encode.
 *
 * <p>The block itself is unchanged from the decoder — same norms, same projections, same rotary
 * embeddings, same gated feed-forward — so it reads from {@link LlamaConfig} and {@link
 * LlamaWeights} exactly as {@link LlamaForwardPass} does.
 *
 * <p>Not thread-safe: one sequence at a time over shared scratch.
 */
public final class EncoderForwardPass {

  /** Overrides how many positions are computed at once; mainly so tests can pin it to one. */
  public static final String PARALLELISM_PROPERTY = "models.encoder.threads";

  private final LlamaConfig config;
  private final LlamaWeights weights;
  private final DenseProjectionHead denseHead;
  private final RotaryTable globalRopeTable;
  private final RotaryTable slidingWindowRopeTable;
  private final int maxSequenceLength;

  private final float[] pooled;
  private final double[] pooledSum;
  private final Scratch[] workers;

  /**
   * The buffers one position needs while it is being computed.
   *
   * <p>Positions within a layer are independent — each reads the whole sequence's keys and values
   * and writes only its own hidden state — so the only thing preventing them from running at once
   * was sharing these. One bundle per worker makes the loops parallel without changing any
   * arithmetic.
   */
  private static final class Scratch {
    final float[] normed;
    final float[] attentionOut;
    final float[] attentionProjected;
    final float[] ffnGate;
    final float[] ffnUp;
    final float[] ffnActivated;
    final float[] ffnProjected;
    final byte[] quantizedActivation;
    final float[] quantizedActivationScales;
    final int[] quantizedActivationZeroPointCorrections;
    final short[] quantizedActivationSums;
    float[] scores = new float[0];
    float[] projection = new float[0];

    Scratch(int dim, int hiddenDim, int attentionOutputDim, int widestActivation) {
      this.normed = new float[dim];
      this.attentionOut = new float[attentionOutputDim];
      this.attentionProjected = new float[dim];
      this.ffnGate = new float[hiddenDim];
      this.ffnUp = new float[hiddenDim];
      this.ffnActivated = new float[hiddenDim];
      this.ffnProjected = new float[dim];
      this.quantizedActivation = new byte[widestActivation];
      this.quantizedActivationScales = new float[widestActivation / 32];
      this.quantizedActivationZeroPointCorrections = new int[(widestActivation + 3) / 4];
      this.quantizedActivationSums = new short[(widestActivation + 15) / 16];
    }

    float[] scores(int sequenceLength) {
      if (scores.length < sequenceLength) {
        scores = new float[sequenceLength];
      }
      return scores;
    }

    float[] projection(int rows) {
      if (projection.length < rows) {
        projection = new float[rows];
      }
      return projection;
    }
  }

  private float[] hidden = new float[0];
  private float[] queries = new float[0];
  private float[] keys = new float[0];
  private float[] values = new float[0];
  private int capacity;

  /**
   * Prepares an encoder over already-loaded weights.
   *
   * @param config the model configuration, which must describe a bidirectional architecture
   * @param weights its tensors
   * @param denseHead the sentence-transformer projection head, or null when the model has none
   * @throws IllegalArgumentException if the configuration describes a causal decoder
   */
  public EncoderForwardPass(
      LlamaConfig config, LlamaWeights weights, DenseProjectionHead denseHead) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    this.denseHead = denseHead;
    if (!config.usesBidirectionalAttention()) {
      throw new IllegalArgumentException(
          "architecture "
              + config.architecture()
              + " is a causal decoder; run it through LlamaForwardPass, which is what its"
              + " semantics require");
    }
    if (denseHead != null && denseHead.dimension() != config.embeddingDim()) {
      throw new IllegalArgumentException(
          "dense projection head is sized for "
              + denseHead.dimension()
              + " dimensions but the model is "
              + config.embeddingDim());
    }

    this.globalRopeTable =
        new RotaryTable(config.keyLength(), config.ropeTheta(), config.ropeFrequencyScale());
    this.slidingWindowRopeTable =
        config.slidingWindow() > 0
            ? new RotaryTable(config.keyLength(), config.slidingWindowRopeTheta(), 1.0f)
            : globalRopeTable;
    this.maxSequenceLength = config.contextLength();

    int dim = config.embeddingDim();
    this.pooled = new float[dim];
    this.pooledSum = new double[dim];

    // Sized for the widest activation any projection consumes, so the quantized kernels never
    // allocate mid-pass. Scales are sized at the smaller of the two GGUF block widths, which is
    // the count a 32-wide block needs and an upper bound for a 256-wide one.
    int widestActivation = Math.max(dim, Math.max(config.hiddenDim(), config.attentionOutputDim()));
    this.workers = new Scratch[resolveParallelism()];
    for (int worker = 0; worker < workers.length; worker++) {
      workers[worker] =
          new Scratch(dim, config.hiddenDim(), config.attentionOutputDim(), widestActivation);
    }
  }

  /**
   * How many positions to compute at once.
   *
   * <p>Defaults to the available processors. The projections here are too narrow to trip vectors'
   * own parallel-matmul threshold — the widest is under a million elements — so without this the
   * whole encode runs on one core however many are free.
   *
   * <p>Settable to one via {@code -Dmodels.encoder.threads}, which is how a test pins the schedule
   * when it wants to prove the result does not depend on it.
   */
  private static int resolveParallelism() {
    String configured = System.getProperty(PARALLELISM_PROPERTY);
    if (configured != null && !configured.isBlank()) {
      int requested = Integer.parseInt(configured.trim());
      if (requested < 1) {
        throw new IllegalArgumentException(PARALLELISM_PROPERTY + " must be >= 1: " + requested);
      }
      return requested;
    }
    return Math.max(1, Runtime.getRuntime().availableProcessors());
  }

  /**
   * Runs {@code body} over every position, splitting them across workers.
   *
   * <p>Strided rather than blocked so a sliding window, whose cost varies with position, does not
   * leave one worker with all the expensive rows.
   */
  private void forEachPosition(int sequenceLength, PositionWork body) {
    int parallelism = Math.min(workers.length, sequenceLength);
    if (parallelism == 1) {
      for (int position = 0; position < sequenceLength; position++) {
        body.run(position, workers[0]);
      }
      return;
    }
    Thread[] threads = new Thread[parallelism];
    AtomicReference<RuntimeException> failure = new AtomicReference<>();
    for (int worker = 0; worker < parallelism; worker++) {
      int index = worker;
      threads[worker] =
          Thread.ofVirtual()
              .start(
                  () -> {
                    try {
                      for (int position = index;
                          position < sequenceLength;
                          position += parallelism) {
                        body.run(position, workers[index]);
                      }
                    } catch (RuntimeException e) {
                      failure.compareAndSet(null, e);
                    }
                  });
    }
    for (Thread thread : threads) {
      try {
        thread.join();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("interrupted while encoding", e);
      }
    }
    RuntimeException thrown = failure.get();
    if (thrown != null) {
      throw thrown;
    }
  }

  /** One position's share of a layer. */
  @FunctionalInterface
  private interface PositionWork {
    void run(int position, Scratch scratch);
  }

  /** Longest sequence this encoder accepts, from the model's trained context length. */
  public int maxSequenceLength() {
    return maxSequenceLength;
  }

  /** Width of the vector {@link #encode(int[])} returns. */
  public int dimension() {
    return config.embeddingDim();
  }

  /**
   * Encodes a whole sequence into one pooled vector.
   *
   * <p>Applies the model's own pooling and projection head, because both are part of the trained
   * model rather than caller policy. L2 normalization is left to the caller: unlike pooling, it is
   * a genuine choice about what the vector is used for.
   *
   * @param tokens the tokenized text, including whatever prefix and suffix tokens the tokenizer
   *     adds
   * @return a newly allocated pooled embedding, not normalized
   * @throws IllegalArgumentException if the sequence is empty or longer than the trained context
   */
  public float[] encode(int[] tokens) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (tokens.length > maxSequenceLength) {
      throw new IllegalArgumentException(
          "sequence of "
              + tokens.length
              + " tokens exceeds the model's context length of "
              + maxSequenceLength);
    }

    int sequenceLength = tokens.length;
    ensureCapacity(sequenceLength);
    embedTokens(tokens, sequenceLength);
    prepareRotaryFactors(sequenceLength);

    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights layerWeights = weights.layer(layer);
      projectQueryKeyValue(layerWeights, layer, sequenceLength);
      applyAttentionAndFeedForward(layerWeights, layer, sequenceLength);
    }

    return poolAndProject(sequenceLength);
  }

  private void embedTokens(int[] tokens, int sequenceLength) {
    int dim = config.embeddingDim();
    float scale = config.embeddingScale();
    forEachPosition(
        sequenceLength,
        (position, scratch) -> {
          weights.embedToken(tokens[position], scratch.normed);
          int base = position * dim;
          for (int index = 0; index < dim; index++) {
            hidden[base + index] = scratch.normed[index] * scale;
          }
        });
  }

  private void prepareRotaryFactors(int sequenceLength) {
    globalRopeTable.prepareBatch(0, sequenceLength);
    if (slidingWindowRopeTable != globalRopeTable) {
      slidingWindowRopeTable.prepareBatch(0, sequenceLength);
    }
  }

  /**
   * Projects every position's query, key and value for one layer.
   *
   * <p>Runs to completion before any attention does. Attention at one position reads keys and
   * values at every other, so a fused loop would have later positions attending to the previous
   * layer's keys — a mistake that costs a little accuracy rather than failing.
   */
  private void projectQueryKeyValue(
      LlamaWeights.LayerWeights layerWeights, int layer, int sequenceLength) {
    int dim = config.embeddingDim();
    int keyLength = config.keyLength();
    int queryDim = config.queryDim();
    int keyDim = config.keyDim();
    int valueDim = config.valueDim();
    boolean neox = config.usesNeoxRope();
    boolean rope = config.usesRope(layer);
    RotaryTable rotary = config.usesSlidingWindow(layer) ? slidingWindowRopeTable : globalRopeTable;

    forEachPosition(
        sequenceLength,
        (position, scratch) -> {
          float[] normed = scratch.normed;
          TensorOps.rmsNorm(
              normed,
              0,
              hidden,
              position * dim,
              layerWeights.attentionNorm(),
              dim,
              config.rmsNormEps());

          int queryOffset = position * queryDim;
          int keyOffset = position * keyDim;
          int valueOffset = position * valueDim;
          matmul(
              scratch,
              queries,
              queryOffset,
              normed,
              layerWeights.wq(),
              layerWeights.wqType(),
              queryDim,
              dim);
          matmul(
              scratch,
              keys,
              keyOffset,
              normed,
              layerWeights.wk(),
              layerWeights.wkType(),
              keyDim,
              dim);
          matmul(
              scratch,
              values,
              valueOffset,
              normed,
              layerWeights.wv(),
              layerWeights.wvType(),
              valueDim,
              dim);
          addOptionalBias(queries, queryOffset, layerWeights.qBias());
          addOptionalBias(keys, keyOffset, layerWeights.kBias());
          addOptionalBias(values, valueOffset, layerWeights.vBias());

          // Per-head norms precede the rotation, matching llama.cpp's build_qkv -> norm -> rope.
          for (int head = 0; head < config.numHeads(); head++) {
            int offset = queryOffset + head * keyLength;
            normalizeHead(queries, offset, layerWeights.qNorm(), keyLength);
            if (rope) {
              rotary.applyBatch(queries, offset, position, neox);
            }
          }
          for (int head = 0; head < config.numKvHeads(); head++) {
            int offset = keyOffset + head * keyLength;
            normalizeHead(keys, offset, layerWeights.kNorm(), keyLength);
            if (rope) {
              rotary.applyBatch(keys, offset, position, neox);
            }
          }
        });
  }

  /**
   * Runs attention and the feed-forward block for every position of one layer.
   *
   * <p>Safe to fuse across positions: attention reads only the keys and values this layer already
   * projected, and each position writes only its own hidden state.
   */
  private void applyAttentionAndFeedForward(
      LlamaWeights.LayerWeights layerWeights, int layer, int sequenceLength) {
    int dim = config.embeddingDim();
    int keyLength = config.keyLength();
    int valueLength = config.valueLength();
    int queryDim = config.queryDim();
    int keyDim = config.keyDim();
    int valueDim = config.valueDim();
    int numHeads = config.numHeads();
    int groupSize = numHeads / config.numKvHeads();
    float scale = (float) (1.0 / Math.sqrt(keyLength));
    int lastPosition = sequenceLength - 1;

    forEachPosition(
        sequenceLength,
        (position, scratch) -> {
          float[] attentionOut = scratch.attentionOut;
          float[] scores = scratch.scores(sequenceLength);
          java.util.Arrays.fill(attentionOut, 0.0f);
          int firstVisible = config.attentionStartPosition(layer, position);
          int lastVisible = config.attentionEndPosition(layer, position, lastPosition);
          int visibleCount = lastVisible - firstVisible + 1;

          for (int head = 0; head < numHeads; head++) {
            int kvHead = head / groupSize;
            int queryOffset = position * queryDim + head * keyLength;

            VectorUtil.batchDotProductExact(
                queries,
                queryOffset,
                keys,
                firstVisible * keyDim + kvHead * keyLength,
                keyDim,
                visibleCount,
                keyLength,
                scores,
                firstVisible);
            for (int visible = firstVisible; visible <= lastVisible; visible++) {
              scores[visible] *= scale;
            }

            TensorOps.softmax(scores, firstVisible, visibleCount);

            VectorUtil.addWeightedRowsInPlace(
                attentionOut,
                head * valueLength,
                values,
                firstVisible * valueDim + kvHead * valueLength,
                valueDim,
                scores,
                firstVisible,
                visibleCount,
                valueLength);
          }

          float[] attentionProjected = scratch.attentionProjected;
          matmul(
              scratch,
              attentionProjected,
              0,
              attentionOut,
              layerWeights.wo(),
              layerWeights.woType(),
              dim,
              config.attentionOutputDim());
          normalizeProjection(attentionProjected, layerWeights.attentionPostNorm(), dim);
          int base = position * dim;
          for (int index = 0; index < dim; index++) {
            hidden[base + index] += attentionProjected[index];
          }

          float[] normed = scratch.normed;
          TensorOps.rmsNorm(
              normed, 0, hidden, base, layerWeights.ffnNorm(), dim, config.rmsNormEps());
          matmul(
              scratch,
              scratch.ffnGate,
              0,
              normed,
              layerWeights.ffnGate(),
              layerWeights.ffnGateType(),
              config.hiddenDim(),
              dim);
          matmul(
              scratch,
              scratch.ffnUp,
              0,
              normed,
              layerWeights.ffnUp(),
              layerWeights.ffnUpType(),
              config.hiddenDim(),
              dim);
          if (config.usesGeluFfn()) {
            TensorOps.geluGlu(
                scratch.ffnActivated, scratch.ffnGate, scratch.ffnUp, config.hiddenDim());
          } else {
            TensorOps.swiGlu(
                scratch.ffnActivated, scratch.ffnGate, scratch.ffnUp, config.hiddenDim());
          }
          float[] ffnProjected = scratch.ffnProjected;
          matmul(
              scratch,
              ffnProjected,
              0,
              scratch.ffnActivated,
              layerWeights.ffnDown(),
              layerWeights.ffnDownType(),
              dim,
              config.hiddenDim());
          normalizeProjection(ffnProjected, layerWeights.ffnPostNorm(), dim);
          for (int index = 0; index < dim; index++) {
            hidden[base + index] += ffnProjected[index];
          }
        });
  }

  /**
   * Reduces the encoded sequence to one vector.
   *
   * <p>Order matters and is llama.cpp's: the final norm applies per position, the mean is taken
   * over those normalized states, and only then does the dense head run. Projecting before pooling
   * would be a different function entirely.
   */
  private float[] poolAndProject(int sequenceLength) {
    int dim = config.embeddingDim();
    java.util.Arrays.fill(pooledSum, 0.0);
    for (int position = 0; position < sequenceLength; position++) {
      TensorOps.rmsNorm(
          workers[0].normed,
          0,
          hidden,
          position * dim,
          weights.outputNormWeight(),
          dim,
          config.rmsNormEps());
      for (int index = 0; index < dim; index++) {
        pooledSum[index] += workers[0].normed[index];
      }
    }
    for (int index = 0; index < dim; index++) {
      pooled[index] = (float) (pooledSum[index] / sequenceLength);
    }
    if (denseHead != null) {
      denseHead.project(pooled);
    }
    return pooled.clone();
  }

  private void ensureCapacity(int sequenceLength) {
    if (capacity >= sequenceLength) {
      return;
    }
    hidden = new float[Math.multiplyExact(sequenceLength, config.embeddingDim())];
    queries = new float[Math.multiplyExact(sequenceLength, config.queryDim())];
    keys = new float[Math.multiplyExact(sequenceLength, config.keyDim())];
    values = new float[Math.multiplyExact(sequenceLength, config.valueDim())];
    // Scores are per-worker rather than shared: each position writes the whole array while
    // scanning its own visible window.
    capacity = sequenceLength;
  }

  private void matmul(
      Scratch scratch,
      float[] out,
      int outOffset,
      float[] input,
      MemorySegment weight,
      GgufTensorType type,
      int rows,
      int cols) {
    // The kernels write from index zero, so a projection landing inside a per-position buffer goes
    // through a scratch row first.
    float[] destination = outOffset == 0 ? out : scratch.projection(rows);
    TensorOps.ggufMatmul(
        destination,
        input,
        weight,
        type,
        rows,
        cols,
        scratch.quantizedActivation,
        scratch.quantizedActivationScales,
        scratch.quantizedActivationZeroPointCorrections,
        scratch.quantizedActivationSums,
        GgufQ4Kernel.WIDENED);
    if (destination != out) {
      System.arraycopy(destination, 0, out, outOffset, rows);
    }
  }

  private void normalizeHead(float[] vector, int offset, float[] weight, int headDim) {
    if (weight.length != 0) {
      TensorOps.rmsNorm(vector, offset, vector, offset, weight, headDim, config.rmsNormEps());
    }
  }

  private void normalizeProjection(float[] projection, float[] norm, int size) {
    if (norm.length != 0) {
      TensorOps.rmsNorm(projection, projection, norm, size, config.rmsNormEps());
    }
  }

  private static void addOptionalBias(float[] vector, int offset, float[] bias) {
    for (int index = 0; index < bias.length; index++) {
      vector[offset + index] += bias[index];
    }
  }
}
