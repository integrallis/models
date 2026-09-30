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
package com.integrallis.models.backend.purejava.gemma3n;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionSpan;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionView;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Objects;

/**
 * Stateful pure-Java graph for Gemma 3n.
 *
 * <p>Single token at a time. Batched prefill is refused rather than approximated: AltUp's per-token
 * router decides the mixing coefficients for every stream, so a batch of rows does not share one
 * coefficient set, and the per-layer input injection is per token as well. That is the same
 * containment the Gemma 4 E-series and the routed Qwen3.5 use, and for the same reason -- a batched
 * path that quietly used one token's coefficients for a whole batch would produce fluent, wrong
 * text.
 *
 * <p>The residual stream is {@code altupInputs} parallel streams in one flat array. See {@link
 * Gemma3nAltUp} for the arithmetic and for the traps in it.
 */
public final class Gemma3nForwardPass {

  private final Gemma3nConfig config;
  private final Gemma3nWeights weights;
  private final LayeredKvCache cache;
  private final RotaryTable rotary;

  private final float[] streams;
  private final float[] predictions;
  private final float[] projectedStreams;
  private final float[] perLayerInputs;
  private final float[] perLayerProjected;
  private final float[] normalized;
  private final float[] active;
  private final float[] laurelLow;
  private final float[] laurelOut;
  private final float[] query;
  private final float[] key;
  private final float[] value;
  private final float[] attended;
  private final float[] attentionOut;
  private final float[] scores;
  private final float[] routerInput;
  private final float[] modalities;
  private final float[] predictCoefficients;
  private final float[] correctCoefficients;
  private final float[] ffnGate;
  private final float[] ffnUp;
  private final float[] ffnOut;
  private final float[] attnLaurel;
  private final float[] gated;
  private final float[] perLayerGate;
  private final float[] firstPrediction;
  private final float[] merged;
  private final float[] projectionInput;
  private final float[] projectionOutput;
  private final float[] logits;

  private int nextPosition;

  Gemma3nForwardPass(Gemma3nConfig config, Gemma3nWeights weights, LayeredKvCache cache) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    this.cache = Objects.requireNonNull(cache, "cache");
    // One table: the published files carry no separate sliding rope base, so every layer shares it.
    this.rotary = new RotaryTable(config.headDim(), config.ropeTheta(), 1.0f);

    int dim = config.embeddingDim();
    int altup = config.altupInputs();
    this.streams = new float[altup * dim];
    this.predictions = new float[altup * dim];
    this.projectedStreams = new float[(altup - 1) * dim];
    this.perLayerInputs = new float[config.perLayerTotalDim()];
    this.perLayerProjected = new float[config.perLayerTotalDim()];
    this.normalized = new float[dim];
    this.active = new float[dim];
    this.laurelLow = new float[weights.laurelRank()];
    this.laurelOut = new float[dim];
    this.query = new float[config.queryDim()];
    this.key = new float[config.keyDim()];
    this.value = new float[config.keyDim()];
    this.attended = new float[config.queryDim()];
    this.attentionOut = new float[dim];
    // Sized by the WIDEST layer, not by layer 0. A sliding layer's ring holds only its window, so
    // taking layer 0's capacity underflows the moment a full-attention layer attends further back
    // than the window -- which is at the first token past it.
    // physicalSequenceCapacity, not allocatedSequenceCapacity: the latter is what is allocated SO
    // FAR
    // and is zero on a fresh cache, which sized this buffer to nothing. And the widest layer, not
    // layer 0: a sliding layer's ring holds only its window, so a full-attention layer attending
    // further back than the window would overflow a buffer sized from the sliding one.
    int widestSpan = 0;
    for (int layer = 0; layer < config.kvOwningLayers(); layer++) {
      widestSpan = Math.max(widestSpan, cache.physicalSequenceCapacity(layer));
    }
    this.scores = new float[widestSpan];
    this.routerInput = new float[dim];
    this.modalities = new float[altup];
    this.predictCoefficients = new float[altup * altup];
    this.correctCoefficients = new float[altup];
    this.ffnGate = new float[config.hiddenDim()];
    this.ffnUp = new float[config.hiddenDim()];
    this.ffnOut = new float[dim];
    this.attnLaurel = new float[dim];
    this.gated = new float[dim];
    this.perLayerGate = new float[config.perLayerEmbeddingDim()];
    this.firstPrediction = new float[dim];
    this.merged = new float[dim];
    // Scratch for the offset-to-offset stream projections, which are all square in the model width.
    this.projectionInput = new float[dim];
    this.projectionOutput = new float[dim];
    this.logits = new float[config.vocabSize()];
  }

  /** Loads a decoder from a parsed GGUF. */
  public static Gemma3nForwardPass fromGgufFile(
      GgufFile file, Gemma3nConfig config, int runtimeContextLength) {
    return new Gemma3nForwardPass(
        config,
        Gemma3nWeights.fromGgufFile(file, config),
        Gemma3nKvCache.create(config, runtimeContextLength, 1));
  }

  public int nextPosition() {
    return nextPosition;
  }

  public void reset() {
    nextPosition = 0;
    cache.clear();
  }

  /**
   * Advances one token and returns the vocabulary logits.
   *
   * @param token the token id
   * @param position the position, which must be the next one in sequence
   * @return the logits. <b>This is the graph's own buffer</b>, valid only until the next call:
   *     clone it to keep it. Copying a vocabulary-sized array per token would allocate once per
   *     generated token for no benefit, which is the contract every decoder in this backend uses.
   */
  public float[] forward(int token, int position) {
    if (position != nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + nextPosition + ", got " + position);
    }
    int dim = config.embeddingDim();
    int altup = config.altupInputs();
    int active0 = config.altupActiveIndex();

    initialiseStreams(token, dim, altup, active0);
    preparePerLayerInputs(token);

    for (int layer = 0; layer < config.numLayers(); layer++) {
      Gemma3nWeights.LayerWeights weight = weights.layer(layer);

      // 1. predict: every stream gains a learned mixture of all of them.
      routerModalities(streams, active0 * dim, weight);
      project(weight.altupPredictCoefficients(), modalities, predictCoefficients);
      Gemma3nAltUp.predict(predictions, streams, predictCoefficients, altup, dim);
      System.arraycopy(predictions, active0 * dim, active, 0, dim);

      // 2. attention and LAuReL, both over the NORMED active prediction.
      TensorOps.rmsNorm(normalized, active, weight.attentionNorm(), dim, config.rmsNormEpsilon());
      laurel(weight, dim);
      attention(layer, weight, position, dim);

      TensorOps.rmsNorm(
          attentionOut, attentionOut, weight.postAttentionNorm(), dim, config.rmsNormEpsilon());
      for (int index = 0; index < dim; index++) {
        // The residual is the active PREDICTION, not the normalised copy of it.
        attnLaurel[index] =
            (attentionOut[index] + active[index] + laurelOut[index]) * INVERSE_ROOT_TWO;
      }

      // 3. feed-forward, gated by GELU with optional activation sparsity before it.
      feedForward(layer, weight, dim);
      for (int index = 0; index < dim; index++) {
        gated[index] = ffnOut[index] + attnLaurel[index];
      }

      // 4. correct: feed the difference from the prediction back into every stream.
      routerModalities(gated, 0, weight);
      project(weight.altupCorrectCoefficients(), modalities, correctCoefficients);
      Gemma3nAltUp.correct(streams, predictions, gated, correctCoefficients, altup, dim, active0);

      // 5. the per-layer input, added to the inactive streams only.
      perLayerContribution(layer, weight, dim, active0);
      Gemma3nAltUp.addToInactiveStreams(streams, firstPrediction, altup, dim, active0);
    }

    mergeStreams(dim, altup, active0);
    TensorOps.rmsNorm(merged, merged, weights.outputNorm(), dim, config.rmsNormEpsilon());
    project(weights.output(), merged, logits);
    // The bounded final-logit transform, which this decoder shipped without. Monotonic, so it
    // cannot
    // change which token a greedy decode picks -- but it decides every logit's value, so a
    // temperature,
    // a probability or a logprob taken from these is wrong without it. Verified against the
    // reference's
    // own graph, whose last three nodes are SCALE, TANH, SCALE around the output projection.
    TensorOps.softcap(logits, config.finalLogitSoftcap());
    nextPosition++;
    return logits;
  }

  private static final float INVERSE_ROOT_TWO = (float) (1.0 / Math.sqrt(2.0));

  /**
   * Builds the parallel streams from the token embedding.
   *
   * <p>The active stream is the scaled embedding; each other stream is a projection of it, rescaled
   * to the active stream's magnitude so no stream starts out dominating the mixture.
   */
  private void initialiseStreams(int token, int dim, int altup, int active0) {
    weights.embedToken(token, active);
    float scale = (float) Math.sqrt(dim);
    for (int index = 0; index < dim; index++) {
      active[index] *= scale;
    }
    System.arraycopy(active, 0, streams, active0 * dim, dim);
    float target = Gemma3nAltUp.magnitude(active, 0, dim);

    int slice = 0;
    for (int stream = 0; stream < altup; stream++) {
      if (stream == active0) {
        continue;
      }
      int offset = stream * dim;
      projectInto(weights.altupProjection(slice++), active, streams, offset);
      Gemma3nAltUp.rescaleToMagnitude(streams, offset, dim, target);
    }
  }

  /**
   * The second embedding table's contribution, one slice per layer.
   *
   * <p>Two scales, both from the reference: the projection is divided by {@code sqrt(n_embd)}
   * before the norm, and the sum of projection and table slice by {@code sqrt(2)} after it.
   *
   * <p>The first of those is <b>very nearly inert, and no test here covers it</b>. An RMS norm is
   * scale-invariant, so dividing before one cancels except through the epsilon: deleting the scale
   * moves the logits by less than the reference comparison's tolerance, which was confirmed by
   * deleting it. It is kept because the reference has it and because the epsilon makes it not
   * exactly nothing -- not because anything here would notice its absence.
   */
  private void preparePerLayerInputs(int token) {
    weights.embedPerLayerToken(token, perLayerInputs);
    project(weights.perLayerModelProjection(), active, perLayerProjected);
    float scale = (float) (1.0 / Math.sqrt(config.embeddingDim()));
    int width = config.perLayerEmbeddingDim();
    for (int layer = 0; layer < config.numLayers(); layer++) {
      int offset = layer * width;
      for (int index = 0; index < width; index++) {
        perLayerProjected[offset + index] *= scale;
      }
      // The norm runs over each layer's slice, not across the whole row.
      TensorOps.rmsNorm(
          perLayerProjected,
          offset,
          perLayerProjected,
          offset,
          weights.perLayerProjectionNorm(),
          width,
          config.rmsNormEpsilon());
      for (int index = 0; index < width; index++) {
        perLayerInputs[offset + index] =
            (perLayerProjected[offset + index] + perLayerInputs[offset + index]) * INVERSE_ROOT_TWO;
      }
    }
  }

  /**
   * The AltUp router: a normalised, scaled projection of one stream, squashed by tanh.
   *
   * <p>The scale is <b>1/n_embd</b>, not 1/sqrt(n_embd). The reference calls it {@code
   * router_input_scale}; getting it wrong changes how hard tanh saturates and therefore how
   * strongly the streams mix, which is a quiet behavioural change rather than an error.
   */
  private void routerModalities(float[] source, int offset, Gemma3nWeights.LayerWeights weight) {
    int dim = config.embeddingDim();
    TensorOps.rmsNorm(
        routerInput, 0, source, offset, weight.altupRouterNorm(), dim, config.rmsNormEpsilon());
    float scale = 1.0f / dim;
    for (int index = 0; index < dim; index++) {
      routerInput[index] *= scale;
    }
    project(weight.altupRouter(), routerInput, modalities);
    Gemma3nAltUp.tanhInPlace(modalities, config.altupInputs());
  }

  /** LAuReL: a rank-limited learned residual around the normalised input. */
  private void laurel(Gemma3nWeights.LayerWeights weight, int dim) {
    project(weight.laurelLeft(), normalized, laurelLow);
    project(weight.laurelRight(), laurelLow, laurelOut);
    TensorOps.rmsNorm(laurelOut, laurelOut, weight.laurelPostNorm(), dim, config.rmsNormEpsilon());
    for (int index = 0; index < dim; index++) {
      laurelOut[index] += normalized[index];
    }
  }

  private void attention(int layer, Gemma3nWeights.LayerWeights weight, int position, int dim) {
    int headDim = config.headDim();
    int heads = config.numHeads();
    int kvHeads = config.numKvHeads();
    int groupSize = heads / kvHeads;

    project(weight.queryProjection(), normalized, query);
    rotary.prepare(position);
    for (int head = 0; head < heads; head++) {
      TensorOps.rmsNorm(
          query,
          head * headDim,
          query,
          head * headDim,
          weight.queryNorm(),
          headDim,
          config.rmsNormEpsilon());
      rotary.apply(query, head * headDim, true);
    }

    int kvLayer = config.kvSourceLayer(layer);
    if (config.ownsKvCache(layer)) {
      project(weight.keyProjection(), normalized, key);
      project(weight.valueProjection(), normalized, value);
      for (int head = 0; head < kvHeads; head++) {
        TensorOps.rmsNorm(
            key,
            head * headDim,
            key,
            head * headDim,
            weight.keyNorm(),
            headDim,
            config.rmsNormEpsilon());
        rotary.apply(key, head * headDim, true);
        // The value norm carries NO weight vector: a plain RMS normalisation, which is what the
        // reference's bare ggml_rms_norm does. Passing the key norm here instead would be
        // invisible.
        normalizeWithoutWeight(value, head * headDim, headDim);
      }
      cache.store(kvLayer, position, key, value);
    }

    int from = firstAttendedPosition(kvLayer, position);
    AttentionView view = cache.attentionView(kvLayer, from, position + 1);
    float[] keys = cache.keyBuffer(kvLayer);
    float[] values = cache.valueBuffer(kvLayer);
    int keyDim = config.keyDim();
    Arrays.fill(attended, 0, config.queryDim(), 0.0f);

    for (int head = 0; head < heads; head++) {
      int kvHead = head / groupSize;
      int count = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        for (int row = 0; row < span.positionCount(); row++) {
          int keyOffset = span.keyOffset() + row * keyDim + kvHead * headDim;
          scores[count++] =
              VectorUtil.dotProduct(query, head * headDim, keys, keyOffset, headDim)
                  * Gemma3nConfig.ATTENTION_SCALE;
        }
      }
      TensorOps.softmax(scores, 0, count);
      int score = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        for (int row = 0; row < span.positionCount(); row++) {
          int valueOffset = span.valueOffset() + row * keyDim + kvHead * headDim;
          VectorUtil.addScaledInPlace(
              attended, head * headDim, values, valueOffset, headDim, scores[score++]);
        }
      }
    }
    project(weight.attentionOutput(), attended, attentionOut);
  }

  /** A sliding layer sees only its window; a full-attention layer sees everything so far. */
  private int firstAttendedPosition(int kvLayer, int position) {
    if (!config.usesSlidingWindow(kvLayer)) {
      return 0;
    }
    return Math.max(0, position - config.slidingWindow() + 1);
  }

  private void feedForward(int layer, Gemma3nWeights.LayerWeights weight, int dim) {
    TensorOps.rmsNorm(normalized, attnLaurel, weight.ffnNorm(), dim, config.rmsNormEpsilon());
    project(weight.ffnGate(), normalized, ffnGate);
    project(weight.ffnUp(), normalized, ffnUp);
    if (config.usesActivationSparsity(layer)) {
      // Before the GELU, not after.
      Gemma3nAltUp.gaussianTopk(ffnGate, config.hiddenDim(), config.activationSparsityScale(layer));
    }
    TensorOps.gelu(ffnGate, 0, ffnGate, 0, config.hiddenDim());
    for (int index = 0; index < config.hiddenDim(); index++) {
      ffnGate[index] *= ffnUp[index];
    }
    project(weight.ffnDown(), ffnGate, ffnOut);
    TensorOps.rmsNorm(ffnOut, ffnOut, weight.postFfnNorm(), dim, config.rmsNormEpsilon());
  }

  /** The active stream's corrected value, gated and projected into this layer's input slice. */
  private void perLayerContribution(
      int layer, Gemma3nWeights.LayerWeights weight, int dim, int active0) {
    int width = config.perLayerEmbeddingDim();
    for (int index = 0; index < dim; index++) {
      firstPrediction[index] = streams[active0 * dim + index] * weight.altupCorrectScale()[index];
    }
    project(weight.perLayerInputGate(), firstPrediction, perLayerGate);
    TensorOps.gelu(perLayerGate, 0, perLayerGate, 0, width);
    int offset = layer * width;
    for (int index = 0; index < width; index++) {
      perLayerGate[index] *= perLayerInputs[offset + index];
    }
    project(weight.perLayerProjection(), perLayerGate, firstPrediction);
    TensorOps.rmsNorm(
        firstPrediction, firstPrediction, weight.perLayerPostNorm(), dim, config.rmsNormEpsilon());
  }

  /** Collapses the streams to one: a mean, with the inactive ones projected and rescaled first. */
  private void mergeStreams(int dim, int altup, int active0) {
    float target = Gemma3nAltUp.magnitude(streams, active0 * dim, dim);
    int slice = 0;
    for (int stream = 0; stream < altup; stream++) {
      if (stream == active0) {
        continue;
      }
      int destination = slice * dim;
      projectInto(
          weights.altupUnembedProjection(slice),
          streams,
          stream * dim,
          projectedStreams,
          destination);
      Gemma3nAltUp.rescaleToMagnitude(projectedStreams, destination, dim, target);
      slice++;
    }
    System.arraycopy(streams, active0 * dim, active, 0, dim);
    Gemma3nAltUp.merge(merged, active, projectedStreams, altup, dim);
  }

  private static void normalizeWithoutWeight(float[] values, int offset, int length) {
    double total = 0.0;
    for (int index = offset; index < offset + length; index++) {
      double value = values[index];
      total += value * value;
    }
    float inverse = (float) (1.0 / Math.sqrt(total / length + 1.0e-6));
    for (int index = offset; index < offset + length; index++) {
      values[index] *= inverse;
    }
  }

  private void project(Gemma3nWeights.Matrix matrix, float[] input, float[] output) {
    TensorOps.ggufMatmul(
        output, input, matrix.data(), matrix.type(), matrix.rows(), matrix.columns());
  }

  private void projectInto(
      Gemma3nWeights.Matrix matrix, float[] input, float[] output, int outputOffset) {
    projectInto(matrix, input, 0, output, outputOffset);
  }

  /**
   * Projects a slice of one array into a slice of another.
   *
   * <p>{@code ggufMatmul} works from offset zero on both sides, and the stream projections are
   * offset on both -- so the slice is copied through two reused buffers rather than the
   * freshly-allocated ones this first had. Six allocations of the model width per token, per
   * layer-free step, is not something to leave in a decode loop even though it was correct.
   */
  private void projectInto(
      Gemma3nWeights.Matrix matrix,
      float[] input,
      int inputOffset,
      float[] output,
      int outputOffset) {
    float[] source = input;
    if (inputOffset != 0) {
      System.arraycopy(input, inputOffset, projectionInput, 0, matrix.columns());
      source = projectionInput;
    }
    boolean offsetOutput = outputOffset != 0;
    float[] destination = offsetOutput ? projectionOutput : output;
    TensorOps.ggufMatmul(
        destination, source, matrix.data(), matrix.type(), matrix.rows(), matrix.columns());
    if (offsetOutput) {
      System.arraycopy(destination, 0, output, outputOffset, matrix.rows());
    }
  }
}
