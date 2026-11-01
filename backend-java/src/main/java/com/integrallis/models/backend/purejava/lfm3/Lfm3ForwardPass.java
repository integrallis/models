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
package com.integrallis.models.backend.purejava.lfm3;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.VectorUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;

/** Layer-major chunked LFM3 inference with bounded local attention and convolution history. */
public final class Lfm3ForwardPass {
  private static final int BATCH_CAPACITY = 32;
  private final Lfm3Config config;
  private final Lfm3Weights weights;
  private final GgufBatchedMatrixKernel kernel;
  private final RotaryTable globalRotary, slidingRotary;
  private final LayeredKvCache cache;
  private final int[] cacheLayer;
  private final float[][] convState;
  private final int maxSeqLen;
  private int nextPosition;
  java.util.function.BiConsumer<String, float[]> trace;

  private void trace(String name, float[] data, int n) {
    if (trace != null) trace.accept(name, Arrays.copyOf(data, n));
  }

  private final float[] state,
      normalized,
      projected,
      query,
      key,
      value,
      attentionOut,
      headGate,
      gate,
      up,
      ffn,
      convProjected,
      convOutput,
      scores,
      logits;
  private final float[] tokenState, exactWeightRow, q4LaneScratch;
  private final byte[] quantized;
  private final float[] scales;
  private final int[] corrections;
  private final short[] sums;

  public static Lfm3ForwardPass fromGgufFile(
      GgufFile file, Lfm3Config config, int context, GgufBatchedMatrixKernel kernel) {
    return new Lfm3ForwardPass(config, Lfm3Weights.fromGgufFile(file, config), context, kernel);
  }

  Lfm3ForwardPass(
      Lfm3Config config, Lfm3Weights weights, int context, GgufBatchedMatrixKernel kernel) {
    this.config = Objects.requireNonNull(config);
    this.weights = Objects.requireNonNull(weights);
    this.kernel = kernel;
    if (context < 1 || context > config.contextLength())
      throw new IllegalArgumentException("invalid LFM3 runtime context");
    maxSeqLen = context;
    if (kernel != null) kernel.requireSequentialQ4KReduction();
    // GGUF contains full-head frequency divisors, including the unrotated tail. Reducing
    // rotaryDim would change both the frequency progression and the NeoX pair layout.
    globalRotary = new RotaryTable(config.headDim(), config.ropeTheta(), 1f, weights.ropeFactors());
    slidingRotary = new RotaryTable(config.headDim(), config.slidingRopeTheta(), 1f);
    int d = config.embeddingDim(),
        q = config.queryDim(),
        k = 0,
        h = config.hiddenDim(),
        b = BATCH_CAPACITY;
    cacheLayer = new int[config.numLayers()];
    Arrays.fill(cacheLayer, -1);
    convState = new float[config.numLayers()][];
    var specs = new ArrayList<LayerSpec>();
    for (int l = 0; l < config.numLayers(); l++) {
      if (config.usesAttention(l)) {
        int kd = config.keyDim(l);
        k = Math.max(k, kd);
        cacheLayer[l] = specs.size();
        specs.add(
            config.usesSlidingWindow(l)
                ? LayerSpec.ring(kd, kd, config.slidingWindow())
                : LayerSpec.linear(kd, kd));
      } else convState[l] = new float[config.shortConvStateSize()];
    }
    cache = new LayeredKvCache(context, specs.toArray(LayerSpec[]::new));
    state = new float[b * d];
    normalized = new float[b * d];
    projected = new float[b * d];
    query = new float[b * q];
    key = new float[b * k];
    value = new float[b * k];
    attentionOut = new float[b * q];
    headGate = new float[b * config.numHeads()];
    gate = new float[b * h];
    up = new float[b * h];
    ffn = new float[b * h];
    convProjected = new float[b * 3 * d];
    convOutput = new float[b * d];
    scores = new float[context];
    logits = new float[config.vocabSize()];
    tokenState = new float[d];
    int widest = Math.max(Math.max(d, h), q);
    exactWeightRow = new float[widest];
    quantized = new byte[b * widest];
    scales = new float[b * ((widest + 31) / 32)];
    corrections = new int[b * ((widest + 3) / 4)];
    sums = new short[b * ((widest + 15) / 16)];
    q4LaneScratch =
        new float
            [Math.max(
                    BATCH_CAPACITY * Math.max(1, weights.maxQ40ProjectionRows()),
                    config.vocabSize())
                * 8];
  }

  public int nextPosition() {
    return nextPosition;
  }

  public int maxBatchSize() {
    return BATCH_CAPACITY;
  }

  public long allocatedCacheBytes() {
    return cache.allocatedBytes();
  }

  public void reset() {
    nextPosition = 0;
    cache.clear();
    for (float[] s : convState) if (s != null) Arrays.fill(s, 0);
  }

  public float[] forward(int token, int position) {
    return process(new int[] {token}, position, true);
  }

  public float[] hiddenState(int token, int position) {
    return process(new int[] {token}, position, false);
  }

  public float[] prefill(int[] tokens, int startPosition) {
    return process(tokens, startPosition, true);
  }

  public float[] prefillHiddenState(int[] tokens, int startPosition) {
    return process(tokens, startPosition, false);
  }

  private float[] process(int[] tokens, int startPosition, boolean outputLogits) {
    Objects.requireNonNull(tokens);
    if (tokens.length == 0
        || startPosition != nextPosition
        || (long) startPosition + tokens.length > maxSeqLen)
      throw new IllegalArgumentException("invalid LFM3 token range");
    for (int token : tokens)
      if (token < 0 || token >= config.vocabSize())
        throw new IllegalArgumentException("token out of range: " + token);
    for (int from = 0; from < tokens.length; from += BATCH_CAPACITY) {
      int n = Math.min(BATCH_CAPACITY, tokens.length - from),
          pos = startPosition + from,
          d = config.embeddingDim();
      for (int t = 0; t < n; t++) {
        weights.embedToken(tokens[from + t], tokenState);
        System.arraycopy(tokenState, 0, state, t * d, d);
      }
      trace("model.embed_tokens", state, n * d);
      for (int layer = 0; layer < config.numLayers(); layer++) {
        var w = weights.layer(layer);
        norm(normalized, state, w.attentionNorm(), n);
        trace("model.layers." + layer + ".operator_norm", normalized, n * d);
        if (config.usesAttention(layer)) attend(layer, w, n, pos);
        else convolve(layer, w, n);
        if (w.postAttentionNorm() != null) norm(projected, projected, w.postAttentionNorm(), n);
        trace("model.layers." + layer + ".post_operator_norm", projected, n * d);
        add(state, projected, n * d);
        norm(normalized, state, w.ffnNorm(), n);
        project(w.gateProjection(), normalized, gate, n);
        project(w.upProjection(), normalized, up, n);
        if (kernel == null || !kernel.trySwiGlu(ffn, 0, gate, 0, up, 0, n * config.hiddenDim()))
          TensorOps.swiGlu(ffn, gate, up, n * config.hiddenDim());
        project(w.downProjection(), ffn, projected, n);
        if (w.postFfnNorm() != null) norm(projected, projected, w.postFfnNorm(), n);
        trace("model.layers." + layer + ".post_ffn_norm", projected, n * d);
        add(state, projected, n * d);
        trace("l_out-" + layer, state, n * d);
      }
      TensorOps.rmsNorm(
          tokenState, 0, state, (n - 1) * d, weights.outputNorm(), d, config.rmsNormEpsilon());
      nextPosition = pos + n;
    }
    if (!outputLogits) return tokenState;
    project(
        new Lfm3Weights.Matrix(
            weights.outputProjection(),
            weights.outputProjectionType(),
            config.vocabSize(),
            config.embeddingDim()),
        tokenState,
        logits,
        1);
    return logits;
  }

  private void norm(float[] out, float[] in, float[] norm, int n) {
    int d = config.embeddingDim();
    for (int t = 0; t < n; t++)
      TensorOps.rmsNorm(out, t * d, in, t * d, norm, d, config.rmsNormEpsilon());
  }

  private static void add(float[] a, float[] b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
  }

  private void convolve(int layer, Lfm3Weights.LayerWeights w, int n) {
    int d = config.embeddingDim();
    project(w.shortConvInProjection(), normalized, convProjected, n);
    trace("model.layers." + layer + ".conv.in_proj", convProjected, n * 3 * d);
    VectorUtil.gatedShortConv1d(
        convOutput,
        0,
        convProjected,
        0,
        n,
        d,
        w.shortConvKernel(),
        convState[layer],
        config.shortConvCache());
    project(w.shortConvOutProjection(), convOutput, projected, n);
  }

  private void attend(int layer, Lfm3Weights.LayerWeights w, int n, int start) {
    int hd = config.headDim(),
        heads = config.numHeads(),
        kvHeads = config.numKvHeads(layer),
        kd = config.keyDim(layer),
        qd = config.queryDim(),
        cl = cacheLayer[layer],
        group = heads / kvHeads;
    project(w.queryProjection(), normalized, query, n);
    project(w.keyProjection(), normalized, key, n);
    project(w.valueProjection(), normalized, value, n);
    project(w.attentionGate(), normalized, headGate, n);
    RotaryTable rotary = config.usesSlidingWindow(layer) ? slidingRotary : globalRotary;
    float scale = (float) (1.0 / Math.sqrt(hd));
    for (int t = 0; t < n; t++) {
      int position = start + t;
      rotary.prepare(position);
      for (int h = 0; h < heads; h++) {
        int off = t * qd + h * hd;
        TensorOps.rmsNorm(query, off, query, off, w.queryNorm(), hd, config.rmsNormEpsilon());
        rotary.apply(query, off, true);
      }
      for (int h = 0; h < kvHeads; h++) {
        int off = t * kd + h * hd;
        TensorOps.rmsNorm(key, off, key, off, w.keyNorm(), hd, config.rmsNormEpsilon());
        rotary.apply(key, off, true);
      }
      cache.store(cl, position, key, t * kd, value, t * kd);
      int from =
          config.usesSlidingWindow(layer) ? Math.max(0, position - config.slidingWindow() + 1) : 0;
      float[] keys = cache.keyBuffer(cl), values = cache.valueBuffer(cl);
      for (int h = 0; h < heads; h++) {
        int qoff = t * qd + h * hd, kvOff = (h / group) * hd;
        for (int past = from; past <= position; past++)
          scores[past - from] =
              PinnedReduction.dot(query, qoff, keys, cache.keyOffset(cl, past) + kvOff, hd) * scale;
        TensorOps.softmax(scores, 0, position - from + 1);
        Arrays.fill(attentionOut, qoff, qoff + hd, 0);
        for (int past = from; past <= position; past++) {
          float probability = scores[past - from];
          int off = cache.valueOffset(cl, past) + kvOff;
          for (int j = 0; j < hd; j++) attentionOut[qoff + j] += probability * values[off + j];
        }
        float g = (float) (1.0 / (1.0 + Math.exp(-headGate[t * heads + h])));
        for (int j = 0; j < hd; j++) attentionOut[qoff + j] *= g;
      }
    }
    project(w.attentionOutputProjection(), attentionOut, projected, n);
  }

  private void project(Lfm3Weights.Matrix w, float[] in, float[] out, int n) {
    if (kernel != null && kernel.isEligible(w.type(), n, w.rows(), w.columns())) {
      kernel.multiply(out, in, w.data(), w.type(), n, w.rows(), w.columns());
      return;
    }
    if (w.type() == GgufTensorType.F32) {
      TensorOps.ggufExactBatchedMatmul(
          out, in, w.data(), w.type(), n, w.rows(), w.columns(), exactWeightRow);
      return;
    }
    TensorOps.ggufBatchedMatmul(
        out,
        in,
        w.data(),
        w.type(),
        n,
        w.rows(),
        w.columns(),
        quantized,
        scales,
        corrections,
        sums,
        q4LaneScratch,
        GgufQ4Kernel.WIDENED);
  }
}
