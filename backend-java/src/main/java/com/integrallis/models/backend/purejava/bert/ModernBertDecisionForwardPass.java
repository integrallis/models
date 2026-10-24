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
package com.integrallis.models.backend.purejava.bert;

import com.integrallis.models.backend.purejava.bert.ModernBertConfig.QuestionType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.util.Objects;

/**
 * One forward pass of a {@code modern-bert} typed-decision model: tokens in, one logit per option.
 *
 * <p>The function this computes, from the reference implementation rather than from llama.cpp:
 *
 * <pre>
 *   h = encoder(ids, mask = option isolation + sliding window, position_ids = parallel layout)
 *   h = h + token_types[qtype]
 *   h = decision_block(h) x decision.block_count
 *   logits = cls.output( GELU( cls( cls.norm( h[markers] ) ) ) )
 * </pre>
 *
 * <p><b>Why not llama.cpp for the head.</b> llama.cpp has a {@code modern-bert} encoder graph but
 * no decision head, and its reranker path applies the same {@code cls.*} tensors in a different
 * order — dense, GELU, norm — than the order this checkpoint trained them in. An implementation
 * following that order would load every tensor, satisfy every shape assertion, and compute a
 * different function. The scorer here is {@code LayerNorm -> Linear -> GELU -> Linear}, which is
 * what the reference's {@code nn.Sequential} says.
 *
 * <p><b>Three places a reimplementation goes wrong, each pinned by a test.</b>
 *
 * <ul>
 *   <li>The GeGLU halves. {@code ffn_up} is twice the declared width, and the <em>first</em> half
 *       takes the activation while the second multiplies it. Swapping them still runs.
 *   <li>The sliding window is a distance in <b>token indices</b>, not in position ids. Under the
 *       parallel layout many tokens share a position id, so a window computed on positions would
 *       let options see each other and make the option isolation useless.
 *   <li>Block 0 has no {@code attn_norm}: the embedding norm feeds its attention directly. Treating
 *       a missing norm as an identity is correct here only because that norm already ran.
 * </ul>
 */
final class ModernBertDecisionForwardPass {

  // TEMPORARY diagnostic switches, removed once the structural question is settled. They exist
  // because the residual against the reference is NOT quantization error -- BF16 tracks the
  // reference no better than Q8_0 -- so the remaining candidates have to be measured, not argued.
  static boolean SWAP_GEGLU_HALVES = false;
  static boolean ALL_LAYERS_GLOBAL = false;
  static boolean TANH_GELU = false;
  static boolean WINDOW_ON_TOKEN_INDEX = false;
  static boolean TYPE_AT_EMBEDDING = false;
  static boolean NORM_AFTER_DECISION = false;
  static boolean NO_OUTPUT_NORM = false;
  static int HALF_WINDOW_OVERRIDE = 0;
  static boolean ROPE_IN_DECISION_BLOCKS = false;
  static boolean ISOLATE_IN_DECISION_BLOCKS = false;
  static boolean SWAP_EMBED_AND_OUTPUT_NORM = false;

  /**
   * Set to collect a fingerprint of the hidden state after each stage.
   *
   * <p>An end-to-end logit comparison cannot say WHERE two implementations diverge, which is why
   * eleven structural hypotheses could be refuted without finding the cause of the residual. This
   * records the same per-stage statistics the reference's hooks record, so the two can be lined up
   * and the first disagreeing stage read off directly.
   */
  static java.util.List<String> TRACE = null;

  private final ModernBertConfig config;
  private final ModernBertWeights weights;
  private final float[] exactWeightRow;
  private final GgufBatchedMatrixKernel matrixKernel;
  private float[] zeroBias;

  ModernBertDecisionForwardPass(ModernBertConfig config, ModernBertWeights weights) {
    this(config, weights, GgufBatchedMatrixKernel.none());
  }

  ModernBertDecisionForwardPass(
      ModernBertConfig config, ModernBertWeights weights, GgufBatchedMatrixKernel matrixKernel) {
    this.matrixKernel = Objects.requireNonNull(matrixKernel, "matrixKernel");
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    int widest = config.embeddingDim();
    for (int layer = 0; layer < config.numLayers(); layer++) {
      int up = config.feedForwardLengthFor(layer) * (config.isDecisionLayer(layer) ? 1 : 2);
      widest = Math.max(widest, Math.max(up, 3 * config.embeddingDim()));
    }
    this.exactWeightRow = new float[widest];
  }

  /**
   * The raw per-option logits, before any temperature is applied.
   *
   * @param tokenIds the head prompt's token ids
   * @param layout the parallel option layout for those tokens
   * @param type the question type, which selects the added type embedding
   * @return one logit per option, in option order
   */
  float[] logits(int[] tokenIds, ModernBertDecisionLayout layout, QuestionType type) {
    Objects.requireNonNull(tokenIds, "tokenIds");
    Objects.requireNonNull(layout, "layout");
    Objects.requireNonNull(type, "type");
    int sequenceLength = tokenIds.length;
    // Read once: the layout's accessors hand back copies, so fetching them per layer would
    // allocate a fresh array for every block.
    int[] positionIds = layout.positionIds();
    if (positionIds.length != sequenceLength) {
      throw new IllegalArgumentException(
          "layout covers "
              + positionIds.length
              + " positions but there are "
              + sequenceLength
              + " tokens");
    }
    int dim = config.embeddingDim();
    float[] hidden = new float[sequenceLength * dim];

    // Embeddings, then the embedding norm. token_embd_norm carries a weight and no bias, unlike
    // classic BERT's.
    float[] row = new float[dim];
    for (int position = 0; position < sequenceLength; position++) {
      weights.tokenEmbedding(tokenIds[position], row);
      System.arraycopy(row, 0, hidden, position * dim, dim);
    }
    trace("embed", hidden);
    layerNorm(
        hidden,
        hidden,
        sequenceLength,
        SWAP_EMBED_AND_OUTPUT_NORM ? weights.outputNorm() : weights.tokenEmbeddingNorm(),
        null);
    trace("embed_norm", hidden);

    weights.questionTypeEmbedding(type.row(), row);
    if (TYPE_AT_EMBEDDING) {
      addType(hidden, sequenceLength, dim, row);
    }

    for (int layer = 0; layer < config.encoderLayers(); layer++) {
      block(hidden, sequenceLength, layer, layout, positionIds);
      trace("enc" + layer, hidden);
    }
    if (!NORM_AFTER_DECISION && !NO_OUTPUT_NORM) {
      layerNorm(
          hidden,
          hidden,
          sequenceLength,
          SWAP_EMBED_AND_OUTPUT_NORM ? weights.tokenEmbeddingNorm() : weights.outputNorm(),
          null);
    }

    // The question type enters once, added at every position, before the decision blocks.
    if (!TYPE_AT_EMBEDDING) {
      addType(hidden, sequenceLength, dim, row);
    }

    trace("plus_type", hidden);
    for (int layer = config.encoderLayers(); layer < config.numLayers(); layer++) {
      block(hidden, sequenceLength, layer, layout, positionIds);
      trace("dec" + layer, hidden);
    }
    if (NORM_AFTER_DECISION && !NO_OUTPUT_NORM) {
      layerNorm(hidden, hidden, sequenceLength, weights.outputNorm(), null);
    }

    return score(hidden, layout.markers());
  }

  /** One pre-norm block, encoder or decision; the config says which, never a null check. */
  private void block(
      float[] hidden,
      int sequenceLength,
      int layer,
      ModernBertDecisionLayout layout,
      int[] positionIds) {
    int dim = config.embeddingDim();
    ModernBertWeights.Layer weightsForLayer = weights.layer(layer);

    float[] normalized = new float[sequenceLength * dim];
    if (weightsForLayer.attentionNormWeight() == null) {
      // Block 0 only. The embedding norm has already run over exactly this tensor.
      System.arraycopy(hidden, 0, normalized, 0, hidden.length);
    } else {
      layerNorm(
          normalized,
          hidden,
          sequenceLength,
          weightsForLayer.attentionNormWeight(),
          weightsForLayer.attentionNormBias());
    }

    float[] attention =
        attention(normalized, sequenceLength, layer, layout, positionIds, weightsForLayer);
    float[] projected = new float[sequenceLength * dim];
    project(projected, attention, sequenceLength, weightsForLayer.attentionOutput());
    addBias(projected, sequenceLength, dim, weightsForLayer.attentionOutputBias());
    for (int index = 0; index < hidden.length; index++) {
      hidden[index] += projected[index];
    }

    float[] feedForwardInput = new float[sequenceLength * dim];
    layerNorm(
        feedForwardInput,
        hidden,
        sequenceLength,
        weightsForLayer.feedForwardNormWeight(),
        weightsForLayer.feedForwardNormBias());

    int intermediate = weightsForLayer.intermediateWidth();
    int upWidth = weightsForLayer.gated() ? 2 * intermediate : intermediate;
    float[] up = new float[sequenceLength * upWidth];
    project(up, feedForwardInput, sequenceLength, weightsForLayer.feedForwardUp());
    addBias(up, sequenceLength, upWidth, weightsForLayer.feedForwardUpBias());

    float[] activated = new float[sequenceLength * intermediate];
    for (int position = 0; position < sequenceLength; position++) {
      int source = position * upWidth;
      int target = position * intermediate;
      if (weightsForLayer.gated()) {
        // GeGLU: the FIRST half is activated, the second multiplies it. The reference's own
        // chunk(2) order; swapping these still produces a plausible-looking model.
        for (int element = 0; element < intermediate; element++) {
          activated[target + element] =
              SWAP_GEGLU_HALVES
                  ? gelu(up[source + intermediate + element]) * up[source + element]
                  : gelu(up[source + element]) * up[source + intermediate + element];
        }
      } else {
        // ReLU, NOT GELU, and read from the reference's source rather than assumed: laya builds
        // the decision blocks as `nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout,
        // batch_first=True, norm_first=True)` (laya/common.py:592) and does not pass `activation`,
        // so it keeps that class's default of `F.relu`. The scorer two lines later DOES ask for
        // `nn.GELU()`, which is why the two differ inside one model.
        for (int element = 0; element < intermediate; element++) {
          activated[target + element] = Math.max(0.0f, up[source + element]);
        }
      }
    }

    float[] down = new float[sequenceLength * dim];
    project(down, activated, sequenceLength, weightsForLayer.feedForwardDown());
    addBias(down, sequenceLength, dim, weightsForLayer.feedForwardDownBias());
    for (int index = 0; index < hidden.length; index++) {
      hidden[index] += down[index];
    }
  }

  private float[] attention(
      float[] normalized,
      int sequenceLength,
      int layer,
      ModernBertDecisionLayout layout,
      int[] positionIds,
      ModernBertWeights.Layer weightsForLayer) {
    int dim = config.embeddingDim();
    int heads = config.numHeads();
    int headDim = config.headDim();

    float[] fused = new float[sequenceLength * 3 * dim];
    project(fused, normalized, sequenceLength, weightsForLayer.queryKeyValue());
    addBias(fused, sequenceLength, 3 * dim, weightsForLayer.queryKeyValueBias());

    float theta = ALL_LAYERS_GLOBAL ? config.ropeTheta() : config.ropeThetaFor(layer);
    // The decision blocks get NO rotary embedding. laya replaces their `self_attn` with
    // `_DynamicMultiheadAttention`, documented in its own source as "Same parameters, same maths"
    // as `nn.MultiheadAttention` -- which is vanilla attention and knows nothing about rope. Only
    // the encoder's custom blocks carry it.
    boolean applyRope = !config.isDecisionLayer(layer) && !ROPE_IN_DECISION_BLOCKS;
    // Rotary on queries and keys, at the layout's position id rather than the token index: that
    // substitution is the whole point of the parallel layout.
    if (applyRope) {
      for (int position = 0; position < sequenceLength; position++) {
        int base = position * 3 * dim;
        for (int head = 0; head < heads; head++) {
          TensorOps.ropeNeox(fused, base + head * headDim, positionIds[position], headDim, theta);
          TensorOps.ropeNeox(
              fused, base + dim + head * headDim, positionIds[position], headDim, theta);
        }
      }
    }

    boolean global = ALL_LAYERS_GLOBAL || config.isGlobalAttentionLayer(layer);
    boolean isolationFree = config.isDecisionLayer(layer) && !ISOLATE_IN_DECISION_BLOCKS;
    int halfWindow = HALF_WINDOW_OVERRIDE > 0 ? HALF_WINDOW_OVERRIDE : config.slidingWindow() / 2;
    float scale = (float) (1.0 / Math.sqrt(headDim));
    float[] output = new float[sequenceLength * dim];
    float[] scores = new float[sequenceLength];

    for (int head = 0; head < heads; head++) {
      for (int query = 0; query < sequenceLength; query++) {
        int queryBase = query * 3 * dim + head * headDim;
        int attendable = 0;
        for (int key = 0; key < sequenceLength; key++) {
          // Option isolation applies to the ENCODER only. The reference's head loop passes
          // `src_key_padding_mask` and no `src_mask` (laya/common.py:625), so the decision blocks
          // attend across the whole sequence -- which is their job: they are what compares the
          // options. Its own comment says where the symmetry comes from instead: "The head below
          // has no positional encoding, so with order-blind marker embeddings the logits permute
          // with the options."
          if (!isolationFree && layout.isolates(query, key)) {
            scores[key] = Float.NEGATIVE_INFINITY;
            continue;
          }
          int distance =
              WINDOW_ON_TOKEN_INDEX
                  ? Math.abs(query - key)
                  : Math.abs(positionIds[query] - positionIds[key]);
          if (!global && distance > halfWindow) {
            // A distance in POSITION IDS, not token indices. Under the parallel layout the
            // position span stays short -- every case here tops out between 52 and 62 even at 130
            // tokens -- so a window of 64 masks nothing, and the windowed layers see the whole
            // sequence. On token indices it would instead mask more and more as options are added,
            // which is exactly the error-growing-with-option-count this started with.
            scores[key] = Float.NEGATIVE_INFINITY;
            continue;
          }
          int keyBase = key * 3 * dim + dim + head * headDim;
          float dot = 0.0f;
          for (int element = 0; element < headDim; element++) {
            dot += fused[queryBase + element] * fused[keyBase + element];
          }
          scores[key] = dot * scale;
          attendable++;
        }
        if (attendable == 0) {
          throw new IllegalStateException(
              "token " + query + " can attend to nothing in layer " + layer);
        }
        TensorOps.softmax(scores, 0, sequenceLength);
        int outputBase = query * dim + head * headDim;
        for (int key = 0; key < sequenceLength; key++) {
          float weight = scores[key];
          if (weight == 0.0f) {
            continue;
          }
          int valueBase = key * 3 * dim + 2 * dim + head * headDim;
          for (int element = 0; element < headDim; element++) {
            output[outputBase + element] += weight * fused[valueBase + element];
          }
        }
      }
    }
    return output;
  }

  /** {@code cls.output(GELU(cls(cls.norm(h[marker]))))}, one logit per option marker. */
  private float[] score(float[] hidden, int[] markers) {
    ModernBertWeights.Scorer scorer =
        weights
            .scorer()
            .orElseThrow(
                () ->
                    new IllegalStateException(
                        "this modern-bert artifact carries no cls.* scorer; it cannot decide"));
    int dim = config.embeddingDim();
    float[] logits = new float[markers.length];
    float[] marker = new float[dim];
    float[] normed = new float[dim];
    float[] dense = new float[dim];

    for (int option = 0; option < markers.length; option++) {
      System.arraycopy(hidden, markers[option] * dim, marker, 0, dim);
      TensorOps.layerNorm(
          normed,
          0,
          marker,
          0,
          scorer.normWeight(),
          scorer.normBias() == null ? zeroBias(dim) : scorer.normBias(),
          dim,
          config.layerNormEps());
      project(dense, normed, 1, scorer.dense());
      addBias(dense, 1, dim, scorer.denseBias());
      float logit = scorer.outputBias();
      for (int element = 0; element < dim; element++) {
        logit += gelu(dense[element]) * scorer.output()[element];
      }
      logits[option] = logit;
    }
    return logits;
  }

  /**
   * A row-wise layer norm that accepts an absent bias.
   *
   * <p>{@code token_embd_norm} and {@code output_norm} carry a weight and no bias in this
   * checkpoint, unlike classic BERT's, and unlike the encoder blocks' own norms. {@link
   * TensorOps#layerNorm} requires both, so the zero vector stands in rather than a second norm
   * implementation existing here -- adding zero is exactly what an absent bias means, and a
   * separate code path would be a second thing to keep correct.
   */
  private void layerNorm(
      float[] output, float[] input, int sequenceLength, float[] weight, float[] bias) {
    int dim = config.embeddingDim();
    float[] effectiveBias = bias == null ? zeroBias(dim) : bias;
    for (int position = 0; position < sequenceLength; position++) {
      TensorOps.layerNorm(
          output,
          position * dim,
          input,
          position * dim,
          weight,
          effectiveBias,
          dim,
          config.layerNormEps());
    }
  }

  private float[] zeroBias(int dim) {
    if (zeroBias == null || zeroBias.length != dim) {
      zeroBias = new float[dim];
    }
    return zeroBias;
  }

  /** Mean, standard deviation and L2 norm of the whole state: enough to localise, cheap to log. */
  private static void trace(String stage, float[] hidden) {
    if (TRACE == null) {
      return;
    }
    double sum = 0.0;
    double squares = 0.0;
    for (float value : hidden) {
      sum += value;
      squares += (double) value * value;
    }
    double mean = sum / hidden.length;
    double l2 = Math.sqrt(squares);
    double variance = squares / hidden.length - mean * mean;
    TRACE.add(
        String.format(
            java.util.Locale.ROOT,
            "%-12s mean=%+.6f sd=%.6f l2=%.4f first=%+.6f",
            stage,
            mean,
            Math.sqrt(Math.max(variance, 0.0)),
            l2,
            hidden[0]));
  }

  private static void addType(float[] hidden, int sequenceLength, int dim, float[] type) {
    for (int position = 0; position < sequenceLength; position++) {
      for (int element = 0; element < dim; element++) {
        hidden[position * dim + element] += type[element];
      }
    }
  }

  private static void addBias(float[] values, int sequenceLength, int width, float[] bias) {
    if (bias == null) {
      return;
    }
    for (int position = 0; position < sequenceLength; position++) {
      int base = position * width;
      for (int element = 0; element < width; element++) {
        values[base + element] += bias[element];
      }
    }
  }

  /**
   * The projection, through an injected kernel when one is eligible and the exact Java matmul
   * otherwise.
   *
   * <p>The exact path keeps F32 activations and a fixed accumulation order, which is what lets the
   * reference comparison attribute its residual to quantization rather than to arithmetic. It is
   * also, measured on 2026-10-23, about 3.3 s per decision on twelve cores -- and it was the only
   * path, because this class took no kernel at all. A decision model is the one place where that
   * matters most: every call pays the whole cost, since there is no prefix to amortise.
   *
   * <p>The injected kernel is asked rather than assumed: {@code isEligible} lets an implementation
   * refuse a shape whose boundary cost exceeds the win.
   *
   * <p>MEASURED 2026-10-23, and the reason the default is still {@code none()}: the Rust shim's
   * Q8_0 route is 2.2x faster per decision and costs three of 120 items on the published cohort
   * (accuracy 0.7833 to 0.7583), picking a different winner on four. The cause is not accumulation
   * order. {@code GgufTensorType::activation_format} in the kernel maps Q8_0 weights to {@code
   * ActivationFormat::Q8_0}, so it quantizes the activations to eight bits; the exact path keeps
   * them in F32. The capability is named Q8_0_F32_BATCHED_MATMUL, where the F32 describes the
   * output rather than the activations -- a name this comment previously misread.
   *
   * <p>So a kernel is safe for this model only if it keeps F32 activations. Q5_1 already has such a
   * route, "folded exactly as Java's PinnedReduction.dot folds", which is the shape a Q8_0 one
   * needs. Until that exists, an injected kernel changes published answers and the exact path stays
   * the default.
   */
  private void project(
      float[] output, float[] input, int sequenceLength, ModernBertWeights.Matrix matrix) {
    if (matrixKernel.isEligible(matrix.type(), sequenceLength, matrix.rows(), matrix.columns())) {
      matrixKernel.multiply(
          output,
          input,
          matrix.data(),
          matrix.type(),
          sequenceLength,
          matrix.rows(),
          matrix.columns());
      return;
    }
    TensorOps.ggufExactBatchedMatmul(
        output,
        input,
        matrix.data(),
        matrix.type(),
        sequenceLength,
        matrix.rows(),
        matrix.columns(),
        exactWeightRow);
  }

  /** The exact erf-based GELU the reference's {@code nn.GELU()} computes, not the tanh variant. */
  private static float gelu(float value) {
    if (TANH_GELU) {
      double inner = 0.7978845608028654 * (value + 0.044715 * value * value * value);
      return (float) (0.5 * value * (1.0 + Math.tanh(inner)));
    }
    return (float) (0.5 * value * (1.0 + erf(value / Math.sqrt(2.0))));
  }

  /**
   * Abramowitz and Stegun 7.1.26, max absolute error 1.5e-7.
   *
   * <p>A tanh approximation of GELU differs from {@code nn.GELU()} by up to 1e-3 around |x| = 2,
   * which is larger than the tolerance the logit comparison runs at — so the approximation would
   * fail the gate for the wrong reason.
   */
  private static double erf(double x) {
    double sign = Math.signum(x);
    double absolute = Math.abs(x);
    double t = 1.0 / (1.0 + 0.3275911 * absolute);
    double y =
        1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                    + 0.254829592)
                * t
                * Math.exp(-absolute * absolute);
    return sign * y;
  }

  static {
    // Referenced so an unused-import warning cannot hide a type change in the matrix contract.
    assert GgufTensorType.values().length > 0;
  }
}
