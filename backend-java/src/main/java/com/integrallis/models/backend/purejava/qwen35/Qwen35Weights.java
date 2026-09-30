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
package com.integrallis.models.backend.purejava.qwen35;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Mapped dense Qwen3.5 tensors, kept in their GGUF encodings. */
final class Qwen35Weights {

  private static final Thread ACCESS_PROBE = Thread.ofPlatform().unstarted(() -> {});

  private static final Set<GgufTensorType> MATRIX_TYPES =
      Set.of(
          GgufTensorType.F32,
          GgufTensorType.BF16,
          GgufTensorType.Q4_0,
          GgufTensorType.Q5_0,
          GgufTensorType.Q8_0,
          GgufTensorType.Q4_K,
          GgufTensorType.Q5_K,
          GgufTensorType.Q6_K,
          // Qwen3-Next stores ffn_gate_shexp and ffn_up_shexp as MXFP4. Verified from the published
          // GGUF, not assumed: every other matrix in that file is a K-quant.
          GgufTensorType.MXFP4);

  record Matrix(MemorySegment data, GgufTensorType type, int rows, int columns) {
    Matrix {
      Objects.requireNonNull(data, "data");
      Objects.requireNonNull(type, "type");
      if (rows <= 0 || columns <= 0) {
        throw new IllegalArgumentException("matrix dimensions must be positive");
      }
    }
  }

  record FullAttention(
      Matrix queryGate,
      Matrix key,
      Matrix value,
      Matrix output,
      float[] queryNorm,
      float[] keyNorm) {}

  /**
   * @param beta per-value-head beta projection, or null when the model fuses it with alpha
   * @param alpha per-value-head alpha projection, or null when fused
   * @param betaAlpha the fused {@code ssm_ba} projection, or null when beta and alpha are separate
   */
  record GatedDeltaNet(
      Matrix queryKeyValue,
      Matrix outputGate,
      Matrix beta,
      Matrix alpha,
      Matrix betaAlpha,
      float[] convolution,
      float[] timeStepBias,
      float[] decay,
      float[] outputNorm,
      Matrix output) {}

  /**
   * The routed feed-forward of a mixture-of-experts layer, or absent on a dense one.
   *
   * <p>Qwen3.5-MoE keeps a SHARED expert alongside the routed ones: it runs for every token and is
   * scaled by a sigmoid of a single scalar per token, which is what {@code sharedGate} produces.
   * The routed half is the familiar softmax-top-k-renormalise.
   *
   * <p>Experts are pre-sliced at load for the same reason as the Llama path: a token routed to 8 of
   * 256 experts across 40 layers would otherwise build thousands of segment views per token.
   */
  record MoeFeedForward(
      Matrix router,
      Matrix[] gate,
      Matrix[] up,
      Matrix[] down,
      Matrix sharedGate,
      Matrix sharedGateProjection,
      Matrix sharedUpProjection,
      Matrix sharedDownProjection) {}

  record Layer(
      float[] attentionNorm,
      float[] postAttentionNorm,
      FullAttention fullAttention,
      GatedDeltaNet gatedDeltaNet,
      Matrix ffnGate,
      Matrix ffnUp,
      Matrix ffnDown,
      // Null on a dense model, which is every qwen35 file; set only for qwen35moe.
      MoeFeedForward moe) {}

  private final MemorySegment tokenEmbedding;
  private final GgufTensorType tokenEmbeddingType;
  private final int embeddingDim;
  private final int vocabSize;
  private final float[] outputNorm;

  /**
   * The output head, or null when the model ties it to the token embedding.
   *
   * <p>Both shapes are real in this family: the dense Qwen3.5 files carry no {@code output.weight}
   * and tie, while Qwen3.5-MoE and Qwen3-Next carry one. Reading the token embedding on an untied
   * model does not fail -- the matrix has the right shape -- it just computes every logit from the
   * wrong weights, which is fluent, confident, wrong text. So it is looked up and used when present
   * rather than assumed absent.
   */
  private final Matrix outputHead;

  private final Layer[] layers;

  private Qwen35Weights(
      MemorySegment tokenEmbedding,
      GgufTensorType tokenEmbeddingType,
      int embeddingDim,
      int vocabSize,
      float[] outputNorm,
      Matrix outputHead,
      Layer[] layers) {
    this.tokenEmbedding = tokenEmbedding;
    this.tokenEmbeddingType = tokenEmbeddingType;
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
    this.outputNorm = outputNorm;
    this.outputHead = outputHead;
    this.layers = layers;
  }

  static Qwen35Weights fromGgufFile(GgufFile file, Qwen35Config config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    GgufTensorData embedding =
        matrixTensor(file, "token_embd.weight", config.vocabSize(), config.embeddingDim());
    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "blk." + layer + ".";
      FullAttention attention = null;
      GatedDeltaNet gatedDeltaNet = null;
      if (config.usesFullAttention(layer)) {
        attention =
            new FullAttention(
                matrix(
                    file,
                    prefix + "attn_q.weight",
                    2 * config.attentionQueryDim(),
                    config.embeddingDim()),
                matrix(
                    file,
                    prefix + "attn_k.weight",
                    config.attentionKeyDim(),
                    config.embeddingDim()),
                matrix(
                    file,
                    prefix + "attn_v.weight",
                    config.attentionKeyDim(),
                    config.embeddingDim()),
                matrix(
                    file,
                    prefix + "attn_output.weight",
                    config.embeddingDim(),
                    config.attentionQueryDim()),
                vector(file, prefix + "attn_q_norm.weight", config.attentionHeadDim()),
                vector(file, prefix + "attn_k_norm.weight", config.attentionHeadDim()));
      } else {
        boolean fused = file.hasTensor(prefix + "ssm_ba.weight");
        gatedDeltaNet =
            new GatedDeltaNet(
                matrix(
                    file, prefix + "attn_qkv.weight", config.gdnConvDim(), config.embeddingDim()),
                matrix(
                    file, prefix + "attn_gate.weight", config.gdnValueDim(), config.embeddingDim()),
                // Qwen3.5 publishes beta and alpha as two tensors; Qwen3-Next fuses them into
                // ssm_ba. Decided by what the file carries, not by the architecture name, because
                // the fused form is a tensor-layout choice rather than a different computation.
                fused
                    ? null
                    : matrix(
                        file,
                        prefix + "ssm_beta.weight",
                        config.gdnValueHeads(),
                        config.embeddingDim()),
                fused
                    ? null
                    : matrix(
                        file,
                        prefix + "ssm_alpha.weight",
                        config.gdnValueHeads(),
                        config.embeddingDim()),
                fused
                    ? matrix(
                        file,
                        prefix + "ssm_ba.weight",
                        2 * config.gdnValueHeads(),
                        config.embeddingDim())
                    : null,
                toTapMajorConvolution(
                    vector(
                        file,
                        prefix + "ssm_conv1d.weight",
                        config.gdnConvDim() * config.gdnConvKernel()),
                    config.gdnConvDim(),
                    config.gdnConvKernel()),
                vector(file, prefix + "ssm_dt.bias", config.gdnValueHeads()),
                vector(file, prefix + "ssm_a", config.gdnValueHeads()),
                vector(file, prefix + "ssm_norm.weight", config.gdnHeadDim()),
                matrix(
                    file, prefix + "ssm_out.weight", config.embeddingDim(), config.gdnValueDim()));
      }
      boolean routed = config.usesMixtureOfExperts();
      layers[layer] =
          new Layer(
              vector(file, prefix + "attn_norm.weight", config.embeddingDim()),
              vector(file, prefix + "post_attention_norm.weight", config.embeddingDim()),
              attention,
              gatedDeltaNet,
              routed
                  ? null
                  : matrix(
                      file, prefix + "ffn_gate.weight", config.hiddenDim(), config.embeddingDim()),
              routed
                  ? null
                  : matrix(
                      file, prefix + "ffn_up.weight", config.hiddenDim(), config.embeddingDim()),
              routed
                  ? null
                  : matrix(
                      file, prefix + "ffn_down.weight", config.embeddingDim(), config.hiddenDim()),
              routed ? moeFeedForward(file, prefix, config) : null);
    }
    return new Qwen35Weights(
        embedding.dataSegment(),
        embedding.type(),
        config.embeddingDim(),
        config.vocabSize(),
        vector(file, "output_norm.weight", config.embeddingDim()),
        optionalOutputHead(file, config),
        layers);
  }

  /**
   * The model's own output head, or null when it ties the head to the token embedding.
   *
   * <p>Absence is a shape, not a fault: the dense Qwen3.5 files genuinely have no {@code
   * output.weight}. Presence is equally real -- Qwen3.5-MoE and Qwen3-Next both ship one -- so this
   * asks the file instead of deciding from the architecture name.
   */
  private static Matrix optionalOutputHead(GgufFile file, Qwen35Config config) {
    if (!file.hasTensor("output.weight")) {
      return null;
    }
    return matrix(file, "output.weight", config.vocabSize(), config.embeddingDim());
  }

  /** The routed feed-forward for one layer, or null when the model is dense. */
  private static MoeFeedForward moeFeedForward(GgufFile file, String prefix, Qwen35Config config) {
    int dim = config.embeddingDim();
    int expertHidden = config.expertHiddenDim();
    int experts = config.numExperts();
    int sharedHidden = config.sharedExpertHiddenDim();
    GgufTensorData gate = file.getTensor(prefix + "ffn_gate_exps.weight");
    GgufTensorData up = file.getTensor(prefix + "ffn_up_exps.weight");
    GgufTensorData down = file.getTensor(prefix + "ffn_down_exps.weight");
    return new MoeFeedForward(
        matrix(file, prefix + "ffn_gate_inp.weight", experts, dim),
        expertSlices(gate, expertHidden, dim, experts, prefix + "ffn_gate_exps.weight"),
        expertSlices(up, expertHidden, dim, experts, prefix + "ffn_up_exps.weight"),
        // The down projection maps the expert width back to the model width, so its rows and
        // columns are the other way round from gate and up.
        expertSlices(down, dim, expertHidden, experts, prefix + "ffn_down_exps.weight"),
        // One scalar per token: the shared expert's own gate, sigmoided before it scales the
        // output.
        matrix(file, prefix + "ffn_gate_inp_shexp.weight", 1, dim),
        matrix(file, prefix + "ffn_gate_shexp.weight", sharedHidden, dim),
        matrix(file, prefix + "ffn_up_shexp.weight", sharedHidden, dim),
        matrix(file, prefix + "ffn_down_shexp.weight", dim, sharedHidden));
  }

  /**
   * Slices one stacked expert tensor into per-expert views.
   *
   * <p>GGUF stacks experts along the last dimension, so expert {@code e} occupies a contiguous
   * block of {@code rows} rows. The cut lands on whole rows, and a row is a whole number of
   * quantization blocks, so no block is straddled -- straddling one would reinterpret a block's
   * scale bytes as weights and yield plausible, wrong numbers rather than an error.
   */
  private static Matrix[] expertSlices(
      GgufTensorData tensor, int rows, int columns, int experts, String name) {
    GgufTensorType type = tensor.type();
    if (columns % type.blockSize() != 0) {
      throw new IllegalArgumentException(
          name
              + ": row length "
              + columns
              + " is not a multiple of the "
              + type.blockSize()
              + "-element "
              + type
              + " block, so experts cannot be sliced exactly");
    }
    long bytesPerRow = (long) (columns / type.blockSize()) * type.typeSize();
    long bytesPerExpert = bytesPerRow * rows;
    MemorySegment segment = tensor.dataSegment();
    long expected = bytesPerExpert * experts;
    if (segment.byteSize() != expected) {
      throw new IllegalArgumentException(
          name
              + " is "
              + segment.byteSize()
              + " bytes but "
              + experts
              + " experts of "
              + rows
              + "x"
              + columns
              + " need "
              + expected);
    }
    Matrix[] slices = new Matrix[experts];
    for (int expert = 0; expert < experts; expert++) {
      slices[expert] =
          new Matrix(segment.asSlice(bytesPerExpert * expert, bytesPerExpert), type, rows, columns);
    }
    return slices;
  }

  void embedToken(int token, float[] output) {
    if (token < 0 || token >= vocabSize) {
      throw new IllegalArgumentException("token out of range: " + token);
    }
    GgufTensorValues.dequantizeRow(tokenEmbedding, tokenEmbeddingType, token, embeddingDim, output);
  }

  Matrix output() {
    return outputHead != null
        ? outputHead
        : new Matrix(tokenEmbedding, tokenEmbeddingType, vocabSize, embeddingDim);
  }

  /** Whether this model carries its own output head rather than tying it to the embedding. */
  boolean hasUntiedOutputHead() {
    return outputHead != null;
  }

  float[] outputNorm() {
    return outputNorm;
  }

  Layer layer(int layer) {
    return layers[layer];
  }

  boolean usesMatrixType(GgufTensorType type) {
    if (tokenEmbeddingType == type) {
      return true;
    }
    for (Layer layer : layers) {
      for (Matrix matrix : layerMatrices(layer)) {
        if (matrix.type() == type) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * The widest input any projection in this model takes, which is what the quantized-activation
   * scratch has to hold.
   *
   * <p>Derived by scanning the matrices rather than by listing the widths a Qwen3.5 is expected to
   * have. The listed form was wrong the moment routing arrived: it named the dense hidden width,
   * which a routed model publishes as 0, and so sized the buffers for a feed-forward that is not
   * there. A scan cannot be wrong by omission, and this is a once-per-session cost.
   */
  int widestProjectionInput() {
    // The token embedding and the output head both project across the model width.
    int widest = embeddingDim;
    for (Layer layer : layers) {
      for (Matrix matrix : layerMatrices(layer)) {
        widest = Math.max(widest, matrix.columns());
      }
    }
    return widest;
  }

  /** Every projection matrix a layer holds, whichever variant of layer it is. */
  private static List<Matrix> layerMatrices(Layer layer) {
    List<Matrix> matrices = new ArrayList<>(feedForwardMatrices(layer));
    if (layer.fullAttention() != null) {
      FullAttention attention = layer.fullAttention();
      matrices.add(attention.queryGate());
      matrices.add(attention.key());
      matrices.add(attention.value());
      matrices.add(attention.output());
    }
    if (layer.gatedDeltaNet() != null) {
      GatedDeltaNet gdn = layer.gatedDeltaNet();
      matrices.add(gdn.queryKeyValue());
      matrices.add(gdn.outputGate());
      // Exactly one of these shapes is present: two separate projections, or the fused ssm_ba.
      if (gdn.betaAlpha() != null) {
        matrices.add(gdn.betaAlpha());
      } else {
        matrices.add(gdn.beta());
        matrices.add(gdn.alpha());
      }
      matrices.add(gdn.output());
    }
    return matrices;
  }

  boolean hasThreadShareableProjectionWeights() {
    if (!tokenEmbedding.isAccessibleBy(ACCESS_PROBE)) {
      return false;
    }
    for (Layer layer : layers) {
      if (!shareable(layerMatrices(layer).toArray(new Matrix[0]))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Every feed-forward projection a layer actually holds, dense or routed.
   *
   * <p>Both callers below scan a layer's weights -- one for a tensor type, one for shareability --
   * and both used to name {@code ffnGate}/{@code ffnUp}/{@code ffnDown} directly. Those are null on
   * a routed layer, and one of the callers runs while building every session's scratch, so the
   * shape of the scan has to come from the layer rather than from an assumption about it.
   */
  private static List<Matrix> feedForwardMatrices(Layer layer) {
    if (layer.moe() == null) {
      return List.of(layer.ffnGate(), layer.ffnUp(), layer.ffnDown());
    }
    MoeFeedForward moe = layer.moe();
    List<Matrix> matrices = new ArrayList<>(3 * moe.gate().length + 5);
    matrices.add(moe.router());
    matrices.addAll(Arrays.asList(moe.gate()));
    matrices.addAll(Arrays.asList(moe.up()));
    matrices.addAll(Arrays.asList(moe.down()));
    matrices.add(moe.sharedGate());
    matrices.add(moe.sharedGateProjection());
    matrices.add(moe.sharedUpProjection());
    matrices.add(moe.sharedDownProjection());
    return matrices;
  }

  private static boolean shareable(Matrix... matrices) {
    for (Matrix matrix : matrices) {
      if (!matrix.data().isAccessibleBy(ACCESS_PROBE)) {
        return false;
      }
    }
    return true;
  }

  private static Matrix matrix(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = matrixTensor(file, name, rows, columns);
    return new Matrix(tensor.dataSegment(), tensor.type(), rows, columns);
  }

  private static GgufTensorData matrixTensor(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, columns, rows);
    if (!MATRIX_TYPES.contains(tensor.type())) {
      throw new IllegalArgumentException(name + " has unsupported matrix type " + tensor.type());
    }
    return tensor;
  }

  private static float[] vector(GgufFile file, String name, int length) {
    GgufTensorData tensor = file.getTensor(name);
    if (tensor.info().elementCount() != length) {
      throw new IllegalArgumentException(
          name + " must contain " + length + " values, found " + tensor.info().elementCount());
    }
    if (tensor.type() != GgufTensorType.F32) {
      throw new IllegalArgumentException(name + " must be F32, found " + tensor.type());
    }
    return GgufTensorValues.toFloatArray(tensor);
  }

  static float[] toTapMajorConvolution(float[] channelMajor, int channels, int kernelSize) {
    Objects.requireNonNull(channelMajor, "channelMajor");
    if (channels < 1 || kernelSize < 1 || channelMajor.length != channels * kernelSize) {
      throw new IllegalArgumentException("invalid causal convolution weight shape");
    }
    float[] tapMajor = new float[channelMajor.length];
    for (int channel = 0; channel < channels; channel++) {
      for (int tap = 0; tap < kernelSize; tap++) {
        tapMajor[tap * channels + channel] = channelMajor[channel * kernelSize + tap];
      }
    }
    return tapMajor;
  }

  private static void requireShape(GgufTensorData tensor, long... expected) {
    // GGUF drops trailing dimensions of 1, so a single-row matrix is written as a 1-D tensor.
    // ffn_gate_inp_shexp is exactly that -- one router row over the model width -- and the real
    // files carry it as [n_embd], not [n_embd, 1]. Requiring the two-dimensional form rejected
    // every
    // published Qwen3.5-MoE; the synthetic fixtures did not notice because a builder given
    // {columns, 1} writes both dimensions.
    if (expected.length == 2 && expected[1] == 1L && tensor.shape().length == 1) {
      if (tensor.shape()[0] == expected[0]) {
        return;
      }
    }
    if (!Arrays.equals(expected, tensor.shape())) {
      throw new IllegalArgumentException(
          tensor.name()
              + " shape must be "
              + Arrays.toString(expected)
              + ", found "
              + Arrays.toString(tensor.shape()));
    }
  }
}
