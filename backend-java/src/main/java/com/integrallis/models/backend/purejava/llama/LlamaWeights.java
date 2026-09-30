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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.huggingface.Qwen2HuggingFaceConfig;
import com.integrallis.models.backend.purejava.tensor.TensorSource;
import com.integrallis.models.backend.purejava.tensor.TensorStorage;
import com.integrallis.models.backend.purejava.tensor.TensorView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Holds mapped weight tensors for a Llama-family model independently of its artifact container. */
public final class LlamaWeights {

  private final MemorySegment tokenEmbeddingSegment;
  private final GgufTensorType tokenEmbeddingType;
  private final int embeddingDim;
  private final float[] outputNormWeight;
  private final MemorySegment outputSegment;
  private final GgufTensorType outputType;
  private final LayerWeights[] layers;

  /**
   * The routed feed-forward of a mixture-of-experts layer, or absent on a dense one.
   *
   * <p>Experts are pre-sliced at load rather than on each use: a token routed to eight of 128
   * experts across 48 layers would otherwise build 1,152 segment views per token. The slices are
   * exact for quantized weights because an expert occupies a whole number of rows and a row is
   * {@code columns / blockSize} whole blocks -- the same property the Phi-3 fused split relies on,
   * asserted rather than assumed in {@link #expertSlices}.
   *
   * @param router the gate projection producing one logit per expert
   * @param gate per-expert gate projections, indexed by expert
   * @param up per-expert up projections
   * @param down per-expert down projections
   */
  public record MoeWeights(
      MemorySegment router,
      GgufTensorType routerType,
      MemorySegment[] gate,
      GgufTensorType gateType,
      MemorySegment[] up,
      GgufTensorType upType,
      MemorySegment[] down,
      GgufTensorType downType) {}

  /** Per-layer weight tensors. */
  public record LayerWeights(
      float[] attentionNorm,
      MemorySegment wq,
      GgufTensorType wqType,
      float[] qBias,
      float[] qNorm,
      MemorySegment wk,
      GgufTensorType wkType,
      float[] kBias,
      float[] kNorm,
      MemorySegment wv,
      GgufTensorType wvType,
      float[] vBias,
      MemorySegment wo,
      GgufTensorType woType,
      float[] attentionPostNorm,
      float[] ffnNorm,
      MemorySegment ffnGate,
      GgufTensorType ffnGateType,
      MemorySegment ffnUp,
      GgufTensorType ffnUpType,
      MemorySegment ffnDown,
      GgufTensorType ffnDownType,
      float[] ffnPostNorm,
      MoeWeights moe) {}

  private LlamaWeights(
      MemorySegment tokenEmbeddingSegment,
      GgufTensorType tokenEmbeddingType,
      int embeddingDim,
      float[] outputNormWeight,
      MemorySegment outputSegment,
      GgufTensorType outputType,
      LayerWeights[] layers) {
    this.tokenEmbeddingSegment = tokenEmbeddingSegment;
    this.tokenEmbeddingType = tokenEmbeddingType;
    this.embeddingDim = embeddingDim;
    this.outputNormWeight = outputNormWeight;
    this.outputSegment = outputSegment;
    this.outputType = outputType;
    this.layers = layers;
  }

  /** Loads weights from a parsed GGUF file using the standard Llama tensor naming convention. */
  public static LlamaWeights fromGgufFile(GgufFile file, LlamaConfig config) {
    GgufTensorData tokenEmbed = file.getTensor("token_embd.weight");
    float[] outputNorm = loadF32Tensor(file, "output_norm.weight");

    GgufTensorData output;
    try {
      output = file.getTensor("output.weight");
    } catch (IllegalArgumentException e) {
      // Some models tie output weights to token embeddings
      output = tokenEmbed;
    }

    LayerWeights[] layers = new LayerWeights[config.numLayers()];
    for (int i = 0; i < config.numLayers(); i++) {
      String prefix = "blk." + i + ".";
      float[] attnNorm = loadF32Tensor(file, prefix + "attn_norm.weight");
      // Phi-3 fuses Q, K and V into one attn_qkv tensor and the gate/up projections into one
      // ffn_up tensor. Split them here so every decoder below sees the same separate projections
      // as any other Llama-family model.
      FusedQkv fused = fusedQkv(file, prefix, config);
      GgufTensorData wq = fused == null ? file.getTensor(prefix + "attn_q.weight") : null;
      GgufTensorData wk = fused == null ? file.getTensor(prefix + "attn_k.weight") : null;
      GgufTensorData wv = fused == null ? file.getTensor(prefix + "attn_v.weight") : null;
      float[] qBias = loadOptionalF32Tensor(file, prefix + "attn_q.bias", config.queryDim());
      float[] qNorm = loadOptionalF32Tensor(file, prefix + "attn_q_norm.weight", config.headDim());
      float[] kBias = loadOptionalF32Tensor(file, prefix + "attn_k.bias", config.keyDim());
      float[] kNorm = loadOptionalF32Tensor(file, prefix + "attn_k_norm.weight", config.headDim());
      float[] vBias = loadOptionalF32Tensor(file, prefix + "attn_v.bias", config.valueDim());
      GgufTensorData wo = file.getTensor(prefix + "attn_output.weight");
      float[] attentionPostNorm =
          config.usesPostAttentionNorm()
              ? loadF32Tensor(file, prefix + "post_attention_norm.weight")
              : new float[0];
      float[] ffnNorm = loadF32Tensor(file, prefix + "ffn_norm.weight");
      FusedGateUp fusedFfn = fusedGateUp(file, prefix, config);
      // A fully routed model publishes no dense feed-forward at all, so these are absent rather
      // than optional-with-a-default: the routed path reads them from MoeWeights instead.
      boolean denseFfn = !config.usesMixtureOfExperts();
      GgufTensorData ffnGate =
          fusedFfn == null && denseFfn ? file.getTensor(prefix + "ffn_gate.weight") : null;
      GgufTensorData ffnUp =
          fusedFfn == null && denseFfn ? file.getTensor(prefix + "ffn_up.weight") : null;
      GgufTensorData ffnDown = denseFfn ? file.getTensor(prefix + "ffn_down.weight") : null;
      float[] ffnPostNorm =
          config.usesPostFfnNorm()
              ? loadF32Tensor(file, prefix + "post_ffw_norm.weight")
              : new float[0];

      MoeWeights moe = moeWeights(file, prefix, config);
      layers[i] =
          new LayerWeights(
              attnNorm,
              fused == null ? wq.dataSegment() : fused.q(),
              fused == null ? wq.type() : fused.type(),
              qBias,
              qNorm,
              fused == null ? wk.dataSegment() : fused.k(),
              fused == null ? wk.type() : fused.type(),
              kBias,
              kNorm,
              fused == null ? wv.dataSegment() : fused.v(),
              fused == null ? wv.type() : fused.type(),
              vBias,
              wo.dataSegment(),
              wo.type(),
              attentionPostNorm,
              ffnNorm,
              fusedFfn != null ? fusedFfn.gate() : ffnGate == null ? null : ffnGate.dataSegment(),
              fusedFfn != null ? fusedFfn.type() : ffnGate == null ? null : ffnGate.type(),
              fusedFfn != null ? fusedFfn.up() : ffnUp == null ? null : ffnUp.dataSegment(),
              fusedFfn != null ? fusedFfn.type() : ffnUp == null ? null : ffnUp.type(),
              ffnDown == null ? null : ffnDown.dataSegment(),
              ffnDown == null ? null : ffnDown.type(),
              ffnPostNorm,
              moe);
    }

    return new LlamaWeights(
        tokenEmbed.dataSegment(),
        tokenEmbed.type(),
        config.embeddingDim(),
        outputNorm,
        output.dataSegment(),
        output.type(),
        layers);
  }

  /**
   * Loads a Qwen-family GGUF whose tensor names retain the original Hugging Face layout.
   *
   * <p>Audio GGUF packagers commonly preserve names such as {@code model.layers.0.self_attn}
   * instead of rewriting them to llama.cpp's {@code blk.0} convention. The tensor storage is still
   * ordinary mapped GGUF, including quantized matrices, so only the binding differs.
   */
  public static LlamaWeights fromHuggingFaceNamedGguf(GgufFile file, LlamaConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    if (config.architecture() != DecoderArchitecture.QWEN2
        && config.architecture() != DecoderArchitecture.QWEN3) {
      throw new IllegalArgumentException(
          "Hugging Face named GGUF loading currently requires Qwen 2 or Qwen 3");
    }

    GgufTensorData tokenEmbed =
        requireMatrix(file, "model.embed_tokens.weight", config.vocabSize(), config.embeddingDim());
    GgufTensorData output;
    boolean tiedOutput = false;
    try {
      output = requireMatrix(file, "lm_head.weight", config.vocabSize(), config.embeddingDim());
    } catch (IllegalArgumentException missingOutput) {
      if (!isMissingTensor(missingOutput)) {
        throw missingOutput;
      }
      output = tokenEmbed;
      tiedOutput = true;
    }
    ExecutionTensor preparedTokenEmbed = prepareHuggingFaceMatrix(tokenEmbed);
    ExecutionTensor preparedOutput =
        tiedOutput ? preparedTokenEmbed : prepareHuggingFaceMatrix(output);

    LayerWeights[] layers = new LayerWeights[config.numLayers()];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "model.layers." + layer + ".";
      GgufTensorData wq =
          requireMatrix(
              file, prefix + "self_attn.q_proj.weight", config.queryDim(), config.embeddingDim());
      GgufTensorData wk =
          requireMatrix(
              file, prefix + "self_attn.k_proj.weight", config.keyDim(), config.embeddingDim());
      GgufTensorData wv =
          requireMatrix(
              file, prefix + "self_attn.v_proj.weight", config.valueDim(), config.embeddingDim());
      GgufTensorData wo =
          requireMatrix(
              file,
              prefix + "self_attn.o_proj.weight",
              config.embeddingDim(),
              config.attentionOutputDim());
      GgufTensorData gate =
          requireMatrix(
              file, prefix + "mlp.gate_proj.weight", config.hiddenDim(), config.embeddingDim());
      GgufTensorData up =
          requireMatrix(
              file, prefix + "mlp.up_proj.weight", config.hiddenDim(), config.embeddingDim());
      GgufTensorData down =
          requireMatrix(
              file, prefix + "mlp.down_proj.weight", config.embeddingDim(), config.hiddenDim());
      ExecutionTensor preparedWq = prepareHuggingFaceMatrix(wq);
      ExecutionTensor preparedWk = prepareHuggingFaceMatrix(wk);
      ExecutionTensor preparedWv = prepareHuggingFaceMatrix(wv);
      ExecutionTensor preparedWo = prepareHuggingFaceMatrix(wo);
      ExecutionTensor preparedGate = prepareHuggingFaceMatrix(gate);
      ExecutionTensor preparedUp = prepareHuggingFaceMatrix(up);
      ExecutionTensor preparedDown = prepareHuggingFaceMatrix(down);
      layers[layer] =
          new LayerWeights(
              loadVector(file, prefix + "input_layernorm.weight", config.embeddingDim()),
              preparedWq.data(),
              preparedWq.type(),
              loadOptionalF32Tensor(file, prefix + "self_attn.q_proj.bias", config.queryDim()),
              loadOptionalF32Tensor(file, prefix + "self_attn.q_norm.weight", config.headDim()),
              preparedWk.data(),
              preparedWk.type(),
              loadOptionalF32Tensor(file, prefix + "self_attn.k_proj.bias", config.keyDim()),
              loadOptionalF32Tensor(file, prefix + "self_attn.k_norm.weight", config.headDim()),
              preparedWv.data(),
              preparedWv.type(),
              loadOptionalF32Tensor(file, prefix + "self_attn.v_proj.bias", config.valueDim()),
              preparedWo.data(),
              preparedWo.type(),
              new float[0],
              loadVector(file, prefix + "post_attention_layernorm.weight", config.embeddingDim()),
              preparedGate.data(),
              preparedGate.type(),
              preparedUp.data(),
              preparedUp.type(),
              preparedDown.data(),
              preparedDown.type(),
              new float[0],
              // Safetensors bundles carry no stacked expert tensors: this path is dense.
              null);
    }

    return new LlamaWeights(
        preparedTokenEmbed.data(),
        preparedTokenEmbed.type(),
        config.embeddingDim(),
        loadVector(file, "model.norm.weight", config.embeddingDim()),
        preparedOutput.data(),
        preparedOutput.type(),
        layers);
  }

  private static ExecutionTensor prepareHuggingFaceMatrix(GgufTensorData tensor) {
    if (tensor.type() != GgufTensorType.BF16) {
      return new ExecutionTensor(tensor.dataSegment(), tensor.type());
    }
    float[] decoded = GgufTensorValues.toFloatArray(tensor);
    return new ExecutionTensor(MemorySegment.ofArray(decoded).asReadOnly(), GgufTensorType.F32);
  }

  private record ExecutionTensor(MemorySegment data, GgufTensorType type) {}

  /**
   * Loads Qwen 2 weights from Hugging Face Safetensors names without expanding BF16 matrices.
   * Vector and bias tensors are decoded once; projection, embedding, and output matrices remain
   * mapped and read-only.
   */
  public static LlamaWeights fromQwen2Safetensors(
      TensorSource source, Qwen2HuggingFaceConfig huggingFaceConfig) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(huggingFaceConfig, "huggingFaceConfig");
    if (!"safetensors".equals(source.format())) {
      throw new IllegalArgumentException(
          "Qwen 2 Hugging Face weights require safetensors; got " + source.format());
    }
    if (huggingFaceConfig.model().architecture() != DecoderArchitecture.QWEN2) {
      throw new IllegalArgumentException("Qwen 2 Safetensors require QWEN2 model configuration");
    }
    if (!"bfloat16".equals(huggingFaceConfig.torchDtype())) {
      throw new IllegalArgumentException(
          "Qwen 2 Safetensors runtime requires torch_dtype bfloat16; got "
              + huggingFaceConfig.torchDtype());
    }

    LlamaConfig config = huggingFaceConfig.model();
    TensorView tokenEmbed =
        requireBf16(source, "model.embed_tokens.weight", config.vocabSize(), config.embeddingDim());
    TensorView output =
        huggingFaceConfig.tieWordEmbeddings()
            ? tokenEmbed
            : requireBf16(source, "lm_head.weight", config.vocabSize(), config.embeddingDim());
    float[] outputNorm = loadBf16Vector(source, "model.norm.weight", config.embeddingDim());

    LayerWeights[] layers = new LayerWeights[config.numLayers()];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "model.layers." + layer + ".";
      TensorView wq =
          requireBf16(
              source, prefix + "self_attn.q_proj.weight", config.queryDim(), config.embeddingDim());
      TensorView wk =
          requireBf16(
              source, prefix + "self_attn.k_proj.weight", config.keyDim(), config.embeddingDim());
      TensorView wv =
          requireBf16(
              source, prefix + "self_attn.v_proj.weight", config.valueDim(), config.embeddingDim());
      TensorView wo =
          requireBf16(
              source,
              prefix + "self_attn.o_proj.weight",
              config.embeddingDim(),
              config.attentionOutputDim());
      TensorView gate =
          requireBf16(
              source, prefix + "mlp.gate_proj.weight", config.hiddenDim(), config.embeddingDim());
      TensorView up =
          requireBf16(
              source, prefix + "mlp.up_proj.weight", config.hiddenDim(), config.embeddingDim());
      TensorView down =
          requireBf16(
              source, prefix + "mlp.down_proj.weight", config.embeddingDim(), config.hiddenDim());
      layers[layer] =
          new LayerWeights(
              loadBf16Vector(source, prefix + "input_layernorm.weight", config.embeddingDim()),
              wq.data(),
              GgufTensorType.BF16,
              loadBf16Vector(source, prefix + "self_attn.q_proj.bias", config.queryDim()),
              new float[0],
              wk.data(),
              GgufTensorType.BF16,
              loadBf16Vector(source, prefix + "self_attn.k_proj.bias", config.keyDim()),
              new float[0],
              wv.data(),
              GgufTensorType.BF16,
              loadBf16Vector(source, prefix + "self_attn.v_proj.bias", config.valueDim()),
              wo.data(),
              GgufTensorType.BF16,
              new float[0],
              loadBf16Vector(
                  source, prefix + "post_attention_layernorm.weight", config.embeddingDim()),
              gate.data(),
              GgufTensorType.BF16,
              up.data(),
              GgufTensorType.BF16,
              down.data(),
              GgufTensorType.BF16,
              new float[0],
              // Safetensors bundles carry no stacked expert tensors: these paths are dense.
              null);
    }

    return new LlamaWeights(
        tokenEmbed.data(),
        GgufTensorType.BF16,
        config.embeddingDim(),
        outputNorm,
        output.data(),
        GgufTensorType.BF16,
        layers);
  }

  /**
   * Dequantizes a single token embedding row into the provided output buffer. Only dequantizes one
   * row of [embeddingDim] floats — avoids materializing the full vocab×dim table.
   */
  public void embedToken(int token, float[] out) {
    dequantizeRow(tokenEmbeddingSegment, tokenEmbeddingType, token, embeddingDim, out);
  }

  /** Returns the quantized output (language model head) weight segment. */
  public MemorySegment outputSegment() {
    return outputSegment;
  }

  /** Returns the quantization type of the output weight. */
  public GgufTensorType outputType() {
    return outputType;
  }

  public float[] outputNormWeight() {
    return outputNormWeight;
  }

  public LayerWeights layer(int i) {
    return layers[i];
  }

  /**
   * Dequantizes a single row of a quantized 2D tensor. For F32 data, directly copies. For quantized
   * types, dequantizes just the row.
   */
  private static void dequantizeRow(
      MemorySegment segment, GgufTensorType type, int row, int cols, float[] out) {
    GgufTensorValues.dequantizeRow(segment, type, row, cols, out);
  }

  /**
   * Loads a tensor and dequantizes it to F32 if needed. Supports F32, F16, Q4_0, and Q8_0 source
   * formats.
   */
  private static float[] loadF32Tensor(GgufFile file, String name) {
    return GgufTensorValues.toFloatArray(file.getTensor(name));
  }

  /**
   * Whether a tensor is present. {@link GgufFile} exposes no presence check, and absence is
   * signalled by the "Tensor not found" message, which is the same convention {@link
   * #loadOptionalF32Tensor} relies on.
   */
  private static boolean hasTensor(GgufFile file, String name) {
    try {
      file.getTensor(name);
      return true;
    } catch (IllegalArgumentException exception) {
      if (exception.getMessage() != null && exception.getMessage().contains("Tensor not found")) {
        return false;
      }
      throw exception;
    }
  }

  /**
   * Q, K and V sliced out of a fused {@code attn_qkv} tensor, or null when the model is not fused.
   */
  private record FusedQkv(MemorySegment q, MemorySegment k, MemorySegment v, GgufTensorType type) {}

  /**
   * Gate and up sliced out of a fused {@code ffn_up} tensor, or null when the model is not fused.
   */
  private record FusedGateUp(MemorySegment gate, MemorySegment up, GgufTensorType type) {}

  /**
   * Splits Phi-3's fused {@code attn_qkv} into Q, K and V.
   *
   * <p>Detected by absence rather than by architecture id: a model either publishes separate
   * projections or it does not, and keying off the name would silently mis-handle any other model
   * that fuses them. The stacking order is Q then K then V along the output dimension, with widths
   * {@code numHeads*headDim}, {@code numKvHeads*headDim} and {@code numKvHeads*headDim} -- verified
   * against phi-3.5-mini-instruct, whose attn_qkv is [3072 x 5120] with 24 heads and 8 KV heads:
   * 3072 + 1024 + 1024 = 5120.
   */
  private static FusedQkv fusedQkv(GgufFile file, String prefix, LlamaConfig config) {
    if (hasTensor(file, prefix + "attn_q.weight")) {
      return null;
    }
    GgufTensorData qkv = file.getTensor(prefix + "attn_qkv.weight");
    int columns = config.embeddingDim();
    int qRows = config.queryDim();
    int kvRows = config.keyDim();
    long expected =
        (long) (columns / qkv.type().blockSize()) * qkv.type().typeSize() * (qRows + 2L * kvRows);
    if (qkv.dataSegment().byteSize() != expected) {
      throw new IllegalArgumentException(
          prefix
              + "attn_qkv.weight is "
              + qkv.dataSegment().byteSize()
              + " bytes but Q+K+V for this config needs "
              + expected);
    }
    MemorySegment seg = qkv.dataSegment();
    return new FusedQkv(
        rowSlice(seg, qkv.type(), columns, 0, qRows, prefix + "attn_qkv(q)"),
        rowSlice(seg, qkv.type(), columns, qRows, kvRows, prefix + "attn_qkv(k)"),
        rowSlice(seg, qkv.type(), columns, qRows + kvRows, kvRows, prefix + "attn_qkv(v)"),
        qkv.type());
  }

  /**
   * Splits Phi-3's fused {@code ffn_up} into the gate and up projections.
   *
   * <p>Gate first, then up, each {@code hiddenDim} rows wide -- verified against
   * phi-3.5-mini-instruct, whose ffn_up is [3072 x 16384] against a feed_forward_length of 8192.
   */
  private static FusedGateUp fusedGateUp(GgufFile file, String prefix, LlamaConfig config) {
    if (hasTensor(file, prefix + "ffn_gate.weight")) {
      return null;
    }
    // A routed layer has no dense gate either, and absence is how the Phi-3 fusion is detected, so
    // without this the mixture-of-experts models fall into the fused split and fail claiming a
    // malformed ffn_up. Verified on Qwen3-30B-A3B: no layer publishes any dense ffn tensor.
    if (config.usesMixtureOfExperts()) {
      return null;
    }
    GgufTensorData up = file.getTensor(prefix + "ffn_up.weight");
    int columns = config.embeddingDim();
    int hidden = config.hiddenDim();
    long expected = (long) (columns / up.type().blockSize()) * up.type().typeSize() * (2L * hidden);
    if (up.dataSegment().byteSize() != expected) {
      throw new IllegalArgumentException(
          prefix
              + "ffn_up.weight is "
              + up.dataSegment().byteSize()
              + " bytes but a fused gate+up for this config needs "
              + expected);
    }
    MemorySegment seg = up.dataSegment();
    return new FusedGateUp(
        rowSlice(seg, up.type(), columns, 0, hidden, prefix + "ffn_up(gate)"),
        rowSlice(seg, up.type(), columns, hidden, hidden, prefix + "ffn_up(up)"),
        up.type());
  }

  /**
   * One row-range of a fused projection, as a zero-copy slice.
   *
   * <p>Phi-3 publishes a single {@code attn_qkv.weight} holding Q, K and V stacked along the output
   * dimension, and a single {@code ffn_up.weight} holding the gate and up projections stacked the
   * same way. Splitting them is a byte-range slice, not a copy, and it is exact for quantized
   * weights only because the cut lands on a whole number of rows: a row is {@code columns /
   * blockSize} whole blocks, so no quantization block is ever straddled. Verified on
   * phi-3.5-mini-instruct, where {@code attn_qkv} is Q5_K [3072 x 5120] and each row is 3072
   * elements = 12 whole 256-element blocks, and {@code ffn_up} is Q4_K [3072 x 16384].
   *
   * <p>Cutting mid-block would silently reinterpret a block's scale bytes as weights, which
   * produces plausible-looking numbers and slightly wrong output rather than an error -- so the
   * alignment is asserted rather than assumed.
   */
  private static MemorySegment rowSlice(
      MemorySegment fused, GgufTensorType type, int columns, int firstRow, int rows, String name) {
    if (columns % type.blockSize() != 0) {
      throw new IllegalArgumentException(
          name
              + ": cannot split a fused "
              + type
              + " tensor whose row length "
              + columns
              + " is not a multiple of its "
              + type.blockSize()
              + "-element quantization block");
    }
    long bytesPerRow = (long) (columns / type.blockSize()) * type.typeSize();
    long offset = bytesPerRow * firstRow;
    long length = bytesPerRow * rows;
    if (offset + length > fused.byteSize()) {
      throw new IllegalArgumentException(
          name
              + ": fused tensor is "
              + fused.byteSize()
              + " bytes, too small for rows "
              + firstRow
              + ".."
              + (firstRow + rows - 1));
    }
    return fused.asSlice(offset, length);
  }

  /**
   * Slices one stacked expert tensor into per-expert views.
   *
   * <p>GGUF stacks experts along the last dimension, so expert {@code e} occupies rows {@code e *
   * rows} through {@code (e + 1) * rows - 1}. The cut lands on whole rows, and a row is a whole
   * number of quantization blocks, so no block is straddled -- straddling one would reinterpret a
   * block's scale bytes as weights and yield plausible, wrong numbers rather than an error.
   */
  private static MemorySegment[] expertSlices(
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
    MemorySegment[] slices = new MemorySegment[experts];
    for (int expert = 0; expert < experts; expert++) {
      slices[expert] = segment.asSlice(bytesPerExpert * expert, bytesPerExpert);
    }
    return slices;
  }

  /** The routed feed-forward for one layer, or null when the model is dense. */
  private static MoeWeights moeWeights(GgufFile file, String prefix, LlamaConfig config) {
    if (!config.usesMixtureOfExperts()) {
      return null;
    }
    int dim = config.embeddingDim();
    int hidden = config.expertHiddenDim();
    int experts = config.numExperts();
    GgufTensorData router = file.getTensor(prefix + "ffn_gate_inp.weight");
    GgufTensorData gate = file.getTensor(prefix + "ffn_gate_exps.weight");
    GgufTensorData up = file.getTensor(prefix + "ffn_up_exps.weight");
    GgufTensorData down = file.getTensor(prefix + "ffn_down_exps.weight");
    return new MoeWeights(
        router.dataSegment(),
        router.type(),
        expertSlices(gate, hidden, dim, experts, prefix + "ffn_gate_exps.weight"),
        gate.type(),
        expertSlices(up, hidden, dim, experts, prefix + "ffn_up_exps.weight"),
        up.type(),
        // The down projection maps hidden back to the embedding dimension, so its rows and columns
        // are the other way round from gate and up.
        expertSlices(down, dim, hidden, experts, prefix + "ffn_down_exps.weight"),
        down.type());
  }

  private static float[] loadOptionalF32Tensor(GgufFile file, String name, int expectedLength) {
    try {
      float[] values = loadF32Tensor(file, name);
      if (values.length != expectedLength) {
        throw new IllegalArgumentException(
            name + " length must be " + expectedLength + ", got " + values.length);
      }
      return values;
    } catch (IllegalArgumentException e) {
      if (e.getMessage() != null && e.getMessage().contains("Tensor not found")) {
        return new float[0];
      }
      throw e;
    }
  }

  private static float[] loadVector(GgufFile file, String name, int expectedLength) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, expectedLength);
    return GgufTensorValues.toFloatArray(tensor);
  }

  private static GgufTensorData requireMatrix(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, columns, rows);
    return tensor;
  }

  private static void requireShape(GgufTensorData tensor, long... expected) {
    long[] actual = tensor.shape();
    if (actual.length < expected.length) {
      throw shapeMismatch(tensor.name(), expected, actual);
    }
    for (int index = 0; index < expected.length; index++) {
      if (actual[index] != expected[index]) {
        throw shapeMismatch(tensor.name(), expected, actual);
      }
    }
    for (int index = expected.length; index < actual.length; index++) {
      if (actual[index] != 1) {
        throw shapeMismatch(tensor.name(), expected, actual);
      }
    }
  }

  private static IllegalArgumentException shapeMismatch(
      String name, long[] expected, long[] actual) {
    return new IllegalArgumentException(
        name
            + " shape must be "
            + Arrays.toString(expected)
            + " with optional singleton dimensions; got "
            + Arrays.toString(actual));
  }

  private static boolean isMissingTensor(IllegalArgumentException failure) {
    return failure.getMessage() != null && failure.getMessage().contains("Tensor not found");
  }

  private static TensorView requireBf16(TensorSource source, String name, long... expectedShape) {
    TensorView tensor = source.tensor(name);
    if (!Arrays.equals(tensor.shape(), expectedShape)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expectedShape)
              + "; got "
              + Arrays.toString(tensor.shape()));
    }
    TensorStorage storage = tensor.storage();
    if (!"safetensors".equals(storage.format())
        || !"BF16".equals(storage.type())
        || storage.blockElements() != 1
        || storage.blockBytes() != Short.BYTES) {
      throw new IllegalArgumentException(
          name + " must use Safetensors BF16 storage; got " + storage);
    }
    return tensor;
  }

  private static float[] loadBf16Vector(TensorSource source, String name, int length) {
    TensorView tensor = requireBf16(source, name, length);
    float[] values = new float[length];
    GgufTensorValues.dequantizeRow(tensor.data(), GgufTensorType.BF16, 0, length, values);
    return values;
  }
}
