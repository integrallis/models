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
      float[] ffnPostNorm) {}

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
      GgufTensorData wq = file.getTensor(prefix + "attn_q.weight");
      GgufTensorData wk = file.getTensor(prefix + "attn_k.weight");
      GgufTensorData wv = file.getTensor(prefix + "attn_v.weight");
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
      GgufTensorData ffnGate = file.getTensor(prefix + "ffn_gate.weight");
      GgufTensorData ffnUp = file.getTensor(prefix + "ffn_up.weight");
      GgufTensorData ffnDown = file.getTensor(prefix + "ffn_down.weight");
      float[] ffnPostNorm =
          config.usesPostFfnNorm()
              ? loadF32Tensor(file, prefix + "post_ffw_norm.weight")
              : new float[0];

      layers[i] =
          new LayerWeights(
              attnNorm,
              wq.dataSegment(),
              wq.type(),
              qBias,
              qNorm,
              wk.dataSegment(),
              wk.type(),
              kBias,
              kNorm,
              wv.dataSegment(),
              wv.type(),
              vBias,
              wo.dataSegment(),
              wo.type(),
              attentionPostNorm,
              ffnNorm,
              ffnGate.dataSegment(),
              ffnGate.type(),
              ffnUp.dataSegment(),
              ffnUp.type(),
              ffnDown.dataSegment(),
              ffnDown.type(),
              ffnPostNorm);
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
              new float[0]);
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
