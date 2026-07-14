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
import com.integrallis.models.backend.purejava.quant.F16Dequantizer;
import com.integrallis.models.backend.purejava.quant.Q4_0Dequantizer;
import com.integrallis.models.backend.purejava.quant.Q8_0Dequantizer;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/** Holds references to weight tensors from a parsed GGUF file for a Llama-family model. */
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
      float[] ffnNorm,
      MemorySegment ffnGate,
      GgufTensorType ffnGateType,
      MemorySegment ffnUp,
      GgufTensorType ffnUpType,
      MemorySegment ffnDown,
      GgufTensorType ffnDownType) {}

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
      float[] ffnNorm = loadF32Tensor(file, prefix + "ffn_norm.weight");
      GgufTensorData ffnGate = file.getTensor(prefix + "ffn_gate.weight");
      GgufTensorData ffnUp = file.getTensor(prefix + "ffn_up.weight");
      GgufTensorData ffnDown = file.getTensor(prefix + "ffn_down.weight");

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
              ffnNorm,
              ffnGate.dataSegment(),
              ffnGate.type(),
              ffnUp.dataSegment(),
              ffnUp.type(),
              ffnDown.dataSegment(),
              ffnDown.type());
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
    switch (type) {
      case F32 -> {
        long offset = (long) row * cols * 4;
        for (int i = 0; i < cols; i++) {
          out[i] =
              segment.get(
                  ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                  offset + (long) i * 4);
        }
      }
      case F16 -> {
        long offset = (long) row * cols * 2;
        new F16Dequantizer().dequantize(segment, offset, out, 0, cols);
      }
      case Q4_0 -> {
        // Q4_0: 32 values per block, 18 bytes per block
        int blocksPerRow = cols / 32;
        long bytesPerRow = (long) blocksPerRow * 18;
        long offset = (long) row * bytesPerRow;
        new Q4_0Dequantizer().dequantize(segment, offset, out, 0, cols);
      }
      case Q4_K -> {
        int blocksPerRow = cols / type.blockSize();
        long bytesPerRow = (long) blocksPerRow * type.typeSize();
        long offset = (long) row * bytesPerRow;
        VectorUtil.ggufQ4_KDequantize(segment, offset, out, 0, cols);
      }
      case Q5_K -> {
        int blocksPerRow = cols / type.blockSize();
        long bytesPerRow = (long) blocksPerRow * type.typeSize();
        long offset = (long) row * bytesPerRow;
        VectorUtil.ggufQ5_KDequantize(segment, offset, out, 0, cols);
      }
      case Q6_K -> {
        int blocksPerRow = cols / type.blockSize();
        long bytesPerRow = (long) blocksPerRow * type.typeSize();
        long offset = (long) row * bytesPerRow;
        VectorUtil.ggufQ6_KDequantize(segment, offset, out, 0, cols);
      }
      case Q8_0 -> {
        // Q8_0: 32 values per block, 34 bytes per block
        int blocksPerRow = cols / 32;
        long bytesPerRow = (long) blocksPerRow * 34;
        long offset = (long) row * bytesPerRow;
        new Q8_0Dequantizer().dequantize(segment, offset, out, 0, cols);
      }
      default -> throw new IllegalArgumentException("Unsupported embedding tensor type: " + type);
    }
  }

  /**
   * Loads a tensor and dequantizes it to F32 if needed. Supports F32, F16, Q4_0, and Q8_0 source
   * formats.
   */
  private static float[] loadF32Tensor(GgufFile file, String name) {
    GgufTensorData tensor = file.getTensor(name);
    int count = (int) tensor.info().elementCount();
    float[] result = new float[count];
    MemorySegment seg = tensor.dataSegment();

    switch (tensor.type()) {
      case F32 -> {
        for (int i = 0; i < count; i++) {
          result[i] =
              seg.get(
                  ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                  (long) i * 4);
        }
      }
      case F16 -> new F16Dequantizer().dequantize(seg, 0, result, 0, count);
      case Q4_0 -> new Q4_0Dequantizer().dequantize(seg, 0, result, 0, count);
      case Q8_0 -> new Q8_0Dequantizer().dequantize(seg, 0, result, 0, count);
      default ->
          throw new IllegalArgumentException(
              "Unsupported tensor type for dequantization: " + tensor.type() + " in " + name);
    }

    return result;
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
}
