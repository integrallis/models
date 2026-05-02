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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/** Holds references to weight tensors from a parsed GGUF file for a Llama-family model. */
public final class LlamaWeights {

  private final float[] tokenEmbedding;
  private final float[] outputNormWeight;
  private final float[] outputWeight;
  private final LayerWeights[] layers;

  /** Per-layer weight tensors. */
  public record LayerWeights(
      float[] attentionNorm,
      MemorySegment wq,
      GgufTensorType wqType,
      MemorySegment wk,
      GgufTensorType wkType,
      MemorySegment wv,
      GgufTensorType wvType,
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
      float[] tokenEmbedding,
      float[] outputNormWeight,
      float[] outputWeight,
      LayerWeights[] layers) {
    this.tokenEmbedding = tokenEmbedding;
    this.outputNormWeight = outputNormWeight;
    this.outputWeight = outputWeight;
    this.layers = layers;
  }

  /** Loads weights from a parsed GGUF file using the standard Llama tensor naming convention. */
  public static LlamaWeights fromGgufFile(GgufFile file, LlamaConfig config) {
    float[] tokenEmbed = loadF32Tensor(file, "token_embd.weight");
    float[] outputNorm = loadF32Tensor(file, "output_norm.weight");

    float[] output;
    try {
      output = loadF32Tensor(file, "output.weight");
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
              wk.dataSegment(),
              wk.type(),
              wv.dataSegment(),
              wv.type(),
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

    return new LlamaWeights(tokenEmbed, outputNorm, output, layers);
  }

  public float[] tokenEmbedding() {
    return tokenEmbedding;
  }

  public float[] outputNormWeight() {
    return outputNormWeight;
  }

  public float[] outputWeight() {
    return outputWeight;
  }

  public LayerWeights layer(int i) {
    return layers[i];
  }

  private static float[] loadF32Tensor(GgufFile file, String name) {
    GgufTensorData tensor = file.getTensor(name);
    if (tensor.type() != GgufTensorType.F32) {
      throw new IllegalArgumentException(
          "Expected F32 tensor for " + name + " but got " + tensor.type());
    }
    int count = (int) tensor.info().elementCount();
    float[] result = new float[count];
    MemorySegment seg = tensor.dataSegment();
    for (int i = 0; i < count; i++) {
      result[i] =
          seg.get(
              ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), (long) i * 4);
    }
    return result;
  }
}
