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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LlamaWeightsHuggingFaceGgufTest {

  @Test
  void loadsQwenWeightsStoredUnderNativeHuggingFaceNames() {
    LlamaConfig config = config();
    GgufFile file = nativeQwenGguf(config);

    LlamaWeights weights = LlamaWeights.fromHuggingFaceNamedGguf(file, config);

    assertThat(weights.outputType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.outputNormWeight()).hasSize(config.embeddingDim());
    assertThat(weights.layer(0).qNorm()).hasSize(config.headDim());
    assertThat(weights.layer(0).kNorm()).hasSize(config.headDim());
    assertThat(weights.layer(0).wqType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.layer(0).ffnDownType()).isEqualTo(GgufTensorType.F32);
  }

  @Test
  void preparesHuggingFaceNamedBf16MatricesAsOwnedF32ExecutionWeights() {
    LlamaConfig config = config();
    GgufFile file = nativeQwenGguf(config, GgufTensorType.BF16);

    LlamaWeights weights = LlamaWeights.fromHuggingFaceNamedGguf(file, config);

    assertThat(weights.outputType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.layer(0).wqType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.layer(0).wkType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.layer(0).wvType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.layer(0).woType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.layer(0).ffnGateType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.layer(0).ffnUpType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.layer(0).ffnDownType()).isEqualTo(GgufTensorType.F32);
  }

  private static LlamaConfig config() {
    return new LlamaConfig(
        DecoderArchitecture.QWEN3,
        4,
        1,
        2,
        1,
        2,
        2,
        8,
        16,
        6,
        10_000.0f,
        1.0f,
        10_000.0f,
        1.0e-6f,
        0,
        1,
        0.0f);
  }

  private static GgufFile nativeQwenGguf(LlamaConfig config) {
    return nativeQwenGguf(config, GgufTensorType.F32);
  }

  private static GgufFile nativeQwenGguf(LlamaConfig config, GgufTensorType matrixType) {
    SyntheticGgufBuilder builder = new SyntheticGgufBuilder();
    tensor(
        builder,
        "model.embed_tokens.weight",
        matrixType,
        config.embeddingDim(),
        config.vocabSize());
    tensor(builder, "model.norm.weight", config.embeddingDim());
    tensor(builder, "lm_head.weight", matrixType, config.embeddingDim(), config.vocabSize());
    String prefix = "model.layers.0.";
    tensor(builder, prefix + "input_layernorm.weight", config.embeddingDim());
    tensor(
        builder,
        prefix + "self_attn.q_proj.weight",
        matrixType,
        config.embeddingDim(),
        config.queryDim());
    tensor(
        builder,
        prefix + "self_attn.k_proj.weight",
        matrixType,
        config.embeddingDim(),
        config.keyDim());
    tensor(
        builder,
        prefix + "self_attn.v_proj.weight",
        matrixType,
        config.embeddingDim(),
        config.valueDim());
    tensor(
        builder,
        prefix + "self_attn.o_proj.weight",
        matrixType,
        config.attentionOutputDim(),
        config.embeddingDim());
    tensor(builder, prefix + "self_attn.q_norm.weight", config.headDim());
    tensor(builder, prefix + "self_attn.k_norm.weight", config.headDim());
    tensor(builder, prefix + "post_attention_layernorm.weight", config.embeddingDim());
    tensor(
        builder,
        prefix + "mlp.gate_proj.weight",
        matrixType,
        config.embeddingDim(),
        config.hiddenDim());
    tensor(
        builder,
        prefix + "mlp.up_proj.weight",
        matrixType,
        config.embeddingDim(),
        config.hiddenDim());
    tensor(
        builder,
        prefix + "mlp.down_proj.weight",
        matrixType,
        config.hiddenDim(),
        config.embeddingDim());
    return GgufParser.parseSegment(MemorySegment.ofArray(builder.build()));
  }

  private static void tensor(SyntheticGgufBuilder builder, String name, long... shape) {
    tensor(builder, name, GgufTensorType.F32, shape);
  }

  private static void tensor(
      SyntheticGgufBuilder builder, String name, GgufTensorType type, long... shape) {
    long elements = 1;
    for (long dimension : shape) {
      elements = Math.multiplyExact(elements, dimension);
    }
    builder.addTensor(
        name,
        type,
        shape,
        new byte
            [Math.toIntExact(Math.multiplyExact(elements, type.typeSize() / type.blockSize()))]);
  }
}
