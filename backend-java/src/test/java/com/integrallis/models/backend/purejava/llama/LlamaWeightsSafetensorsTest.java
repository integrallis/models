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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.huggingface.Qwen2HuggingFaceConfig;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.safetensors.SyntheticSafetensorsBuilder;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class LlamaWeightsSafetensorsTest {

  @Test
  void loadsOfficialQwen2NamesAsMappedBfloat16Weights(@TempDir Path directory) throws IOException {
    Path artifact = writeModel(directory, true, false);

    try (Arena arena = Arena.ofConfined()) {
      LlamaWeights weights =
          LlamaWeights.fromQwen2Safetensors(
              new SafetensorsTensorSource(SafetensorsBundle.open(artifact, arena)), config(true));
      float[] embedding = new float[4];
      weights.embedToken(1, embedding);
      LlamaWeights.LayerWeights layer = weights.layer(0);

      assertThat(embedding).containsExactly(4.0f, 5.0f, 6.0f, 7.0f);
      assertThat(weights.outputNormWeight()).containsExactly(1.0f, 2.0f, 3.0f, 4.0f);
      assertThat(weights.outputType()).isEqualTo(GgufTensorType.BF16);
      assertThat(weights.outputSegment().isReadOnly()).isTrue();
      assertThat(layer.attentionNorm()).containsExactly(2.0f, 3.0f, 4.0f, 5.0f);
      assertThat(layer.ffnNorm()).containsExactly(3.0f, 4.0f, 5.0f, 6.0f);
      assertThat(layer.qBias()).containsExactly(1.0f, 2.0f, 3.0f, 4.0f);
      assertThat(layer.kBias()).containsExactly(5.0f, 6.0f);
      assertThat(layer.vBias()).containsExactly(7.0f, 8.0f);
      assertThat(layer.qNorm()).isEmpty();
      assertThat(layer.kNorm()).isEmpty();
      assertThat(layer.attentionPostNorm()).isEmpty();
      assertThat(layer.ffnPostNorm()).isEmpty();
      assertThat(
              List.of(
                  layer.wqType(),
                  layer.wkType(),
                  layer.wvType(),
                  layer.woType(),
                  layer.ffnGateType(),
                  layer.ffnUpType(),
                  layer.ffnDownType()))
          .containsOnly(GgufTensorType.BF16);
    }
  }

  @Test
  void rejectsAProjectionWhoseHuggingFaceShapeDoesNotMatchTheConfig(@TempDir Path directory)
      throws IOException {
    Path artifact = writeModel(directory, true, true);

    try (Arena arena = Arena.ofConfined()) {
      SafetensorsTensorSource source =
          new SafetensorsTensorSource(SafetensorsBundle.open(artifact, arena));

      assertThatThrownBy(() -> LlamaWeights.fromQwen2Safetensors(source, config(true)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("model.layers.0.self_attn.q_proj.weight")
          .hasMessageContaining("[4, 4]");
    }
  }

  @Test
  void requiresAnLmHeadWhenTheConfigDoesNotTieWordEmbeddings(@TempDir Path directory)
      throws IOException {
    Path artifact = writeModel(directory, true, false);

    try (Arena arena = Arena.ofConfined()) {
      SafetensorsTensorSource source =
          new SafetensorsTensorSource(SafetensorsBundle.open(artifact, arena));

      assertThatThrownBy(() -> LlamaWeights.fromQwen2Safetensors(source, config(false)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("lm_head.weight");
    }
  }

  private static Path writeModel(Path directory, boolean tied, boolean invalidQueryShape)
      throws IOException {
    SyntheticSafetensorsBuilder builder = new SyntheticSafetensorsBuilder();
    addBf16(builder, "model.embed_tokens.weight", new long[] {6, 4}, sequence(24, 0));
    addBf16(builder, "model.norm.weight", new long[] {4}, sequence(4, 1));
    if (!tied) {
      addBf16(builder, "lm_head.weight", new long[] {6, 4}, sequence(24, 10));
    }
    addBf16(builder, "model.layers.0.input_layernorm.weight", new long[] {4}, sequence(4, 2));
    addBf16(
        builder,
        "model.layers.0.self_attn.q_proj.weight",
        invalidQueryShape ? new long[] {5, 4} : new long[] {4, 4},
        sequence(invalidQueryShape ? 20 : 16, 1));
    addBf16(builder, "model.layers.0.self_attn.q_proj.bias", new long[] {4}, sequence(4, 1));
    addBf16(builder, "model.layers.0.self_attn.k_proj.weight", new long[] {2, 4}, sequence(8, 2));
    addBf16(builder, "model.layers.0.self_attn.k_proj.bias", new long[] {2}, sequence(2, 5));
    addBf16(builder, "model.layers.0.self_attn.v_proj.weight", new long[] {2, 4}, sequence(8, 3));
    addBf16(builder, "model.layers.0.self_attn.v_proj.bias", new long[] {2}, sequence(2, 7));
    addBf16(builder, "model.layers.0.self_attn.o_proj.weight", new long[] {4, 4}, sequence(16, 4));
    addBf16(
        builder, "model.layers.0.post_attention_layernorm.weight", new long[] {4}, sequence(4, 3));
    addBf16(builder, "model.layers.0.mlp.gate_proj.weight", new long[] {6, 4}, sequence(24, 5));
    addBf16(builder, "model.layers.0.mlp.up_proj.weight", new long[] {6, 4}, sequence(24, 6));
    addBf16(builder, "model.layers.0.mlp.down_proj.weight", new long[] {4, 6}, sequence(24, 7));
    Path artifact = directory.resolve("model.safetensors");
    Files.write(artifact, builder.build());
    return artifact;
  }

  private static Qwen2HuggingFaceConfig config(boolean tied) {
    LlamaConfig model =
        new LlamaConfig(
            DecoderArchitecture.QWEN2,
            4,
            1,
            2,
            1,
            2,
            2,
            6,
            16,
            6,
            10_000.0f,
            1.0f,
            10_000.0f,
            1.0e-6f,
            0,
            6,
            0.0f);
    return new Qwen2HuggingFaceConfig(
        model, List.of("Qwen2ForCausalLM"), "silu", "bfloat16", tied, 2, 3);
  }

  private static void addBf16(
      SyntheticSafetensorsBuilder builder, String name, long[] shape, float[] values) {
    int[] bytes = new int[values.length * Short.BYTES];
    for (int index = 0; index < values.length; index++) {
      int bits = Float.floatToRawIntBits(values[index]) >>> Short.SIZE;
      bytes[index * Short.BYTES] = bits & 0xff;
      bytes[index * Short.BYTES + 1] = bits >>> Byte.SIZE;
    }
    builder.add(name, "BF16", shape, bytes);
  }

  private static float[] sequence(int count, int start) {
    float[] values = new float[count];
    for (int index = 0; index < count; index++) {
      values[index] = start + index;
    }
    return values;
  }
}
