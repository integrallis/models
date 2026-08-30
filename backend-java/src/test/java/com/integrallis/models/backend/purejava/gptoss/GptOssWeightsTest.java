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
package com.integrallis.models.backend.purejava.gptoss;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.safetensors.SyntheticSafetensorsBuilder;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class GptOssWeightsTest {

  private static final int HIDDEN = 32;
  private static final int QUERY = 64;
  private static final int KV = 32;
  private static final int VOCAB = 32;

  @Test
  void mapsTheCompleteOfficialTensorContractWithoutExpandingMatrices(@TempDir Path directory)
      throws IOException {
    GptOssWeights weights = GptOssWeights.load(source(directory, false), config());

    assertThat(weights.tokenEmbedding().rows()).isEqualTo(VOCAB);
    assertThat(weights.tokenEmbedding().columns()).isEqualTo(HIDDEN);
    assertThat(weights.tokenEmbedding().value(0, 0)).isEqualTo(1.0f);
    assertThat(weights.output().rows()).isEqualTo(VOCAB);
    assertThat(weights.outputNorm()).startsWith(2.0f).hasSize(HIDDEN);

    GptOssWeights.Layer layer = weights.layer(0);
    assertThat(layer.attentionNorm()).startsWith(3.0f).hasSize(HIDDEN);
    assertThat(layer.query().rows()).isEqualTo(QUERY);
    assertThat(layer.query().columns()).isEqualTo(HIDDEN);
    assertThat(layer.key().rows()).isEqualTo(KV);
    assertThat(layer.value().rows()).isEqualTo(KV);
    assertThat(layer.output().rows()).isEqualTo(HIDDEN);
    assertThat(layer.output().columns()).isEqualTo(QUERY);
    assertThat(layer.queryBias()).startsWith(8.0f).hasSize(QUERY);
    assertThat(layer.keyBias()).startsWith(9.0f).hasSize(KV);
    assertThat(layer.valueBias()).startsWith(10.0f).hasSize(KV);
    assertThat(layer.outputBias()).startsWith(11.0f).hasSize(HIDDEN);
    assertThat(layer.sinks()).startsWith(12.0f).hasSize(2);
    assertThat(layer.postAttentionNorm()).startsWith(13.0f).hasSize(HIDDEN);
    assertThat(layer.router().rows()).isEqualTo(1);
    assertThat(layer.router().columns()).isEqualTo(HIDDEN);
    assertThat(layer.routerBias()).containsExactly(15.0f);
    assertThat(layer.experts().expertCount()).isEqualTo(1);
    assertThat(layer.experts().expert(0).gateUp().rows()).isEqualTo(2 * HIDDEN);
    assertThat(layer.experts().expert(0).down().rows()).isEqualTo(HIDDEN);
  }

  @Test
  void rejectsADenseProjectionThatDoesNotMatchTheConfig(@TempDir Path directory)
      throws IOException {
    SafetensorsTensorSource source = source(directory, true);

    assertThatThrownBy(() -> GptOssWeights.load(source, config()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("self_attn.q_proj.weight")
        .hasMessageContaining("[64, 32]");
  }

  private static GptOssHuggingFaceConfig config() {
    return new GptOssHuggingFaceConfig(
        List.of("GptOssForCausalLM"),
        HIDDEN,
        1,
        2,
        1,
        32,
        VOCAB,
        256,
        128,
        HIDDEN,
        1,
        1,
        64,
        1.0e-5f,
        10_000.0f,
        2.0f,
        32.0f,
        1.0f,
        128,
        7.0f,
        1.702f,
        "silu",
        "mxfp4",
        true,
        false,
        2,
        0,
        List.of("sliding_attention"));
  }

  private static SafetensorsTensorSource source(Path directory, boolean invalidQuery)
      throws IOException {
    String prefix = "model.layers.0.";
    SyntheticSafetensorsBuilder builder =
        new SyntheticSafetensorsBuilder()
            .add(
                "model.embed_tokens.weight",
                "BF16",
                new long[] {VOCAB, HIDDEN},
                bf16(VOCAB * HIDDEN, 1))
            .add("model.norm.weight", "BF16", new long[] {HIDDEN}, bf16(HIDDEN, 2))
            .add("lm_head.weight", "BF16", new long[] {VOCAB, HIDDEN}, bf16(VOCAB * HIDDEN, 3))
            .add(prefix + "input_layernorm.weight", "BF16", new long[] {HIDDEN}, bf16(HIDDEN, 3))
            .add(
                prefix + "self_attn.q_proj.weight",
                "BF16",
                invalidQuery ? new long[] {HIDDEN, QUERY} : new long[] {QUERY, HIDDEN},
                bf16(QUERY * HIDDEN, 4))
            .add(
                prefix + "self_attn.k_proj.weight",
                "BF16",
                new long[] {KV, HIDDEN},
                bf16(KV * HIDDEN, 5))
            .add(
                prefix + "self_attn.v_proj.weight",
                "BF16",
                new long[] {KV, HIDDEN},
                bf16(KV * HIDDEN, 6))
            .add(
                prefix + "self_attn.o_proj.weight",
                "BF16",
                new long[] {HIDDEN, QUERY},
                bf16(HIDDEN * QUERY, 7))
            .add(prefix + "self_attn.q_proj.bias", "BF16", new long[] {QUERY}, bf16(QUERY, 8))
            .add(prefix + "self_attn.k_proj.bias", "BF16", new long[] {KV}, bf16(KV, 9))
            .add(prefix + "self_attn.v_proj.bias", "BF16", new long[] {KV}, bf16(KV, 10))
            .add(prefix + "self_attn.o_proj.bias", "BF16", new long[] {HIDDEN}, bf16(HIDDEN, 11))
            .add(prefix + "self_attn.sinks", "BF16", new long[] {2}, bf16(2, 12))
            .add(
                prefix + "post_attention_layernorm.weight",
                "BF16",
                new long[] {HIDDEN},
                bf16(HIDDEN, 13))
            .add(prefix + "mlp.router.weight", "BF16", new long[] {1, HIDDEN}, bf16(HIDDEN, 14))
            .add(prefix + "mlp.router.bias", "BF16", new long[] {1}, bf16(1, 15))
            .add(
                prefix + "mlp.experts.gate_up_proj_blocks",
                "U8",
                new long[] {1, 2L * HIDDEN, 1, 16},
                filled(2 * HIDDEN * 16, 0x21))
            .add(
                prefix + "mlp.experts.gate_up_proj_scales",
                "U8",
                new long[] {1, 2L * HIDDEN, 1},
                filled(2 * HIDDEN, 127))
            .add(
                prefix + "mlp.experts.gate_up_proj_bias",
                "BF16",
                new long[] {1, 2L * HIDDEN},
                bf16(2 * HIDDEN, 16))
            .add(
                prefix + "mlp.experts.down_proj_blocks",
                "U8",
                new long[] {1, HIDDEN, 1, 16},
                filled(HIDDEN * 16, 0x32))
            .add(
                prefix + "mlp.experts.down_proj_scales",
                "U8",
                new long[] {1, HIDDEN, 1},
                filled(HIDDEN, 127))
            .add(
                prefix + "mlp.experts.down_proj_bias",
                "BF16",
                new long[] {1, HIDDEN},
                bf16(HIDDEN, 17));
    Path artifact = Files.write(directory.resolve("model.safetensors"), builder.build());
    return new SafetensorsTensorSource(SafetensorsBundle.open(artifact, Arena.global()));
  }

  private static int[] bf16(int count, int start) {
    int[] bytes = new int[count * Short.BYTES];
    for (int index = 0; index < count; index++) {
      int bits = Float.floatToRawIntBits(start + index % 4) >>> Short.SIZE;
      bytes[index * Short.BYTES] = bits & 0xff;
      bytes[index * Short.BYTES + 1] = bits >>> Byte.SIZE;
    }
    return bytes;
  }

  private static int[] filled(int count, int value) {
    int[] values = new int[count];
    Arrays.fill(values, value);
    return values;
  }
}
