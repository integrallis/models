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
import static org.assertj.core.api.Assertions.within;

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
class GptOssForwardPassTest {

  private static final int HIDDEN = 32;
  private static final int HEADS = 2;
  private static final int HEAD_DIM = 32;
  private static final int QUERY = HEADS * HEAD_DIM;
  private static final int KV = HEAD_DIM;
  private static final int VOCAB = 4;
  private static final float EPSILON = 1.0e-5f;
  private static final float[] SINKS = {-0.5f, 0.25f};

  @Test
  void executesACompleteTokenAgainstAnIndependentScalarOracle(@TempDir Path directory)
      throws IOException {
    GptOssHuggingFaceConfig config = config();
    GptOssWeights weights = GptOssWeights.load(source(directory), config);
    GptOssForwardPass forwardPass = new GptOssForwardPass(config, weights, 8);
    GptOssForwardPass.Session session = forwardPass.openSession();

    float[] actual = forwardPass.forward(session, 0, 0);

    assertThat(actual).containsExactly(referenceFirstToken(), within(1.0e-6f));
    assertThat(session.checkpoint()).isEqualTo(1);
  }

  @Test
  void ownsIndependentSequentialStateAndCanResetIt(@TempDir Path directory) throws IOException {
    GptOssHuggingFaceConfig config = config();
    GptOssForwardPass forwardPass =
        new GptOssForwardPass(config, GptOssWeights.load(source(directory), config), 8);
    GptOssForwardPass.Session first = forwardPass.openSession();
    GptOssForwardPass.Session second = forwardPass.openSession();

    float[] firstInitial = forwardPass.forward(first, 0, 0);
    assertThat(forwardPass.forward(second, 0, 0)).containsExactly(firstInitial);
    float[] firstContinuation = forwardPass.forward(first, 1, 1);
    assertThat(firstContinuation).isNotEqualTo(firstInitial);
    assertThat(second.checkpoint()).isEqualTo(1);
    assertThatThrownBy(() -> forwardPass.forward(second, 1, 2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sequential")
        .hasMessageContaining("expected 1");
    for (int position = 2; position < 6; position++) {
      forwardPass.forward(first, position % VOCAB, position);
    }
    assertThat(first.checkpoint()).isEqualTo(6);

    forwardPass.reset(first);

    assertThat(first.checkpoint()).isZero();
    assertThat(forwardPass.forward(first, 0, 0)).containsExactly(firstInitial);
  }

  private static float[] referenceFirstToken() {
    float[] embedding = embedding(0);
    float[] normalized = rmsNorm(embedding);
    float ropeMagnitude = 0.1f * (float) Math.log(2.0f) + 1.0f;
    float scoreBase = 0.0f;
    for (float value : normalized) {
      scoreBase += (ropeMagnitude * value) * (ropeMagnitude * value);
    }
    scoreBase /= (float) Math.sqrt(HEAD_DIM);

    float headZeroProbability = sigmoid(scoreBase - SINKS[0]);
    float headOneProbability = sigmoid(0.5f * scoreBase - SINKS[1]);
    float[] residual = embedding.clone();
    for (int index = 0; index < HIDDEN; index++) {
      float attention =
          0.5f * headZeroProbability * normalized[index]
              + 0.25f * headOneProbability * normalized[index];
      residual[index] += attention;
    }

    for (int index = 0; index < HIDDEN; index++) {
      residual[index] += expertDownBias(index);
    }
    float[] output = rmsNorm(residual);
    return Arrays.copyOf(output, VOCAB);
  }

  private static float sigmoid(float value) {
    return (float) (1.0 / (1.0 + Math.exp(-value)));
  }

  private static float[] rmsNorm(float[] values) {
    float sumSquares = 0.0f;
    for (float value : values) {
      sumSquares += value * value;
    }
    float scale = (float) (1.0 / Math.sqrt(sumSquares / values.length + EPSILON));
    float[] output = new float[values.length];
    for (int index = 0; index < values.length; index++) {
      output[index] = values[index] * scale;
    }
    return output;
  }

  private static GptOssHuggingFaceConfig config() {
    return new GptOssHuggingFaceConfig(
        List.of("GptOssForCausalLM"),
        HIDDEN,
        1,
        HEADS,
        1,
        HEAD_DIM,
        VOCAB,
        8,
        4,
        HIDDEN,
        1,
        1,
        2,
        EPSILON,
        10_000.0f,
        2.0f,
        32.0f,
        1.0f,
        4,
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

  private static SafetensorsTensorSource source(Path directory) throws IOException {
    String prefix = "model.layers.0.";
    SyntheticSafetensorsBuilder builder =
        new SyntheticSafetensorsBuilder()
            .add(
                "model.embed_tokens.weight",
                "BF16",
                new long[] {VOCAB, HIDDEN},
                bf16(embeddingMatrix()))
            .add("model.norm.weight", "BF16", new long[] {HIDDEN}, bf16(ones(HIDDEN)))
            .add("lm_head.weight", "BF16", new long[] {VOCAB, HIDDEN}, bf16(lmHead()))
            .add(prefix + "input_layernorm.weight", "BF16", new long[] {HIDDEN}, bf16(ones(HIDDEN)))
            .add(
                prefix + "self_attn.q_proj.weight",
                "BF16",
                new long[] {QUERY, HIDDEN},
                bf16(queryProjection()))
            .add(
                prefix + "self_attn.k_proj.weight",
                "BF16",
                new long[] {KV, HIDDEN},
                bf16(identity(HIDDEN)))
            .add(
                prefix + "self_attn.v_proj.weight",
                "BF16",
                new long[] {KV, HIDDEN},
                bf16(identity(HIDDEN)))
            .add(
                prefix + "self_attn.o_proj.weight",
                "BF16",
                new long[] {HIDDEN, QUERY},
                bf16(outputProjection()))
            .add(
                prefix + "self_attn.q_proj.bias",
                "BF16",
                new long[] {QUERY},
                bf16(new float[QUERY]))
            .add(prefix + "self_attn.k_proj.bias", "BF16", new long[] {KV}, bf16(new float[KV]))
            .add(prefix + "self_attn.v_proj.bias", "BF16", new long[] {KV}, bf16(new float[KV]))
            .add(
                prefix + "self_attn.o_proj.bias",
                "BF16",
                new long[] {HIDDEN},
                bf16(new float[HIDDEN]))
            .add(prefix + "self_attn.sinks", "BF16", new long[] {HEADS}, bf16(SINKS))
            .add(
                prefix + "post_attention_layernorm.weight",
                "BF16",
                new long[] {HIDDEN},
                bf16(ones(HIDDEN)))
            .add(
                prefix + "mlp.router.weight",
                "BF16",
                new long[] {1, HIDDEN},
                bf16(new float[HIDDEN]))
            .add(prefix + "mlp.router.bias", "BF16", new long[] {1}, bf16(new float[1]))
            .add(
                prefix + "mlp.experts.gate_up_proj_blocks",
                "U8",
                new long[] {1, 2L * HIDDEN, 1, 16},
                new int[2 * HIDDEN * 16])
            .add(
                prefix + "mlp.experts.gate_up_proj_scales",
                "U8",
                new long[] {1, 2L * HIDDEN, 1},
                filled(2 * HIDDEN, 127))
            .add(
                prefix + "mlp.experts.gate_up_proj_bias",
                "BF16",
                new long[] {1, 2L * HIDDEN},
                bf16(new float[2 * HIDDEN]))
            .add(
                prefix + "mlp.experts.down_proj_blocks",
                "U8",
                new long[] {1, HIDDEN, 1, 16},
                new int[HIDDEN * 16])
            .add(
                prefix + "mlp.experts.down_proj_scales",
                "U8",
                new long[] {1, HIDDEN, 1},
                filled(HIDDEN, 127))
            .add(
                prefix + "mlp.experts.down_proj_bias",
                "BF16",
                new long[] {1, HIDDEN},
                bf16(expertDownBias()));
    Path artifact = Files.write(directory.resolve("model.safetensors"), builder.build());
    return new SafetensorsTensorSource(SafetensorsBundle.open(artifact, Arena.global()));
  }

  private static float[] embeddingMatrix() {
    float[] matrix = new float[VOCAB * HIDDEN];
    for (int token = 0; token < VOCAB; token++) {
      System.arraycopy(embedding(token), 0, matrix, token * HIDDEN, HIDDEN);
    }
    return matrix;
  }

  private static float[] embedding(int token) {
    float[] values = new float[HIDDEN];
    for (int index = 0; index < HIDDEN; index++) {
      values[index] = (token + 1) * 0.25f + (index % 4) * 0.125f;
    }
    return values;
  }

  private static float[] queryProjection() {
    float[] matrix = new float[QUERY * HIDDEN];
    for (int index = 0; index < HIDDEN; index++) {
      matrix[index * HIDDEN + index] = 1.0f;
      matrix[(HIDDEN + index) * HIDDEN + index] = 0.5f;
    }
    return matrix;
  }

  private static float[] outputProjection() {
    float[] matrix = new float[HIDDEN * QUERY];
    for (int index = 0; index < HIDDEN; index++) {
      matrix[index * QUERY + index] = 0.5f;
      matrix[index * QUERY + HIDDEN + index] = 0.25f;
    }
    return matrix;
  }

  private static float[] lmHead() {
    float[] matrix = new float[VOCAB * HIDDEN];
    for (int token = 0; token < VOCAB; token++) {
      matrix[token * HIDDEN + token] = 1.0f;
    }
    return matrix;
  }

  private static float[] identity(int size) {
    float[] matrix = new float[size * size];
    for (int index = 0; index < size; index++) {
      matrix[index * size + index] = 1.0f;
    }
    return matrix;
  }

  private static float[] ones(int size) {
    float[] values = new float[size];
    Arrays.fill(values, 1.0f);
    return values;
  }

  private static float[] expertDownBias() {
    float[] values = new float[HIDDEN];
    for (int index = 0; index < HIDDEN; index++) {
      values[index] = expertDownBias(index);
    }
    return values;
  }

  private static float expertDownBias(int index) {
    return 0.03125f * (1 + index % 4);
  }

  private static int[] bf16(float[] values) {
    int[] bytes = new int[values.length * Short.BYTES];
    for (int index = 0; index < values.length; index++) {
      int bits = Float.floatToRawIntBits(values[index]) >>> Short.SIZE;
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
