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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Parity test: forwardEmbeddings with embedToken(token) as input should match forwardToken(token)
 * exactly, since both paths run the same decoder blocks.
 */
@Tag("unit")
class Qwen35ForwardEmbeddingsParityTest {

  private static final int DIM = 8;
  private static final int HEAD_DIM = 4;
  private static final int HIDDEN_DIM = 8;
  private static final int VOCAB = 8;
  private static final int KV_HEADS = 1;
  private static final int CONV_DIM = 2 * HEAD_DIM + HEAD_DIM * KV_HEADS;

  @Test
  void embeddingPathMatchesTokenPathWhenStartingFromSameEmbedding(@TempDir Path directory)
      throws Exception {
    Path model = writeToyModel(directory);

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(model, arena);
      var graph = Qwen35ForwardPass.fromGgufFile(file);

      // Token path: forward token 0 from a fresh session
      float[] tokenLogits = graph.forwardFresh(0);

      // Embedding path: forward the SAME token 0 embedding via embedding injection
      float[] token0Embedding = extractTokenEmbedding(file, 0);
      var embSession = graph.openSession(8);
      float[] embLogits = graph.forwardEmbeddings(embSession, token0Embedding, 0);

      // Both paths start from position 0 on a fresh cache, so logits should match
      assertThat(embLogits).hasSize(VOCAB);
      assertThat(allFinite(embLogits)).isTrue();
      assertThat(embLogits).containsExactly(tokenLogits);
    }
  }

  @Test
  void embeddingPathRejectsWrongEmbeddingDimension(@TempDir Path directory) throws Exception {
    Path model = writeToyModel(directory);

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(model, arena);
      var graph = Qwen35ForwardPass.fromGgufFile(file);
      var session = graph.openSession(8);

      float[] wrongDimension = new float[DIM + 1];
      assertThatThrownBy(() -> graph.forwardEmbeddings(session, wrongDimension, 0))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("embeddings length");
    }
  }

  @Test
  void embeddingPathAdvancesSessionCheckpoint(@TempDir Path directory) throws Exception {
    Path model = writeToyModel(directory);

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(model, arena);
      var graph = Qwen35ForwardPass.fromGgufFile(file);
      var session = graph.openSession(8);

      assertThat(session.checkpoint()).isEqualTo(0);

      float[] embeddings = new float[DIM];
      graph.forwardEmbeddings(session, embeddings, 0);

      assertThat(session.checkpoint()).isEqualTo(1);
    }
  }

  private static float[] extractTokenEmbedding(
      com.integrallis.models.backend.purejava.gguf.GgufFile file, int token) {
    var tensor = file.getTensor("token_embd.weight");
    float[] emb = new float[DIM];
    com.integrallis.models.backend.purejava.gguf.GgufTensorValues.dequantizeRow(
        tensor.dataSegment(), tensor.type(), token, DIM, emb);
    return emb;
  }

  private static float[] softmax(float[] logits) {
    float max = logits[0];
    for (float v : logits) if (v > max) max = v;
    float sum = 0;
    float[] result = new float[logits.length];
    for (int i = 0; i < logits.length; i++) {
      result[i] = (float) Math.exp(logits[i] - max);
      sum += result[i];
    }
    for (int i = 0; i < result.length; i++) result[i] /= sum;
    return result;
  }

  private static float sum(float[] values) {
    float s = 0;
    for (float v : values) s += v;
    return s;
  }

  private static Path writeToyModel(Path directory) throws Exception {
    Random random = new Random(35);
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "qwen35")
            .addString("general.name", "Toy Qwen 3.5")
            .addUint32("qwen35.embedding_length", DIM)
            .addUint32("qwen35.block_count", 2)
            .addUint32("qwen35.attention.head_count", 2)
            .addUint32("qwen35.attention.head_count_kv", KV_HEADS)
            .addUint32("qwen35.attention.key_length", HEAD_DIM)
            .addUint32("qwen35.vocab_size", VOCAB)
            .addUint32("qwen35.context_length", 8)
            .addUint32("qwen35.feed_forward_length", HIDDEN_DIM)
            .addFloat32("qwen35.rope.freq_base", 10_000.0f)
            .addUint32("qwen35.rope.dimension_count", HEAD_DIM)
            .addFloat32("qwen35.attention.layer_norm_rms_epsilon", 1.0e-6f)
            .addUint32("qwen35.ssm.conv_kernel", 3)
            .addUint32("qwen35.ssm.state_size", HEAD_DIM)
            .addUint32("qwen35.ssm.group_count", 1)
            .addUint32("qwen35.ssm.time_step_rank", KV_HEADS)
            .addUint32("qwen35.ssm.inner_size", HEAD_DIM * KV_HEADS)
            .addUint32("qwen35.full_attention_interval", 2)
            .addStringArray(
                "tokenizer.ggml.tokens", List.of("<s>", "</s>", "a", "b", "c", "d", "e", "f"))
            .addFloat32Array(
                "tokenizer.ggml.scores", List.of(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f))
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1)
            .addTensor(
                "token_embd.weight",
                GgufTensorType.F32,
                new long[] {DIM, VOCAB},
                randomFloats(random, DIM * VOCAB))
            .addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, ones(DIM));

    // Layer 0: Gated Delta Net
    addGatedDeltaNetLayer(builder, random, 0);
    // Layer 1: Full attention
    addFullAttentionLayer(builder, random, 1);

    Path model = directory.resolve("toy-qwen35-embeddings-parity.gguf");
    Files.write(model, builder.build());
    return model;
  }

  private static void addGatedDeltaNetLayer(
      SyntheticGgufBuilder builder, Random random, int layer) {
    String p = "blk." + layer + ".";
    builder
        .addTensor(p + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, ones(DIM))
        .addTensor(
            p + "post_attention_norm.weight", GgufTensorType.F32, new long[] {DIM}, ones(DIM));
    addMatrix(builder, random, p + "attn_qkv.weight", CONV_DIM, DIM);
    addMatrix(builder, random, p + "attn_gate.weight", HEAD_DIM * KV_HEADS, DIM);
    builder
        .addTensor(
            p + "ssm_beta.weight",
            GgufTensorType.F32,
            new long[] {DIM, KV_HEADS},
            randomFloats(random, DIM * KV_HEADS))
        .addTensor(
            p + "ssm_alpha.weight",
            GgufTensorType.F32,
            new long[] {DIM, KV_HEADS},
            randomFloats(random, DIM * KV_HEADS))
        .addTensor(
            p + "ssm_conv1d.weight",
            GgufTensorType.F32,
            new long[] {3, CONV_DIM},
            randomFloats(random, 3 * CONV_DIM))
        .addTensor(
            p + "ssm_dt.bias", GgufTensorType.F32, new long[] {KV_HEADS}, floats(0.1f, -0.2f))
        .addTensor(p + "ssm_a", GgufTensorType.F32, new long[] {KV_HEADS}, floats(-0.5f, -0.75f))
        .addTensor(p + "ssm_norm.weight", GgufTensorType.F32, new long[] {HEAD_DIM}, ones(HEAD_DIM))
        .addTensor(
            p + "ssm_out.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIM * KV_HEADS, DIM},
            randomFloats(random, HEAD_DIM * KV_HEADS * DIM));
    addMatrix(builder, random, p + "ffn_gate.weight", HIDDEN_DIM, DIM);
    addMatrix(builder, random, p + "ffn_up.weight", HIDDEN_DIM, DIM);
    addMatrix(builder, random, p + "ffn_down.weight", DIM, HIDDEN_DIM);
  }

  private static void addFullAttentionLayer(
      SyntheticGgufBuilder builder, Random random, int layer) {
    String p = "blk." + layer + ".";
    builder
        .addTensor(p + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, ones(DIM))
        .addTensor(
            p + "post_attention_norm.weight", GgufTensorType.F32, new long[] {DIM}, ones(DIM));
    addMatrix(builder, random, p + "attn_q.weight", 2 * DIM, DIM);
    addMatrix(builder, random, p + "attn_k.weight", HEAD_DIM, DIM);
    addMatrix(builder, random, p + "attn_v.weight", HEAD_DIM, DIM);
    addMatrix(builder, random, p + "attn_output.weight", DIM, DIM);
    builder
        .addTensor(
            p + "attn_q_norm.weight", GgufTensorType.F32, new long[] {HEAD_DIM}, ones(HEAD_DIM))
        .addTensor(
            p + "attn_k_norm.weight", GgufTensorType.F32, new long[] {HEAD_DIM}, ones(HEAD_DIM));
    addMatrix(builder, random, p + "ffn_gate.weight", HIDDEN_DIM, DIM);
    addMatrix(builder, random, p + "ffn_up.weight", HIDDEN_DIM, DIM);
    addMatrix(builder, random, p + "ffn_down.weight", DIM, HIDDEN_DIM);
  }

  private static void addMatrix(
      SyntheticGgufBuilder builder, Random random, String name, int rows, int cols) {
    builder.addTensor(
        name, GgufTensorType.F32, new long[] {cols, rows}, randomFloats(random, rows * cols));
  }

  private static byte[] randomFloats(Random random, int count) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) values[i] = (random.nextFloat() - 0.5f) * 0.2f;
    return floats(values);
  }

  private static byte[] ones(int count) {
    float[] values = new float[count];
    java.util.Arrays.fill(values, 1.0f);
    return floats(values);
  }

  private static byte[] floats(float... values) {
    ByteBuffer buf =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : values) buf.putFloat(v);
    return buf.array();
  }

  private static boolean allFinite(float[] values) {
    for (float v : values) if (!Float.isFinite(v)) return false;
    return true;
  }
}
