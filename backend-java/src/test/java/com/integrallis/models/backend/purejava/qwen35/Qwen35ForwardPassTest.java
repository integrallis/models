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
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class Qwen35ForwardPassTest {

  private static final int DIMENSION = 8;
  private static final int HEAD_DIMENSION = 4;
  private static final int VALUE_HEADS = 2;
  private static final int VALUE_DIMENSION = HEAD_DIMENSION * VALUE_HEADS;
  private static final int KEY_DIMENSION = HEAD_DIMENSION;
  private static final int CONVOLUTION_DIMENSION = 2 * KEY_DIMENSION + VALUE_DIMENSION;
  private static final int HIDDEN_DIMENSION = 8;
  private static final int VOCABULARY_SIZE = 8;

  @Test
  void toyHybridGraphPreservesStateAcrossPrefillRewindAndReset(@TempDir Path directory)
      throws Exception {
    Path model = writeToyModel(directory);

    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena));
      Qwen35ForwardPass.Session session = graph.openSession(4);

      float[] first = graph.forward(session, 1, 0);
      float[] second = graph.forward(session, 2, 1);
      assertThat(first).hasSize(VOCABULARY_SIZE);
      assertThat(second).hasSize(VOCABULARY_SIZE);
      assertThat(allFinite(first)).isTrue();
      assertThat(allFinite(second)).isTrue();
      assertThat(session.checkpoint()).isEqualTo(2);

      float[] complete = graph.forward(new int[] {1, 2});
      assertThat(second).containsExactly(complete);

      graph.rewind(session, 1);
      assertThat(graph.forward(session, 2, 1)).containsExactly(second);

      graph.reset(session);
      assertThat(session.checkpoint()).isZero();
      assertThat(graph.forward(session, 1, 0)).containsExactly(first);
    }
  }

  @Test
  void batchedPrefillMatchesSequentialExecutionAcrossChunks(@TempDir Path directory)
      throws Exception {
    Path model = writeToyModel(directory);

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(model, arena);
      Qwen35ForwardPass sequential = Qwen35ForwardPass.fromGgufFile(file, 1);
      Qwen35ForwardPass batched = Qwen35ForwardPass.fromGgufFile(file, 2);
      Qwen35ForwardPass.Session sequentialSession = sequential.openSession(6);
      Qwen35ForwardPass.Session batchedSession = batched.openSession(6);

      float[] expected = sequential.prefill(sequentialSession, new int[] {1, 2, 3, 4, 5}, 0);
      float[] actual = batched.prefill(batchedSession, new int[] {1, 2, 3, 4, 5}, 0);

      assertThat(batched.prefillBatchSize()).isEqualTo(2);
      assertThat(actual).containsExactly(expected);
      assertThat(batchedSession.checkpoint()).isEqualTo(5);
      assertThat(batched.forward(batchedSession, 6, 5))
          .containsExactly(sequential.forward(sequentialSession, 6, 5));

      batched.rewind(batchedSession, 2);
      assertThat(batched.prefill(batchedSession, new int[] {3, 4, 5, 6}, 2))
          .containsExactly(sequential.forward(new int[] {1, 2, 3, 4, 5, 6}));
    }
  }

  @Test
  void sessionsEnforceOwnershipPositionAndCapacity(@TempDir Path directory) throws Exception {
    Path model = writeToyModel(directory);

    try (Arena firstArena = Arena.ofConfined();
        Arena secondArena = Arena.ofConfined()) {
      Qwen35ForwardPass first = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, firstArena));
      Qwen35ForwardPass second =
          Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, secondArena));
      Qwen35ForwardPass.Session session = first.openSession(1);

      assertThatThrownBy(() -> first.openSession(0))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("capacity");
      assertThatThrownBy(() -> second.forward(session, 1, 0))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("different Qwen3.5 graph");
      assertThatThrownBy(() -> first.forward(session, 1, 1))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("position must be sequential");

      first.forward(session, 1, 0);
      assertThatThrownBy(() -> first.forward(session, 2, 1))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("capacity exhausted");
      assertThatThrownBy(() -> first.rewind(session, 2))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("checkpoint must be between");
      assertThatThrownBy(() -> first.prefill(first.openSession(2), new int[0], 0))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("must not be empty");
      assertThatThrownBy(() -> first.prefill(first.openSession(1), new int[] {1, 2}, 0))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("remaining session capacity");
    }
  }

  private static Path writeToyModel(Path directory) throws Exception {
    Random random = new Random(35);
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "qwen35")
            .addUint32("qwen35.embedding_length", DIMENSION)
            .addUint32("qwen35.block_count", 2)
            .addUint32("qwen35.attention.head_count", 2)
            .addUint32("qwen35.attention.head_count_kv", 1)
            .addUint32("qwen35.attention.key_length", HEAD_DIMENSION)
            .addUint32("qwen35.vocab_size", VOCABULARY_SIZE)
            .addUint32("qwen35.context_length", 8)
            .addUint32("qwen35.feed_forward_length", HIDDEN_DIMENSION)
            .addFloat32("qwen35.rope.freq_base", 10_000.0f)
            .addUint32("qwen35.rope.dimension_count", HEAD_DIMENSION)
            .addFloat32("qwen35.attention.layer_norm_rms_epsilon", 1.0e-6f)
            .addUint32("qwen35.ssm.conv_kernel", 3)
            .addUint32("qwen35.ssm.state_size", HEAD_DIMENSION)
            .addUint32("qwen35.ssm.group_count", 1)
            .addUint32("qwen35.ssm.time_step_rank", VALUE_HEADS)
            .addUint32("qwen35.ssm.inner_size", VALUE_DIMENSION)
            .addUint32("qwen35.full_attention_interval", 2)
            .addTensor(
                "token_embd.weight",
                GgufTensorType.F32,
                new long[] {DIMENSION, VOCABULARY_SIZE},
                randomFloats(random, DIMENSION * VOCABULARY_SIZE))
            .addTensor(
                "output_norm.weight", GgufTensorType.F32, new long[] {DIMENSION}, ones(DIMENSION));

    addGatedDeltaNetLayer(builder, random, 0);
    addFullAttentionLayer(builder, random, 1);
    Path model = directory.resolve("toy-qwen35.gguf");
    Files.write(model, builder.build());
    return model;
  }

  private static void addGatedDeltaNetLayer(
      SyntheticGgufBuilder builder, Random random, int layer) {
    String prefix = "blk." + layer + ".";
    addCommonLayer(builder, random, prefix);
    addMatrix(builder, random, prefix + "attn_qkv.weight", CONVOLUTION_DIMENSION, DIMENSION);
    addMatrix(builder, random, prefix + "attn_gate.weight", VALUE_DIMENSION, DIMENSION);
    addMatrix(builder, random, prefix + "ssm_beta.weight", VALUE_HEADS, DIMENSION);
    addMatrix(builder, random, prefix + "ssm_alpha.weight", VALUE_HEADS, DIMENSION);
    builder
        .addTensor(
            prefix + "ssm_conv1d.weight",
            GgufTensorType.F32,
            new long[] {3, CONVOLUTION_DIMENSION},
            randomFloats(random, 3 * CONVOLUTION_DIMENSION))
        .addTensor(
            prefix + "ssm_dt.bias",
            GgufTensorType.F32,
            new long[] {VALUE_HEADS},
            floats(0.1f, -0.2f))
        .addTensor(
            prefix + "ssm_a", GgufTensorType.F32, new long[] {VALUE_HEADS}, floats(-0.5f, -0.75f))
        .addTensor(
            prefix + "ssm_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIMENSION},
            ones(HEAD_DIMENSION));
    addMatrix(builder, random, prefix + "ssm_out.weight", DIMENSION, VALUE_DIMENSION);
  }

  private static void addFullAttentionLayer(
      SyntheticGgufBuilder builder, Random random, int layer) {
    String prefix = "blk." + layer + ".";
    addCommonLayer(builder, random, prefix);
    addMatrix(builder, random, prefix + "attn_q.weight", 2 * DIMENSION, DIMENSION);
    addMatrix(builder, random, prefix + "attn_k.weight", KEY_DIMENSION, DIMENSION);
    addMatrix(builder, random, prefix + "attn_v.weight", KEY_DIMENSION, DIMENSION);
    addMatrix(builder, random, prefix + "attn_output.weight", DIMENSION, DIMENSION);
    builder
        .addTensor(
            prefix + "attn_q_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIMENSION},
            ones(HEAD_DIMENSION))
        .addTensor(
            prefix + "attn_k_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIMENSION},
            ones(HEAD_DIMENSION));
  }

  private static void addCommonLayer(SyntheticGgufBuilder builder, Random random, String prefix) {
    builder
        .addTensor(
            prefix + "attn_norm.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION},
            ones(DIMENSION))
        .addTensor(
            prefix + "post_attention_norm.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION},
            ones(DIMENSION));
    addMatrix(builder, random, prefix + "ffn_gate.weight", HIDDEN_DIMENSION, DIMENSION);
    addMatrix(builder, random, prefix + "ffn_up.weight", HIDDEN_DIMENSION, DIMENSION);
    addMatrix(builder, random, prefix + "ffn_down.weight", DIMENSION, HIDDEN_DIMENSION);
  }

  private static void addMatrix(
      SyntheticGgufBuilder builder, Random random, String name, int rows, int columns) {
    builder.addTensor(
        name, GgufTensorType.F32, new long[] {columns, rows}, randomFloats(random, rows * columns));
  }

  private static byte[] randomFloats(Random random, int length) {
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = (random.nextFloat() - 0.5f) * 0.2f;
    }
    return floats(values);
  }

  private static boolean allFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  private static byte[] ones(int length) {
    float[] values = new float[length];
    java.util.Arrays.fill(values, 1.0f);
    return floats(values);
  }

  private static byte[] floats(float... values) {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putFloat(value);
    }
    return buffer.array();
  }
}
