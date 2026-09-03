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
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class Qwen35ForwardPassTest {

  private static final float SIMD_REDUCTION_TOLERANCE = 2.0e-7f;

  private static final int DIMENSION = 8;
  private static final int HEAD_DIMENSION = 4;
  private static final int VALUE_HEADS = 2;
  private static final int VALUE_DIMENSION = HEAD_DIMENSION * VALUE_HEADS;
  private static final int KEY_DIMENSION = HEAD_DIMENSION;
  private static final int CONVOLUTION_DIMENSION = 2 * KEY_DIMENSION + VALUE_DIMENSION;
  private static final int HIDDEN_DIMENSION = 8;
  private static final int VOCABULARY_SIZE = 8;

  @Test
  void transposesChannelMajorConvolutionWeightsForVectorizedExecution() {
    float[] channelMajor = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f
    };

    assertThat(Qwen35Weights.toTapMajorConvolution(channelMajor, 4, 3))
        .containsExactly(1.0f, 4.0f, 7.0f, 10.0f, 2.0f, 5.0f, 8.0f, 11.0f, 3.0f, 6.0f, 9.0f, 12.0f);
  }

  @Test
  void attentionHeadKernelMatchesScalarReference() {
    int positions = 3;
    int headDimension = 4;
    int cacheStride = 8;
    float scale = 0.5f;
    float[] query = {9.0f, 0.25f, -0.5f, 0.75f, 1.0f, 8.0f};
    float[] keys = {
      7, 7, 7, 7, 0.1f, 0.2f, -0.3f, 0.4f,
      6, 6, 6, 6, -0.5f, 0.6f, 0.7f, -0.8f,
      5, 5, 5, 5, 0.9f, -1.0f, 1.1f, 1.2f
    };
    float[] values = {
      4, 4, 4, 4, 0.5f, -0.25f, 0.75f, 1.0f,
      3, 3, 3, 3, -0.4f, 0.3f, 0.2f, -0.1f,
      2, 2, 2, 2, 1.2f, 0.8f, -0.6f, 0.4f
    };
    float[] gate = {6.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f};
    float[] expectedScores = new float[positions];
    for (int position = 0; position < positions; position++) {
      float dot = 0.0f;
      for (int column = 0; column < headDimension; column++) {
        dot += query[1 + column] * keys[position * cacheStride + 4 + column];
      }
      expectedScores[position] = dot * scale;
    }
    TensorOps.softmax(expectedScores, 0, positions);
    float[] expected = new float[headDimension];
    for (int column = 0; column < headDimension; column++) {
      float sum = 0.0f;
      for (int position = 0; position < positions; position++) {
        sum += expectedScores[position] * values[position * cacheStride + 4 + column];
      }
      float gateValue = gate[1 + column];
      expected[column] = sum / (1.0f + (float) Math.exp(-gateValue));
    }

    float[] actual = {99.0f, Float.NaN, Float.NaN, Float.NaN, Float.NaN, 98.0f};
    Qwen35ForwardPass.attendHead(
        actual,
        1,
        query,
        1,
        keys,
        4,
        cacheStride,
        values,
        4,
        cacheStride,
        gate,
        1,
        new float[positions],
        positions,
        headDimension,
        scale);

    assertThat(actual[0]).isEqualTo(99.0f);
    for (int column = 0; column < headDimension; column++) {
      assertThat(actual[1 + column]).isCloseTo(expected[column], within(2.0e-6f));
    }
    assertThat(actual[5]).isEqualTo(98.0f);
  }

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
      assertThat(second).containsExactly(complete, within(SIMD_REDUCTION_TOLERANCE));

      graph.rewind(session, 1);
      assertThat(graph.forward(session, 2, 1))
          .containsExactly(second, within(SIMD_REDUCTION_TOLERANCE));

      graph.reset(session);
      assertThat(session.checkpoint()).isZero();
      assertThat(graph.forward(session, 1, 0))
          .containsExactly(first, within(SIMD_REDUCTION_TOLERANCE));
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
  void injectedKernelExecutesEligibleQwen35Projections(@TempDir Path directory) throws Exception {
    Path model = writeToyModel(directory);

    AtomicInteger invocations = new AtomicInteger();
    AtomicInteger gatedDeltaNetInvocations = new AtomicInteger();
    AtomicInteger maximumBatchSize = new AtomicInteger();
    GgufBatchedMatrixKernel kernel =
        new GgufBatchedMatrixKernel() {
          @Override
          public boolean supports(GgufTensorType type) {
            return type == GgufTensorType.F32;
          }

          @Override
          public boolean supportsGatedDeltaNet() {
            return true;
          }

          @Override
          public void gatedDeltaNet(
              float[] query,
              float[] key,
              float[] value,
              float[] logDecay,
              float[] beta,
              float[] state,
              float[] output,
              int tokenCount,
              int keyHeadCount,
              int valueHeadCount,
              int keyDimension,
              int valueDimension) {
            gatedDeltaNetInvocations.incrementAndGet();
            GatedDeltaNetRecurrence.forwardPrefixInPlace(
                query,
                key,
                value,
                logDecay,
                beta,
                state,
                output,
                new float[keyDimension],
                new float[keyDimension],
                new float[valueDimension],
                new float[valueDimension],
                tokenCount,
                keyHeadCount,
                valueHeadCount,
                keyDimension,
                valueDimension);
          }

          @Override
          public void multiply(
              float[] output,
              float[] input,
              MemorySegment weights,
              GgufTensorType type,
              int batchSize,
              int rows,
              int cols) {
            invocations.incrementAndGet();
            maximumBatchSize.accumulateAndGet(batchSize, Math::max);
            float[] rowInput = new float[cols];
            float[] rowOutput = new float[rows];
            for (int batch = 0; batch < batchSize; batch++) {
              System.arraycopy(input, batch * cols, rowInput, 0, cols);
              TensorOps.ggufMatmul(rowOutput, rowInput, weights, type, rows, cols);
              System.arraycopy(rowOutput, 0, output, batch * rows, rows);
            }
          }
        };

    try (PureJavaBackend baseline = PureJavaBackend.load(model);
        PureJavaBackend accelerated = PureJavaBackend.load(model, kernel)) {
      float[] expected = baseline.prefill(new int[] {1, 2, 3, 4}, 0);
      float[] actual = accelerated.prefill(new int[] {1, 2, 3, 4}, 0);

      assertThat(invocations).hasValueGreaterThan(0);
      assertThat(gatedDeltaNetInvocations).hasValueGreaterThan(0);
      assertThat(maximumBatchSize).hasValueGreaterThanOrEqualTo(1);
      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void linearStateSnapshotRestoresContinuationWithoutReplayAndReportsBytes(@TempDir Path directory)
      throws Exception {
    Path model = writeToyModel(directory);

    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena));
      Qwen35ForwardPass.Session session = graph.openSession(6);
      graph.prefill(session, new int[] {1, 2}, 0);

      Qwen35ForwardPass.LinearStateSnapshot snapshot = graph.captureLinearState(session);
      float[] expected = graph.prefill(session, new int[] {3, 4}, 2);

      graph.restoreLinearState(session, snapshot);

      assertThat(snapshot.checkpoint()).isEqualTo(2);
      assertThat(snapshot.bytes()).isEqualTo(256L);
      assertThat(session.checkpoint()).isEqualTo(2);
      assertThat(graph.prefill(session, new int[] {3, 4}, 2)).containsExactly(expected);
    }
  }

  @Test
  void linearStateSnapshotIsBoundToItsSessionAndRetainedPrefix(@TempDir Path directory)
      throws Exception {
    Path model = writeToyModel(directory);

    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena));
      Qwen35ForwardPass.Session original = graph.openSession(4);
      graph.prefill(original, new int[] {1, 2}, 0);
      Qwen35ForwardPass.LinearStateSnapshot snapshot = graph.captureLinearState(original);

      assertThatThrownBy(() -> graph.restoreLinearState(graph.openSession(4), snapshot))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("same session");

      graph.rewind(original, 1);
      graph.forward(original, 3, 1);
      assertThatThrownBy(() -> graph.restoreLinearState(original, snapshot))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("prefix");
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
            .addString("general.name", "Toy Qwen 3.5")
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
            .addStringArray(
                "tokenizer.ggml.tokens", List.of("<s>", "</s>", "a", "b", "c", "d", "e", "f"))
            .addFloat32Array(
                "tokenizer.ggml.scores", List.of(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f))
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1)
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
