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
import static org.assertj.core.api.Assertions.within;

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
 * Qwen3-Next fuses the Gated DeltaNet's beta and alpha projections into one {@code ssm_ba} tensor.
 *
 * <p>The packing is <b>interleaved by key-head group</b>, which is the only thing these tests are
 * about. The reference reshapes the {@code 2 * n_value_heads} rows to {@code [2 * group,
 * n_key_heads]} and reads beta from the first {@code group} entries of each column and alpha from
 * the next. For four value heads over two key heads that is {@code [b0,b1,a0,a1,b2,b3,a2,a3]} --
 * <i>not</i> four betas followed by four alphas.
 *
 * <p><b>The fixture uses {@code group_count = 2} on purpose.</b> With one key head the interleave
 * degenerates into exactly the naive two-halves split, so a fixture built that way passes whichever
 * reading the code uses and proves nothing. That is the trap this file exists to avoid, and {@link
 * #theNaivePackingIsNotEquivalent()} keeps it honest by requiring the wrong packing to disagree.
 */
@Tag("unit")
class Qwen35FusedBetaAlphaTest {

  private static final int DIMENSION = 8;
  private static final int HEAD_DIMENSION = 4;

  /** Four value heads over two key heads, so each group holds two of each parameter. */
  private static final int VALUE_HEADS = 4;

  private static final int KEY_HEADS = 2;
  private static final int VALUE_DIMENSION = VALUE_HEADS * HEAD_DIMENSION;
  private static final int KEY_DIMENSION = KEY_HEADS * HEAD_DIMENSION;
  private static final int CONVOLUTION_DIMENSION = 2 * KEY_DIMENSION + VALUE_DIMENSION;
  private static final int HIDDEN_DIMENSION = 8;
  private static final int VOCABULARY_SIZE = 8;

  /** How the fused tensor's rows are ordered. */
  private enum Packing {
    /** Two separate tensors, as Qwen3.5 publishes them. */
    SEPARATE,
    /** Fused and interleaved per key-head group, as Qwen3-Next publishes it. */
    FUSED_INTERLEAVED,
    /** Fused as all betas then all alphas: the plausible misreading. */
    FUSED_NAIVE_HALVES
  }

  @Test
  void aFusedInterleavedTensorComputesWhatTwoSeparateTensorsDo(@TempDir Path directory)
      throws Exception {
    float[] separate = decode(write(directory, "separate.gguf", Packing.SEPARATE));
    float[] fused = decode(write(directory, "fused.gguf", Packing.FUSED_INTERLEAVED));

    assertThat(fused).hasSameSizeAs(separate);
    for (int index = 0; index < separate.length; index++) {
      assertThat(fused[index])
          .describedAs("fused logit %s must equal the separate-tensor equivalent", index)
          .isEqualTo(separate[index], within(1.0e-6f));
    }
  }

  @Test
  void theNaivePackingIsNotEquivalent(@TempDir Path directory) throws Exception {
    float[] separate = decode(write(directory, "separate.gguf", Packing.SEPARATE));
    float[] naive = decode(write(directory, "naive.gguf", Packing.FUSED_NAIVE_HALVES));

    // If this ever passes, the fixture has stopped discriminating -- most likely because
    // group_count
    // drifted to 1, where the two packings are the same bytes.
    assertThat(naive)
        .describedAs("all-betas-then-all-alphas must not match the interleaved reading")
        .isNotEqualTo(separate);
  }

  /**
   * The route the fleet takes: a qwen3next file must reach this decoder and report its own name.
   */
  @Test
  void theBackendLoadsAQwen3NextFileUnderItsOwnArchitecture(@TempDir Path directory)
      throws Exception {
    Path model = write(directory, "fused.gguf", Packing.FUSED_INTERLEAVED);

    try (com.integrallis.models.backend.purejava.PureJavaBackend backend =
        com.integrallis.models.backend.purejava.PureJavaBackend.load(model)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen3next");
      float[] logits = backend.prefill(new int[] {1, 2, 3}, 0);

      assertThat(logits).hasSize(VOCABULARY_SIZE);
      for (float logit : logits) {
        assertThat(Float.isFinite(logit)).isTrue();
      }
    }
  }

  @Test
  void theFixtureGeometryActuallyGroups(@TempDir Path directory) throws Exception {
    Path model = write(directory, "fused.gguf", Packing.FUSED_INTERLEAVED);

    try (Arena arena = Arena.ofConfined()) {
      Qwen35Config config = Qwen35Config.fromMetadata(GgufParser.parse(model, arena).metadata());

      assertThat(config.architecture()).isEqualTo("qwen3next");
      // More than one key head and more than one value head per key head, or the interleave is a
      // no-op and the equivalence above would hold for the wrong reading too.
      assertThat(config.gdnKeyHeads()).isGreaterThan(1);
      assertThat(config.gdnValueHeads() / config.gdnKeyHeads()).isGreaterThan(1);
    }
  }

  private static float[] decode(Path model) throws Exception {
    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena));
      Qwen35ForwardPass.Session session = graph.openSession(4);
      graph.forward(session, 1, 0);
      return graph.forward(session, 2, 1);
    }
  }

  /** Beta for value head {@code v}; the same numbers whichever packing writes them. */
  private static float[] betaRow(int head) {
    return deterministic(500 + head, DIMENSION);
  }

  private static float[] alphaRow(int head) {
    return deterministic(600 + head, DIMENSION);
  }

  private static Path write(Path directory, String name, Packing packing) throws Exception {
    Random random = new Random(357);
    // qwen3next for every fixture, including the separate-tensor one: the architecture name does
    // not decide the packing, the presence of ssm_ba does, and that is worth pinning here too.
    String prefix = "qwen3next.";
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "qwen3next")
            .addString("general.name", "Toy Qwen3-Next")
            .addUint32(prefix + "embedding_length", DIMENSION)
            .addUint32(prefix + "block_count", 1)
            .addUint32(prefix + "attention.head_count", 2)
            .addUint32(prefix + "attention.head_count_kv", 1)
            .addUint32(prefix + "attention.key_length", HEAD_DIMENSION)
            .addUint32(prefix + "vocab_size", VOCABULARY_SIZE)
            .addUint32(prefix + "context_length", 8)
            .addUint32(prefix + "feed_forward_length", HIDDEN_DIMENSION)
            .addFloat32(prefix + "rope.freq_base", 5_000_000.0f)
            .addUint32(prefix + "rope.dimension_count", HEAD_DIMENSION)
            .addFloat32(prefix + "attention.layer_norm_rms_epsilon", 1.0e-6f)
            .addUint32(prefix + "ssm.conv_kernel", 3)
            .addUint32(prefix + "ssm.state_size", HEAD_DIMENSION)
            .addUint32(prefix + "ssm.group_count", KEY_HEADS)
            .addUint32(prefix + "ssm.time_step_rank", VALUE_HEADS)
            .addUint32(prefix + "ssm.inner_size", VALUE_DIMENSION)
            // Interval 2 over one layer: (0 + 1) % 2 != 0, so the single layer is Gated DeltaNet.
            .addUint32(prefix + "full_attention_interval", 2)
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
                bytes(randomFloats(random, DIMENSION * VOCABULARY_SIZE)))
            .addTensor(
                "output_norm.weight",
                GgufTensorType.F32,
                new long[] {DIMENSION},
                bytes(ones(DIMENSION)));

    String layer = "blk.0.";
    builder
        .addTensor(
            layer + "attn_norm.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION},
            bytes(ones(DIMENSION)))
        .addTensor(
            layer + "post_attention_norm.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION},
            bytes(ones(DIMENSION)));
    matrix(builder, layer + "ffn_gate.weight", HIDDEN_DIMENSION, DIMENSION, random);
    matrix(builder, layer + "ffn_up.weight", HIDDEN_DIMENSION, DIMENSION, random);
    matrix(builder, layer + "ffn_down.weight", DIMENSION, HIDDEN_DIMENSION, random);
    matrix(builder, layer + "attn_qkv.weight", CONVOLUTION_DIMENSION, DIMENSION, random);
    matrix(builder, layer + "attn_gate.weight", VALUE_DIMENSION, DIMENSION, random);
    builder
        .addTensor(
            layer + "ssm_conv1d.weight",
            GgufTensorType.F32,
            new long[] {3, CONVOLUTION_DIMENSION},
            bytes(randomFloats(random, 3 * CONVOLUTION_DIMENSION)))
        .addTensor(
            layer + "ssm_dt.bias",
            GgufTensorType.F32,
            new long[] {VALUE_HEADS},
            bytes(new float[] {0.1f, -0.2f, 0.05f, -0.15f}))
        .addTensor(
            layer + "ssm_a",
            GgufTensorType.F32,
            new long[] {VALUE_HEADS},
            bytes(new float[] {-0.5f, -0.75f, -0.6f, -0.9f}))
        .addTensor(
            layer + "ssm_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIMENSION},
            bytes(ones(HEAD_DIMENSION)));
    matrix(builder, layer + "ssm_out.weight", DIMENSION, VALUE_DIMENSION, random);
    writeBetaAlpha(builder, layer, packing);

    Path model = directory.resolve(name);
    Files.write(model, builder.build());
    return model;
  }

  private static void writeBetaAlpha(SyntheticGgufBuilder builder, String layer, Packing packing) {
    if (packing == Packing.SEPARATE) {
      float[] beta = new float[VALUE_HEADS * DIMENSION];
      float[] alpha = new float[VALUE_HEADS * DIMENSION];
      for (int head = 0; head < VALUE_HEADS; head++) {
        System.arraycopy(betaRow(head), 0, beta, head * DIMENSION, DIMENSION);
        System.arraycopy(alphaRow(head), 0, alpha, head * DIMENSION, DIMENSION);
      }
      addRows(builder, layer + "ssm_beta.weight", VALUE_HEADS, beta);
      addRows(builder, layer + "ssm_alpha.weight", VALUE_HEADS, alpha);
      return;
    }
    float[] fused = new float[2 * VALUE_HEADS * DIMENSION];
    int group = VALUE_HEADS / KEY_HEADS;
    for (int head = 0; head < VALUE_HEADS; head++) {
      int betaRowIndex;
      int alphaRowIndex;
      if (packing == Packing.FUSED_INTERLEAVED) {
        int keyHead = head / group;
        int within = head % group;
        betaRowIndex = keyHead * 2 * group + within;
        alphaRowIndex = keyHead * 2 * group + group + within;
      } else {
        // All betas, then all alphas.
        betaRowIndex = head;
        alphaRowIndex = VALUE_HEADS + head;
      }
      System.arraycopy(betaRow(head), 0, fused, betaRowIndex * DIMENSION, DIMENSION);
      System.arraycopy(alphaRow(head), 0, fused, alphaRowIndex * DIMENSION, DIMENSION);
    }
    addRows(builder, layer + "ssm_ba.weight", 2 * VALUE_HEADS, fused);
  }

  private static void addRows(SyntheticGgufBuilder builder, String name, int rows, float[] values) {
    builder.addTensor(name, GgufTensorType.F32, new long[] {DIMENSION, rows}, bytes(values));
  }

  private static void matrix(
      SyntheticGgufBuilder builder, String name, int rows, int columns, Random random) {
    builder.addTensor(
        name,
        GgufTensorType.F32,
        new long[] {columns, rows},
        bytes(randomFloats(random, rows * columns)));
  }

  private static float[] deterministic(int seed, int length) {
    return randomFloats(new Random(seed), length);
  }

  private static float[] randomFloats(Random random, int length) {
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = (random.nextFloat() - 0.5f) * 0.2f;
    }
    return values;
  }

  private static float[] ones(int length) {
    float[] values = new float[length];
    java.util.Arrays.fill(values, 1.0f);
    return values;
  }

  private static byte[] bytes(float[] values) {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putFloat(value);
    }
    return buffer.array();
  }
}
