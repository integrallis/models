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
 * Routed experts stored quantized, which is how every model this decoder will actually be given
 * stores them.
 *
 * <p>Separate from {@link Qwen35MixtureOfExpertsTest} because it needs a different geometry: a
 * {@code Q8_0} row is 32 weights, and experts are sliced on whole rows, so every width involved has
 * to be a multiple of 32. The F32 fixtures next door are 8 wide and cannot express that at all --
 * which is exactly why they cannot test any of this.
 *
 * <p>What only a quantized fixture can catch: a slice that does not start on a block boundary. The
 * bytes it then reads are a block's scale reinterpreted as weights, which produces plausible finite
 * numbers rather than an error, so nothing downstream would report it.
 */
@Tag("unit")
class Qwen35QuantizedExpertsTest {

  /** Every projection width here is a multiple of the 32-weight Q8_0 block. */
  private static final int DIMENSION = 32;

  private static final int HEAD_DIMENSION = 16;
  private static final int KEY_DIMENSION = HEAD_DIMENSION;
  private static final int EXPERT_HIDDEN = 32;
  private static final int SHARED_HIDDEN = 64;
  private static final int EXPERTS = 4;
  private static final int EXPERTS_USED = 2;
  private static final int VOCABULARY_SIZE = 8;

  private static final int[] NATURAL = {0, 1, 2, 3};

  /**
   * A quantized expert slice has to dequantize to what a standalone quantized tensor of the same
   * bytes dequantizes to.
   *
   * <p>The reference is a <b>dense Q8_0</b> model holding those weights, not an F32 one. That
   * distinction was learned here: an F32 reference disagrees by about 0.3%, because the Q8_0 matmul
   * quantizes the <i>activations</i> as well, so the two are different arithmetic rather than the
   * same arithmetic in a different order. Comparing against F32 would have meant choosing a
   * tolerance loose enough to absorb activation quantization -- and a slice reading a block's scale
   * bytes as weights can easily land inside a tolerance that loose.
   *
   * <p>With every expert holding the same bytes and routing weights that sum to one, the routed
   * model and the dense one run the same kernel over the same numbers.
   */
  @Test
  void clonedQuantizedExpertsReproduceTheDenseQuantizedEquivalent(@TempDir Path directory)
      throws Exception {
    float[] routed = decode(write(directory, "q8-cloned.gguf", Mode.CLONED_EXPERTS, NATURAL));
    float[] dense = decode(write(directory, "q8-dense.gguf", Mode.DENSE, NATURAL));

    assertThat(routed).hasSameSizeAs(dense);
    for (int index = 0; index < dense.length; index++) {
      assertThat(routed[index])
          .describedAs("routed Q8_0 logit %s must equal the dense Q8_0 equivalent", index)
          .isEqualTo(dense[index], within(1.0e-5f));
    }
  }

  /**
   * The control for the tolerance above: an F32 model of the same weights does <b>not</b> agree to
   * it, which is what makes 1e-5 a statement about slicing rather than about quantization error.
   */
  @Test
  void anF32ReferenceIsNotCloseEnoughToStandInForTheQuantizedOne(@TempDir Path directory)
      throws Exception {
    float[] dense = decode(write(directory, "q8-dense.gguf", Mode.DENSE, NATURAL));
    float[] unquantized = decode(write(directory, "f32-dense.gguf", Mode.DENSE_F32, NATURAL));

    float largest = 0.0f;
    for (int index = 0; index < dense.length; index++) {
      largest = Math.max(largest, Math.abs(dense[index] - unquantized[index]));
    }
    assertThat(largest)
        .describedAs("activation quantization alone moves the logits well past the slicing bar")
        .isGreaterThan(1.0e-4f);
  }

  /**
   * A quantized output head must be decoded with <b>its own</b> type.
   *
   * <p>Every published Qwen3.5-MoE and Qwen3-Next carries {@code output.weight} as Q6_K while its
   * token embedding is Q8_0 or Q4_K. An all-F32 fixture cannot tell "reads the head's type" from
   * "reuses the embedding's type", because there the two are the same -- the blind spot this file
   * exists to cover.
   *
   * <p>The bar is loose on purpose. A Q8_0 head and an F32 head of the same dequantized values
   * differ by activation quantization, which is small; decoding Q8_0 bytes as raw floats is not
   * small, it is nonsense. So the assertion is "close, and finite", which separates exactly those
   * two outcomes.
   */
  @Test
  void aQuantizedOutputHeadIsDecodedWithItsOwnType(@TempDir Path directory) throws Exception {
    float[] quantizedHead =
        decode(
            write(
                directory,
                "q8-head.gguf",
                Mode.DISTINCT_EXPERTS,
                NATURAL,
                EXPERT_HIDDEN,
                GgufTensorType.Q8_0));
    float[] unquantizedHead =
        decode(
            write(
                directory,
                "f32-head.gguf",
                Mode.DISTINCT_EXPERTS,
                NATURAL,
                EXPERT_HIDDEN,
                GgufTensorType.F32));

    for (float logit : quantizedHead) {
      assertThat(Float.isFinite(logit)).describedAs("a misread head produces NaN or huge").isTrue();
    }
    assertThat(quantizedHead).hasSameSizeAs(unquantizedHead);
    for (int index = 0; index < unquantizedHead.length; index++) {
      assertThat(quantizedHead[index])
          .describedAs("Q8_0 head logit %s against its F32 twin", index)
          .isEqualTo(unquantizedHead[index], within(0.05f));
    }
  }

  /**
   * And the head has to be read at all: absent, the logits come from the token embedding instead.
   */
  @Test
  void anOutputHeadChangesTheLogitsOnAQuantizedModel(@TempDir Path directory) throws Exception {
    float[] tied = decode(write(directory, "tied.gguf", Mode.DISTINCT_EXPERTS, NATURAL));
    float[] untied =
        decode(
            write(
                directory,
                "q8-head.gguf",
                Mode.DISTINCT_EXPERTS,
                NATURAL,
                EXPERT_HIDDEN,
                GgufTensorType.Q8_0));

    assertThat(untied).isNotEqualTo(tied);
  }

  /**
   * And the pairing of router row to expert slice has to survive quantization, where the offset is
   * a byte count derived from the block size rather than a float count.
   */
  @Test
  void permutingQuantizedExpertsAndTheirRouterRowsLeavesTheOutputUnchanged(@TempDir Path directory)
      throws Exception {
    float[] natural = decode(write(directory, "q8.gguf", Mode.DISTINCT_EXPERTS, NATURAL));
    float[] swapped =
        decode(write(directory, "q8-swapped.gguf", Mode.DISTINCT_EXPERTS, new int[] {1, 0, 2, 3}));

    assertThat(swapped).hasSameSizeAs(natural);
    for (int index = 0; index < natural.length; index++) {
      assertThat(swapped[index])
          .describedAs("permuted quantized logit %s", index)
          .isEqualTo(natural[index], within(1.0e-6f));
    }
  }

  /**
   * A width that is not a whole number of blocks cannot be sliced into experts at all, and has to
   * be refused rather than silently rounded -- rounding would straddle a block and read its scale
   * bytes as weights.
   */
  @Test
  void anExpertWidthThatStraddlesABlockIsRefused(@TempDir Path directory) throws Exception {
    // 48 is not a multiple of 32, so ffn_down_exps rows cannot land on block boundaries.
    Path model = write(directory, "ragged.gguf", Mode.DISTINCT_EXPERTS, NATURAL, 48);

    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(() -> Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("ffn_down_exps.weight")
          .hasMessageContaining("block");
    }
  }

  /** The quantized-activation scratch has to be sized for the widest projection in the model. */
  @Test
  void theScratchCoversTheWidestRoutedProjection(@TempDir Path directory) throws Exception {
    Path model = write(directory, "q8.gguf", Mode.DISTINCT_EXPERTS, NATURAL);

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(model, arena);
      Qwen35Weights weights =
          Qwen35Weights.fromGgufFile(file, Qwen35Config.fromMetadata(file.metadata()));

      // ffn_down_shexp takes the widest input in this fixture, wider than the model width itself,
      // so a scratch sized from the dense hidden width (0 here) or from the embedding would be too
      // small the moment the weights are quantized.
      assertThat(weights.widestProjectionInput())
          .describedAs("the shared expert's hidden width is the widest projection input here")
          .isEqualTo(SHARED_HIDDEN);
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

  /** What a fixture's feed-forward holds, and in what tensor type. */
  private enum Mode {
    /** Dense Q8_0 {@code ffn_gate}/{@code ffn_up}/{@code ffn_down}: the reference. */
    DENSE(GgufTensorType.Q8_0, false),
    /** The same weights unquantized, to show the reference has to be the quantized one. */
    DENSE_F32(GgufTensorType.F32, false),
    /** Routed Q8_0, every expert holding the bytes {@link #DENSE} puts in its dense triple. */
    CLONED_EXPERTS(GgufTensorType.Q8_0, true),
    /** Routed Q8_0, each expert its own weights. */
    DISTINCT_EXPERTS(GgufTensorType.Q8_0, true);

    private final GgufTensorType type;
    private final boolean routed;

    Mode(GgufTensorType type, boolean routed) {
      this.type = type;
      this.routed = routed;
    }
  }

  private static Path write(Path directory, String name, Mode mode, int[] order) throws Exception {
    return write(directory, name, mode, order, EXPERT_HIDDEN, null);
  }

  private static Path write(Path directory, String name, Mode mode, int[] order, int expertHidden)
      throws Exception {
    return write(directory, name, mode, order, expertHidden, null);
  }

  /**
   * @param outputHeadType when non-null, also write an {@code output.weight} in this type. The
   *     published files carry the head as Q6_K while the token embedding is Q8_0 or Q4_K, so the
   *     head's type is its own and a loader that reused the embedding's type would decode the bytes
   *     as the wrong format.
   */
  private static Path write(
      Path directory,
      String name,
      Mode mode,
      int[] order,
      int expertHidden,
      GgufTensorType outputHeadType)
      throws Exception {
    Random random = new Random(351);
    GgufTensorType expertType = mode.type;
    String architecture = mode.routed ? "qwen35moe" : "qwen35";
    String prefix = architecture + ".";
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", architecture)
            .addString("general.name", "Toy quantized Qwen 3.5 MoE")
            .addUint32(prefix + "embedding_length", DIMENSION)
            .addUint32(prefix + "block_count", 1)
            .addUint32(prefix + "attention.head_count", 2)
            .addUint32(prefix + "attention.head_count_kv", 1)
            .addUint32(prefix + "attention.key_length", HEAD_DIMENSION)
            .addUint32(prefix + "vocab_size", VOCABULARY_SIZE)
            .addUint32(prefix + "context_length", 8)
            .addFloat32(prefix + "rope.freq_base", 10_000.0f)
            .addUint32(prefix + "rope.dimension_count", HEAD_DIMENSION)
            .addFloat32(prefix + "attention.layer_norm_rms_epsilon", 1.0e-6f)
            .addUint32(prefix + "ssm.conv_kernel", 3)
            .addUint32(prefix + "ssm.state_size", HEAD_DIMENSION)
            .addUint32(prefix + "ssm.group_count", 1)
            .addUint32(prefix + "ssm.time_step_rank", 2)
            .addUint32(prefix + "ssm.inner_size", DIMENSION)
            // Interval 1: the single layer is full attention, so no Gated DeltaNet tensors are
            // needed and the fixture stays about the experts.
            .addUint32(prefix + "full_attention_interval", 1)
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
                bytes(ones(DIMENSION)))
            .addTensor(
                "blk.0.attn_norm.weight",
                GgufTensorType.F32,
                new long[] {DIMENSION},
                bytes(ones(DIMENSION)))
            .addTensor(
                "blk.0.post_attention_norm.weight",
                GgufTensorType.F32,
                new long[] {DIMENSION},
                bytes(ones(DIMENSION)))
            .addTensor(
                "blk.0.attn_q_norm.weight",
                GgufTensorType.F32,
                new long[] {HEAD_DIMENSION},
                bytes(ones(HEAD_DIMENSION)))
            .addTensor(
                "blk.0.attn_k_norm.weight",
                GgufTensorType.F32,
                new long[] {HEAD_DIMENSION},
                bytes(ones(HEAD_DIMENSION)));
    if (mode.routed) {
      builder
          .addUint32(prefix + "expert_count", EXPERTS)
          .addUint32(prefix + "expert_used_count", EXPERTS_USED)
          .addUint32(prefix + "expert_feed_forward_length", expertHidden)
          .addUint32(prefix + "expert_shared_feed_forward_length", SHARED_HIDDEN);
    } else {
      builder.addUint32(prefix + "feed_forward_length", expertHidden);
    }
    if (outputHeadType != null) {
      // Deliberately a different type from the token embedding, which is F32 here.
      float[] head = deterministic(1800, DIMENSION * VOCABULARY_SIZE);
      builder.addTensor(
          "output.weight",
          outputHeadType,
          new long[] {DIMENSION, VOCABULARY_SIZE},
          encode(head, outputHeadType));
    }
    // Attention stays F32 so any divergence is attributable to the feed-forward.
    float32Matrix(builder, "blk.0.attn_q.weight", 2 * DIMENSION, DIMENSION, random);
    float32Matrix(builder, "blk.0.attn_k.weight", KEY_DIMENSION, DIMENSION, random);
    float32Matrix(builder, "blk.0.attn_v.weight", KEY_DIMENSION, DIMENSION, random);
    float32Matrix(builder, "blk.0.attn_output.weight", DIMENSION, DIMENSION, random);

    if (!mode.routed) {
      // Expert 0's weights, so a dense fixture and a cloned routed one hold the same numbers.
      quantizedMatrix(builder, "blk.0.ffn_gate.weight", expertHidden, DIMENSION, 1000, expertType);
      quantizedMatrix(builder, "blk.0.ffn_up.weight", expertHidden, DIMENSION, 1100, expertType);
      quantizedMatrix(builder, "blk.0.ffn_down.weight", DIMENSION, expertHidden, 1200, expertType);
      Path denseModel = directory.resolve(name);
      Files.write(denseModel, builder.build());
      return denseModel;
    }
    boolean cloned = mode == Mode.CLONED_EXPERTS;
    float[] gate = new float[EXPERTS * expertHidden * DIMENSION];
    float[] up = new float[EXPERTS * expertHidden * DIMENSION];
    float[] down = new float[EXPERTS * DIMENSION * expertHidden];
    float[] router = new float[EXPERTS * DIMENSION];
    for (int slot = 0; slot < EXPERTS; slot++) {
      int source = cloned ? 0 : order[slot];
      System.arraycopy(
          deterministic(1000 + source, expertHidden * DIMENSION),
          0,
          gate,
          slot * expertHidden * DIMENSION,
          expertHidden * DIMENSION);
      System.arraycopy(
          deterministic(1100 + source, expertHidden * DIMENSION),
          0,
          up,
          slot * expertHidden * DIMENSION,
          expertHidden * DIMENSION);
      System.arraycopy(
          deterministic(1200 + source, DIMENSION * expertHidden),
          0,
          down,
          slot * DIMENSION * expertHidden,
          DIMENSION * expertHidden);
      System.arraycopy(
          deterministic(1300 + source, DIMENSION), 0, router, slot * DIMENSION, DIMENSION);
    }
    builder
        .addTensor(
            "blk.0.ffn_gate_exps.weight",
            expertType,
            new long[] {DIMENSION, expertHidden, EXPERTS},
            encode(gate, expertType))
        .addTensor(
            "blk.0.ffn_up_exps.weight",
            expertType,
            new long[] {DIMENSION, expertHidden, EXPERTS},
            encode(up, expertType))
        .addTensor(
            "blk.0.ffn_down_exps.weight",
            expertType,
            new long[] {expertHidden, DIMENSION, EXPERTS},
            encode(down, expertType));
    // The router is one row per expert over the model width, always F32 in practice.
    float32Matrix(builder, "blk.0.ffn_gate_inp.weight", EXPERTS, DIMENSION, router);
    float32Matrix(
        builder, "blk.0.ffn_gate_inp_shexp.weight", 1, DIMENSION, deterministic(1400, DIMENSION));
    quantizedMatrix(
        builder, "blk.0.ffn_gate_shexp.weight", SHARED_HIDDEN, DIMENSION, 1500, expertType);
    quantizedMatrix(
        builder, "blk.0.ffn_up_shexp.weight", SHARED_HIDDEN, DIMENSION, 1600, expertType);
    // Zeroed for the cloned fixture, so the dense equivalence is about the routed experts only.
    // A Q8_0 block of zeros is a zero scale and zero codes, which dequantizes to exact zeros.
    builder.addTensor(
        "blk.0.ffn_down_shexp.weight",
        expertType,
        new long[] {SHARED_HIDDEN, DIMENSION},
        encode(
            cloned
                ? new float[DIMENSION * SHARED_HIDDEN]
                : deterministic(1700, DIMENSION * SHARED_HIDDEN),
            expertType));

    Path model = directory.resolve(name);
    Files.write(model, builder.build());
    return model;
  }

  private static void quantizedMatrix(
      SyntheticGgufBuilder builder,
      String name,
      int rows,
      int columns,
      int seed,
      GgufTensorType type) {
    builder.addTensor(
        name, type, new long[] {columns, rows}, encode(deterministic(seed, rows * columns), type));
  }

  private static void float32Matrix(
      SyntheticGgufBuilder builder, String name, int rows, int columns, Random random) {
    float32Matrix(builder, name, rows, columns, randomFloats(random, rows * columns));
  }

  private static void float32Matrix(
      SyntheticGgufBuilder builder, String name, int rows, int columns, float[] values) {
    builder.addTensor(name, GgufTensorType.F32, new long[] {columns, rows}, bytes(values));
  }

  /**
   * Encodes as Q8_0, or as the F32 values that encoding round-trips to.
   *
   * <p>The F32 arm deliberately stores the <i>dequantized</i> numbers rather than the originals, so
   * the reference model is the same model and not a nearby one. Comparing against the originals
   * would fold quantization error into the tolerance and hide a real divergence inside it.
   */
  private static byte[] encode(float[] values, GgufTensorType type) {
    if (type == GgufTensorType.F32) {
      return bytes(roundTrip(values));
    }
    if (type != GgufTensorType.Q8_0) {
      throw new IllegalArgumentException("fixture supports F32 and Q8_0 only, not " + type);
    }
    int blocks = values.length / 32;
    ByteBuffer buffer = ByteBuffer.allocate(blocks * 34).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < blocks; block++) {
      float scale = blockScale(values, block);
      buffer.putShort(Float.floatToFloat16(scale));
      float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
      for (int index = 0; index < 32; index++) {
        buffer.put((byte) Math.round(values[block * 32 + index] * inverse));
      }
    }
    return buffer.array();
  }

  private static float[] roundTrip(float[] values) {
    float[] result = new float[values.length];
    for (int block = 0; block < values.length / 32; block++) {
      float scale = blockScale(values, block);
      float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
      // The scale is stored as a half, and the reference dequantizer reads that half back.
      float stored = Float.float16ToFloat(Float.floatToFloat16(scale));
      for (int index = 0; index < 32; index++) {
        int quantized = Math.round(values[block * 32 + index] * inverse);
        result[block * 32 + index] = quantized * stored;
      }
    }
    return result;
  }

  private static float blockScale(float[] values, int block) {
    float largest = 0.0f;
    for (int index = 0; index < 32; index++) {
      largest = Math.max(largest, Math.abs(values[block * 32 + index]));
    }
    return largest / 127.0f;
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
