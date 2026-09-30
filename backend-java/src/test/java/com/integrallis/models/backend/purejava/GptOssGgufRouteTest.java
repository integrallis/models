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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The GGUF release of GPT-OSS, through the backend's own dispatch.
 *
 * <p>GPT-OSS already qualified through the safetensors path, so what is new here is the artifact
 * format: K-quant attention matteres rather than BF16, three stacked expert tensors rather than
 * fused blocks and scales, biases on every attention projection, and a sliding-window pattern that
 * the header does not state and the loader has to derive.
 *
 * <p>Every width here is a multiple of 32 because MXFP4 blocks 32 values, and the expert tensors
 * are sliced on whole rows.
 */
@Tag("unit")
class GptOssGgufRouteTest {

  private static final int HIDDEN = 32;
  private static final int LAYERS = 2;
  private static final int HEADS = 4;
  private static final int KV_HEADS = 2;
  private static final int HEAD_DIM = 8;
  private static final int INTERMEDIATE = 32;
  private static final int EXPERTS = 2;
  private static final int EXPERTS_USED = 2;
  private static final int VOCAB = 8;
  private static final int SLIDING_WINDOW = 2;

  @Test
  void theBackendLoadsAGptOssGgufUnderItsOwnArchitecture(@TempDir Path directory) throws Exception {
    Path model = write(directory);

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("gpt-oss");
      assertThat(backend.metadata().numLayers()).isEqualTo(LAYERS);
      assertThat(backend.metadata().embeddingDim()).isEqualTo(HIDDEN);

      float[] logits = backend.prefill(new int[] {1, 2, 3}, 0);

      assertThat(logits).hasSize(VOCAB);
      for (float logit : logits) {
        assertThat(Float.isFinite(logit))
            .describedAs("a misread expert or a missing bias shows up as NaN here")
            .isTrue();
      }
    }
  }

  /**
   * A golden vector for the whole GGUF path.
   *
   * <p>The perturbation tests below prove individual tensors are read; none of them pins how the
   * expert tensors are <b>wired</b>. Swapping the gate and up stacks, for instance, passes every
   * other test here -- SwiGLU is asymmetric so the output changes, but nothing was comparing it to
   * anything. A golden vector makes any silent rewiring of this path a failure that has to be
   * explained.
   *
   * <p>Captured from this implementation, so it pins behaviour rather than proving correctness; the
   * safetensors path is what is checked against an oracle. Its value is that a change here cannot
   * be accidental.
   */
  @Test
  void theGgufPathOutputIsPinned(@TempDir Path directory) throws Exception {
    float[] expected = {
      0.118602686f,
      0.0864715f,
      0.20693126f,
      0.02026198f,
      0.21622775f,
      -0.27813503f,
      -0.101803094f,
      0.058025442f
    };

    float[] actual = logitsOf(write(directory, java.util.Map.of(), "pinned.gguf"));

    assertThat(actual).hasSameSizeAs(expected);
    for (int index = 0; index < expected.length; index++) {
      assertThat(actual[index])
          .describedAs("pinned gpt-oss GGUF logit %s", index)
          .isEqualTo(expected[index], within(1.0e-5f));
    }
  }

  /**
   * The attention sinks must reach the output.
   *
   * <p>A sink is one learned logit per head added to the softmax denominator, so it changes every
   * attention distribution. Loading zeros instead would still give finite, plausible logits --
   * which is all the load test above checks -- so this perturbs them and requires the output to
   * move.
   */
  @Test
  void theAttentionSinksReachTheOutput(@TempDir Path directory) throws Exception {
    float[] baseline = logitsOf(write(directory, java.util.Map.of(), "base.gguf"));

    float[] sinks = new float[HEADS];
    java.util.Arrays.fill(sinks, 4.0f);
    float[] changed =
        logitsOf(
            write(directory, java.util.Map.of("blk.0.attn_sinks.weight", sinks), "sinks.gguf"));

    assertThat(changed).isNotEqualTo(baseline);
  }

  /** And the untied head: the GGUF release carries its own output.weight, not the embedding. */
  @Test
  void theUntiedOutputHeadIsUsed(@TempDir Path directory) throws Exception {
    float[] baseline = logitsOf(write(directory, java.util.Map.of(), "base.gguf"));

    float[] head = values(VOCAB * HIDDEN, 5);
    head[0] += 0.75f;
    float[] changed =
        logitsOf(write(directory, java.util.Map.of("output.weight", head), "head.gguf"));

    assertThat(changed)
        .describedAs("reading the token embedding instead would ignore this edit entirely")
        .isNotEqualTo(baseline);
  }

  private static float[] logitsOf(Path model) throws Exception {
    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      return backend.prefill(new int[] {1, 2, 3}, 0).clone();
    }
  }

  /**
   * The derived sliding-window pattern: even layers slide, odd layers attend to everything.
   *
   * <p>GGUF publishes the window size but not the pattern. The reference defaults this family to a
   * period of 2 with the sliding layer first, so the loader derives it -- and a wrong parity would
   * give the wrong layers a bounded context, which is not visible in the output.
   */
  @Test
  void theSlidingPatternIsDerivedWithTheSlidingLayerFirst(@TempDir Path directory)
      throws Exception {
    Path model = write(directory);

    try (var arena = java.lang.foreign.Arena.ofConfined()) {
      var file = com.integrallis.models.backend.purejava.gguf.GgufParser.parse(model, arena);
      var config =
          com.integrallis.models.backend.purejava.gptoss.GptOssConfig.fromGgufMetadata(
              file.metadata());

      assertThat(config.usesSlidingAttention(0)).isTrue();
      assertThat(config.usesSlidingAttention(1)).isFalse();
      assertThat(config.slidingWindow()).isEqualTo(SLIDING_WINDOW);
      // And the constants GGUF does not publish come through as the architecture's own.
      assertThat(config.swigluLimit()).isEqualTo(7.0f);
      assertThat(config.hiddenActAlpha()).isEqualTo(1.702f);
      assertThat(config.tieWordEmbeddings())
          .describedAs("the GGUF release carries its own output.weight")
          .isFalse();
    }
  }

  private static Path write(Path directory) throws Exception {
    return write(directory, java.util.Map.of(), "toy-gpt-oss.gguf");
  }

  /**
   * @param overrides tensors to write with values of the caller's choosing, applied at write time
   *     -- the builder appends, so rewriting a name later would leave two entries and the loader
   *     reads the first
   */
  private static Path write(
      Path directory, java.util.Map<String, float[]> overrides, String fileName) throws Exception {
    active = overrides;
    List<String> tokens = new ArrayList<>();
    List<Float> scores = new ArrayList<>();
    for (int index = 0; index < VOCAB; index++) {
      tokens.add("t" + index);
      scores.add(0.0f);
    }

    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "gpt-oss")
            .addString("general.name", "Toy GPT-OSS")
            .addUint32("gpt-oss.block_count", LAYERS)
            .addUint32("gpt-oss.embedding_length", HIDDEN)
            .addUint32("gpt-oss.attention.head_count", HEADS)
            .addUint32("gpt-oss.attention.head_count_kv", KV_HEADS)
            .addUint32("gpt-oss.attention.key_length", HEAD_DIM)
            .addUint32("gpt-oss.attention.value_length", HEAD_DIM)
            .addUint32("gpt-oss.context_length", 16)
            .addUint32("gpt-oss.feed_forward_length", INTERMEDIATE)
            .addUint32("gpt-oss.expert_feed_forward_length", INTERMEDIATE)
            .addUint32("gpt-oss.expert_count", EXPERTS)
            .addUint32("gpt-oss.expert_used_count", EXPERTS_USED)
            .addUint32("gpt-oss.attention.sliding_window", SLIDING_WINDOW)
            .addFloat32("gpt-oss.rope.freq_base", 150_000.0f)
            .addFloat32("gpt-oss.attention.layer_norm_rms_epsilon", 1.0e-5f)
            .addStringArray("tokenizer.ggml.tokens", tokens)
            .addFloat32Array("tokenizer.ggml.scores", scores)
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1);

    matrix(builder, "token_embd.weight", VOCAB, HIDDEN, 3);
    matrix(builder, "output.weight", VOCAB, HIDDEN, 5);
    vector(builder, "output_norm.weight", HIDDEN, 7);

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      int queryDim = HEADS * HEAD_DIM;
      int kvDim = KV_HEADS * HEAD_DIM;
      vector(builder, prefix + "attn_norm.weight", HIDDEN, 10 + layer);
      matrix(builder, prefix + "attn_q.weight", queryDim, HIDDEN, 20 + layer);
      vector(builder, prefix + "attn_q.bias", queryDim, 25 + layer);
      matrix(builder, prefix + "attn_k.weight", kvDim, HIDDEN, 30 + layer);
      vector(builder, prefix + "attn_k.bias", kvDim, 35 + layer);
      matrix(builder, prefix + "attn_v.weight", kvDim, HIDDEN, 40 + layer);
      vector(builder, prefix + "attn_v.bias", kvDim, 45 + layer);
      matrix(builder, prefix + "attn_output.weight", HIDDEN, queryDim, 50 + layer);
      vector(builder, prefix + "attn_output.bias", HIDDEN, 55 + layer);
      // One sink per query head.
      vector(builder, prefix + "attn_sinks.weight", HEADS, 60 + layer);
      vector(builder, prefix + "post_attention_norm.weight", HIDDEN, 65 + layer);
      matrix(builder, prefix + "ffn_gate_inp.weight", EXPERTS, HIDDEN, 70 + layer);
      vector(builder, prefix + "ffn_gate_inp.bias", EXPERTS, 75 + layer);
      // MXFP4, stacked per expert, exactly as the published file stores them.
      mxfp4(builder, prefix + "ffn_gate_exps.weight", INTERMEDIATE, HIDDEN, EXPERTS, 1);
      mxfp4(builder, prefix + "ffn_up_exps.weight", INTERMEDIATE, HIDDEN, EXPERTS, 2);
      mxfp4(builder, prefix + "ffn_down_exps.weight", HIDDEN, INTERMEDIATE, EXPERTS, 3);
      stackedVector(builder, prefix + "ffn_gate_exps.bias", INTERMEDIATE, EXPERTS, 80 + layer);
      stackedVector(builder, prefix + "ffn_up_exps.bias", INTERMEDIATE, EXPERTS, 85 + layer);
      stackedVector(builder, prefix + "ffn_down_exps.bias", HIDDEN, EXPERTS, 90 + layer);
    }

    Path model = directory.resolve(fileName);
    Files.write(model, builder.build());
    return model;
  }

  /** Tensors a test wants written with values of its own, keyed by name. */
  private static java.util.Map<String, float[]> active = java.util.Map.of();

  private static void matrix(
      SyntheticGgufBuilder builder, String name, int rows, int columns, int seed) {
    builder.addTensor(
        name,
        GgufTensorType.F32,
        new long[] {columns, rows},
        bytes(active.getOrDefault(name, values(rows * columns, seed))));
  }

  private static void vector(SyntheticGgufBuilder builder, String name, int length, int seed) {
    builder.addTensor(
        name,
        GgufTensorType.F32,
        new long[] {length},
        bytes(active.getOrDefault(name, values(length, seed))));
  }

  /** A stacked bias: one row of {@code width} per expert. */
  private static void stackedVector(
      SyntheticGgufBuilder builder, String name, int width, int experts, int seed) {
    builder.addTensor(
        name,
        GgufTensorType.F32,
        new long[] {width, experts},
        bytes(values(width * experts, seed)));
  }

  /**
   * A stacked MXFP4 expert tensor: one byte of exponent then sixteen of nibble pairs per 32 values.
   *
   * <p>Codes vary by position so the experts are not identical -- identical experts would make the
   * routing weights the only thing distinguishing them, and a mis-sliced stack would go unnoticed.
   */
  private static void mxfp4(
      SyntheticGgufBuilder builder, String name, int rows, int columns, int experts, int seed) {
    int blocksPerRow = columns / 32;
    int bytesPerBlock = 17;
    byte[] data = new byte[experts * rows * blocksPerRow * bytesPerBlock];
    int at = 0;
    for (int expert = 0; expert < experts; expert++) {
      for (int row = 0; row < rows; row++) {
        for (int block = 0; block < blocksPerRow; block++) {
          // An exponent byte near 127 keeps the values around unity.
          data[at++] = (byte) (125 + ((seed + expert + row) % 5));
          for (int pair = 0; pair < 16; pair++) {
            int low = (seed + expert * 3 + row * 5 + pair * 7) % 16;
            int high = (seed + expert * 5 + row * 3 + pair * 11 + 2) % 16;
            data[at++] = (byte) ((high << 4) | low);
          }
        }
      }
    }
    builder.addTensor(name, GgufTensorType.MXFP4, new long[] {columns, rows, experts}, data);
  }

  private static float[] values(int count, int seed) {
    float[] result = new float[count];
    for (int index = 0; index < count; index++) {
      result[index] = (((seed + index * 7) % 13) - 6) * 0.05f;
    }
    return result;
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
