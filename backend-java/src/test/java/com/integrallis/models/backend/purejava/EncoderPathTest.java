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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.Pooling;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Contract tests for the encoder path against a synthetic {@code gemma-embedding} nano model.
 *
 * <p>Random weights carry no semantics, so nothing here asserts that the vectors mean anything —
 * the equivalence gate against llama.cpp does that on real weights. What these cover is the wiring
 * that would otherwise only be exercised by a 300M model: which pass the backend selects, that an
 * encoder refuses to generate, and that the dense head is loaded and applied rather than skipped.
 */
@Tag("unit")
class EncoderPathTest {

  private static final int DIM = 32;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HEAD_DIM = DIM / HEADS;
  private static final int HIDDEN_DIM = 32;
  private static final int VOCAB_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT = 64;
  private static final int DENSE_INNER = 64;
  private static final String ARCH = "gemma-embedding";

  private static final String TEXT = "t5 t7 t11";

  /**
   * Writes a nano encoder.
   *
   * @param withDenseHead whether to include the sentence-transformer projection tensors
   * @param withDenseTail whether to include the second of the two, used to prove a half-present
   *     head is rejected rather than partly applied
   */
  private static Path buildNanoEncoder(
      Path dir, Random rng, boolean withDenseHead, boolean withDenseTail, String name)
      throws IOException {
    List<String> tokens = new ArrayList<>();
    List<Float> scores = new ArrayList<>();
    for (int index = 0; index < VOCAB_SIZE; index++) {
      tokens.add("t" + index);
      scores.add(0.0f);
    }

    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.name", "NanoEncoder")
            .addString("general.architecture", ARCH)
            .addUint32(ARCH + ".embedding_length", DIM)
            .addUint32(ARCH + ".block_count", LAYERS)
            .addUint32(ARCH + ".attention.head_count", HEADS)
            .addUint32(ARCH + ".attention.head_count_kv", KV_HEADS)
            .addUint32(ARCH + ".attention.key_length", HEAD_DIM)
            .addUint32(ARCH + ".attention.value_length", HEAD_DIM)
            .addUint32(ARCH + ".vocab_size", VOCAB_SIZE)
            .addUint32(ARCH + ".context_length", CONTEXT)
            .addUint32(ARCH + ".feed_forward_length", HIDDEN_DIM)
            .addUint32(ARCH + ".attention.sliding_window", 8)
            .addFloat32(ARCH + ".attention.layer_norm_rms_epsilon", 1e-6f)
            .addStringArray("tokenizer.ggml.tokens", tokens)
            .addFloat32Array("tokenizer.ggml.scores", scores)
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1);

    if (withDenseHead) {
      builder
          .addUint32(ARCH + ".dense_2_feat_in", DIM)
          .addUint32(ARCH + ".dense_2_feat_out", DENSE_INNER)
          .addUint32(ARCH + ".dense_3_feat_in", DENSE_INNER)
          .addUint32(ARCH + ".dense_3_feat_out", DIM)
          .addTensor(
              "dense_2.weight",
              GgufTensorType.F32,
              new long[] {DIM, DENSE_INNER},
              randomF32(rng, DIM * DENSE_INNER));
      if (withDenseTail) {
        builder.addTensor(
            "dense_3.weight",
            GgufTensorType.F32,
            new long[] {DENSE_INNER, DIM},
            randomF32(rng, DENSE_INNER * DIM));
      }
    }

    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));
    builder.addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));

    int kvDim = KV_HEADS * HEAD_DIM;
    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder
          .addTensor(
              prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM))
          .addTensor(
              prefix + "attn_q.weight",
              GgufTensorType.F32,
              new long[] {DIM, DIM},
              randomF32(rng, DIM * DIM))
          .addTensor(
              prefix + "attn_k.weight",
              GgufTensorType.F32,
              new long[] {DIM, kvDim},
              randomF32(rng, kvDim * DIM))
          .addTensor(
              prefix + "attn_v.weight",
              GgufTensorType.F32,
              new long[] {DIM, kvDim},
              randomF32(rng, kvDim * DIM))
          .addTensor(
              prefix + "attn_q_norm.weight",
              GgufTensorType.F32,
              new long[] {HEAD_DIM},
              onesF32(HEAD_DIM))
          .addTensor(
              prefix + "attn_k_norm.weight",
              GgufTensorType.F32,
              new long[] {HEAD_DIM},
              onesF32(HEAD_DIM))
          .addTensor(
              prefix + "attn_output.weight",
              GgufTensorType.F32,
              new long[] {DIM, DIM},
              randomF32(rng, DIM * DIM))
          .addTensor(
              prefix + "post_attention_norm.weight",
              GgufTensorType.F32,
              new long[] {DIM},
              onesF32(DIM))
          .addTensor(prefix + "ffn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM))
          .addTensor(
              prefix + "ffn_gate.weight",
              GgufTensorType.F32,
              new long[] {DIM, HIDDEN_DIM},
              randomF32(rng, HIDDEN_DIM * DIM))
          .addTensor(
              prefix + "ffn_up.weight",
              GgufTensorType.F32,
              new long[] {DIM, HIDDEN_DIM},
              randomF32(rng, HIDDEN_DIM * DIM))
          .addTensor(
              prefix + "ffn_down.weight",
              GgufTensorType.F32,
              new long[] {HIDDEN_DIM, DIM},
              randomF32(rng, DIM * HIDDEN_DIM))
          .addTensor(
              prefix + "post_ffw_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
    }

    Path modelPath = dir.resolve(name);
    Files.write(modelPath, builder.build());
    return modelPath;
  }

  private static Path nanoEncoder(Path dir) throws IOException {
    return buildNanoEncoder(dir, new Random(42), true, true, "nano-encoder.gguf");
  }

  @Test
  void loadsAsAnEncoderRatherThanADecoder(@TempDir Path dir) throws IOException {
    try (PureJavaBackend backend = PureJavaBackend.load(nanoEncoder(dir))) {
      assertThat(backend.metadata().modelFamily()).isEqualTo(ARCH);
      assertThat(backend.supportsSequenceEmbedding()).isTrue();
    }
  }

  @Test
  void refusesToGenerate(@TempDir Path dir) throws IOException {
    try (PureJavaBackend backend = PureJavaBackend.load(nanoEncoder(dir))) {
      // The alternative is worse than an exception: an encoder has no language model head, so any
      // value returned here would be meaningless in a shape that looks usable.
      assertThatThrownBy(() -> backend.forward(5, 0))
          .isInstanceOf(UnsupportedOperationException.class)
          .hasMessageContaining("encoder");
      assertThatThrownBy(() -> backend.prefill(new int[] {5, 7}, 0))
          .isInstanceOf(UnsupportedOperationException.class);
      assertThatThrownBy(backend::openSession).isInstanceOf(UnsupportedOperationException.class);
    }
  }

  @Test
  void embedsWholeSequencesToTheModelWidth(@TempDir Path dir) throws IOException {
    try (PureJavaBackend backend = PureJavaBackend.load(nanoEncoder(dir));
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend).normalize(true).build()) {
      float[] vector = embedding.embed(TEXT);

      assertThat(vector).hasSize(DIM);
      assertThat(magnitude(vector)).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-5));
    }
  }

  @Test
  void embeddingIsIndependentOfCallOrder(@TempDir Path dir) throws IOException {
    try (PureJavaBackend backend = PureJavaBackend.load(nanoEncoder(dir));
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend).normalize(true).build()) {
      float[] first = embedding.embed(TEXT);
      embedding.embed("t13 t17 t3 t9");
      float[] again = embedding.embed(TEXT);

      assertThat(again).containsExactly(first);
    }
  }

  @Test
  void rejectsAPoolingChoiceTheModelAlreadyMakes(@TempDir Path dir) throws IOException {
    try (PureJavaBackend backend = PureJavaBackend.load(nanoEncoder(dir))) {
      // Honouring it would return a vector of the right width computed the wrong way, which
      // retrieval absorbs as slightly worse results rather than reporting.
      assertThatThrownBy(
              () -> GgufEmbeddingBackend.builder(backend).pooling(Pooling.LAST_TOKEN).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("pools internally");
    }
  }

  @Test
  void appliesTheDenseProjectionHead(@TempDir Path dir) throws IOException {
    Path withHead = buildNanoEncoder(dir, new Random(42), true, true, "with-head.gguf");
    Path withoutHead = buildNanoEncoder(dir, new Random(42), false, false, "without-head.gguf");

    float[] projected = embedOnce(withHead);
    float[] unprojected = embedOnce(withoutHead);

    // Same weights and same seed either side, so a head that was loaded but never applied would
    // produce identical vectors and pass unnoticed.
    assertThat(projected).isNotEqualTo(unprojected);
  }

  @Test
  void rejectsAHalfPresentDenseHead(@TempDir Path dir) throws IOException {
    Path halfHead = buildNanoEncoder(dir, new Random(42), true, false, "half-head.gguf");

    assertThatThrownBy(() -> PureJavaBackend.load(halfHead))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("complete");
  }

  @Test
  void refusesSequencesLongerThanTheTrainedContext(@TempDir Path dir) throws IOException {
    try (PureJavaBackend backend = PureJavaBackend.load(nanoEncoder(dir))) {
      int[] tooLong = new int[CONTEXT + 1];

      assertThatThrownBy(() -> backend.embedSequence(tooLong))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("context length");
    }
  }

  private static float[] embedOnce(Path model) {
    try (PureJavaBackend backend = PureJavaBackend.load(model);
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend).normalize(true).build()) {
      return embedding.embed(TEXT);
    }
  }

  private static double magnitude(float[] vector) {
    double sum = 0;
    for (float value : vector) {
      sum += (double) value * value;
    }
    return Math.sqrt(sum);
  }

  private static byte[] randomF32(Random rng, int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      buffer.putFloat((rng.nextFloat() - 0.5f) * 0.2f);
    }
    return data;
  }

  private static byte[] onesF32(int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      buffer.putFloat(1.0f);
    }
    return data;
  }
}
