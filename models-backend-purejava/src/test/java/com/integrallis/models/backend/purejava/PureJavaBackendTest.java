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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class PureJavaBackendTest {

  private static final int DIM = 16;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HIDDEN_DIM = 32;
  private static final int VOCAB_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT = 64;

  static Path buildNanoModelFile(Path dir, Random rng) throws IOException {
    List<String> tokens = new ArrayList<>();
    for (int i = 0; i < VOCAB_SIZE; i++) {
      tokens.add("t" + i);
    }

    List<Float> scores = new ArrayList<>();
    for (int i = 0; i < VOCAB_SIZE; i++) {
      scores.add(0.0f);
    }

    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.name", "NanoTest")
            .addString("general.architecture", "llama")
            .addUint32("llama.embedding_length", DIM)
            .addUint32("llama.block_count", LAYERS)
            .addUint32("llama.attention.head_count", HEADS)
            .addUint32("llama.attention.head_count_kv", KV_HEADS)
            .addUint32("llama.vocab_size", VOCAB_SIZE)
            .addUint32("llama.context_length", CONTEXT)
            .addUint32("llama.feed_forward_length", HIDDEN_DIM)
            .addStringArray("tokenizer.ggml.tokens", tokens)
            .addFloat32Array("tokenizer.ggml.scores", scores)
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1);

    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));
    builder.addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
    builder.addTensor(
        "output.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));

    int kvDim = KV_HEADS * (DIM / HEADS);
    for (int l = 0; l < LAYERS; l++) {
      String prefix = "blk." + l + ".";
      builder.addTensor(
          prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
      builder.addTensor(
          prefix + "attn_q.weight",
          GgufTensorType.F32,
          new long[] {DIM, DIM},
          randomF32(rng, DIM * DIM));
      builder.addTensor(
          prefix + "attn_k.weight",
          GgufTensorType.F32,
          new long[] {DIM, kvDim},
          randomF32(rng, kvDim * DIM));
      builder.addTensor(
          prefix + "attn_v.weight",
          GgufTensorType.F32,
          new long[] {DIM, kvDim},
          randomF32(rng, kvDim * DIM));
      builder.addTensor(
          prefix + "attn_output.weight",
          GgufTensorType.F32,
          new long[] {DIM, DIM},
          randomF32(rng, DIM * DIM));
      builder.addTensor(
          prefix + "ffn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
      builder.addTensor(
          prefix + "ffn_gate.weight",
          GgufTensorType.F32,
          new long[] {DIM, HIDDEN_DIM},
          randomF32(rng, HIDDEN_DIM * DIM));
      builder.addTensor(
          prefix + "ffn_up.weight",
          GgufTensorType.F32,
          new long[] {DIM, HIDDEN_DIM},
          randomF32(rng, HIDDEN_DIM * DIM));
      builder.addTensor(
          prefix + "ffn_down.weight",
          GgufTensorType.F32,
          new long[] {HIDDEN_DIM, DIM},
          randomF32(rng, DIM * HIDDEN_DIM));
    }

    byte[] data = builder.build();
    Path modelPath = dir.resolve("nano.gguf");
    Files.write(modelPath, data);
    return modelPath;
  }

  @Nested
  class LoadAndInfer {

    @Test
    void loadsModelAndRunsForward(@TempDir Path dir) throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(42));

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        assertThat(backend.name()).isEqualTo("pure-java");
        assertThat(backend.metadata().modelName()).isEqualTo("NanoTest");
        assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
        assertThat(backend.tokenizer().vocabSize()).isEqualTo(VOCAB_SIZE);

        float[] logits = backend.forward(5, 0);
        assertThat(logits).hasSize(VOCAB_SIZE);
      }
    }

    @Test
    void nonExistentFileThrows(@TempDir Path dir) {
      Path noFile = dir.resolve("missing.gguf");
      assertThatThrownBy(() -> PureJavaBackend.load(noFile))
          .isInstanceOf(UncheckedIOException.class);
    }
  }

  private static byte[] randomF32(Random rng, int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(i * 4, (rng.nextFloat() - 0.5f) * 0.1f);
    }
    return data;
  }

  private static byte[] onesF32(int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(i * 4, 1.0f);
    }
    return data;
  }
}
