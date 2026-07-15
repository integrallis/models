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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LlamaForwardPassTest {

  // Nano model: 2 layers, dim=16, heads=2, kvHeads=1, hiddenDim=32, vocabSize=32
  private static final int DIM = 16;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HIDDEN_DIM = 32;
  private static final int VOCAB_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT = 64;

  private GgufFile buildNanoModel(Random rng) {
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addUint32("llama.embedding_length", DIM)
            .addUint32("llama.block_count", LAYERS)
            .addUint32("llama.attention.head_count", HEADS)
            .addUint32("llama.attention.head_count_kv", KV_HEADS)
            .addUint32("llama.vocab_size", VOCAB_SIZE)
            .addUint32("llama.context_length", CONTEXT)
            .addUint32("llama.feed_forward_length", HIDDEN_DIM);

    // token_embd.weight: [vocab_size x dim]
    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));
    // output_norm.weight: [dim]
    builder.addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
    // output.weight: [vocab_size x dim]
    builder.addTensor(
        "output.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));

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
          new long[] {DIM, KV_HEADS * (DIM / HEADS)},
          randomF32(rng, KV_HEADS * (DIM / HEADS) * DIM));
      builder.addTensor(
          prefix + "attn_v.weight",
          GgufTensorType.F32,
          new long[] {DIM, KV_HEADS * (DIM / HEADS)},
          randomF32(rng, KV_HEADS * (DIM / HEADS) * DIM));
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
    var segment = Arena.ofConfined().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  private GgufFile buildQ4NanoModel(Random rng) {
    int dim = 32;
    int hiddenDim = 64;
    int headDim = dim / HEADS;
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "qwen3")
            .addUint32("llama.embedding_length", dim)
            .addUint32("llama.block_count", LAYERS)
            .addUint32("llama.attention.head_count", HEADS)
            .addUint32("llama.attention.head_count_kv", KV_HEADS)
            .addUint32("llama.vocab_size", VOCAB_SIZE)
            .addUint32("llama.context_length", CONTEXT)
            .addUint32("llama.feed_forward_length", hiddenDim)
            .addTensor(
                "token_embd.weight",
                GgufTensorType.F32,
                new long[] {dim, VOCAB_SIZE},
                randomF32(rng, VOCAB_SIZE * dim))
            .addTensor("output_norm.weight", GgufTensorType.F32, new long[] {dim}, onesF32(dim))
            .addTensor(
                "output.weight",
                GgufTensorType.F32,
                new long[] {dim, VOCAB_SIZE},
                randomF32(rng, VOCAB_SIZE * dim));

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder
          .addTensor(
              prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {dim}, onesF32(dim))
          .addTensor(
              prefix + "attn_q.weight",
              GgufTensorType.Q4_0,
              new long[] {dim, dim},
              randomQ4(rng, dim * dim))
          .addTensor(
              prefix + "attn_q.bias", GgufTensorType.F32, new long[] {dim}, randomF32(rng, dim))
          .addTensor(
              prefix + "attn_q_norm.weight",
              GgufTensorType.F32,
              new long[] {headDim},
              onesF32(headDim))
          .addTensor(
              prefix + "attn_k.weight",
              GgufTensorType.Q4_0,
              new long[] {dim, headDim},
              randomQ4(rng, headDim * dim))
          .addTensor(
              prefix + "attn_k.bias",
              GgufTensorType.F32,
              new long[] {headDim},
              randomF32(rng, headDim))
          .addTensor(
              prefix + "attn_k_norm.weight",
              GgufTensorType.F32,
              new long[] {headDim},
              onesF32(headDim))
          .addTensor(
              prefix + "attn_v.weight",
              GgufTensorType.Q4_0,
              new long[] {dim, headDim},
              randomQ4(rng, headDim * dim))
          .addTensor(
              prefix + "attn_v.bias",
              GgufTensorType.F32,
              new long[] {headDim},
              randomF32(rng, headDim))
          .addTensor(
              prefix + "attn_output.weight",
              GgufTensorType.Q4_0,
              new long[] {dim, dim},
              randomQ4(rng, dim * dim))
          .addTensor(prefix + "ffn_norm.weight", GgufTensorType.F32, new long[] {dim}, onesF32(dim))
          .addTensor(
              prefix + "ffn_gate.weight",
              GgufTensorType.Q4_0,
              new long[] {dim, hiddenDim},
              randomQ4(rng, hiddenDim * dim))
          .addTensor(
              prefix + "ffn_up.weight",
              GgufTensorType.Q4_0,
              new long[] {dim, hiddenDim},
              randomQ4(rng, hiddenDim * dim))
          .addTensor(
              prefix + "ffn_down.weight",
              GgufTensorType.Q4_0,
              new long[] {hiddenDim, dim},
              randomQ4(rng, dim * hiddenDim));
    }

    byte[] data = builder.build();
    MemorySegment segment = Arena.ofConfined().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  @Nested
  class NanoModel {

    @Test
    void producesLogitsOfVocabSize() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      float[] logits = forwardPass.forward(1, 0);

      assertThat(logits).hasSize(VOCAB_SIZE);
    }

    @Test
    void deterministicForSameInput() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);

      KvCache cache1 = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass pass1 = new LlamaForwardPass(config, weights, cache1);
      float[] logits1 = pass1.forward(5, 0);

      KvCache cache2 = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass pass2 = new LlamaForwardPass(config, weights, cache2);
      float[] logits2 = pass2.forward(5, 0);

      assertThat(logits1).containsExactly(logits2);
    }

    @Test
    void prefillMatchesTokenAtATimeForwardPass() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13};

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(config.numLayers(), config.contextLength(), config.kvDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(config.numLayers(), config.contextLength(), config.kvDim()));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(actual).containsExactly(expected);
      assertThat(batched.forward(17, tokens.length)).hasSize(VOCAB_SIZE);
    }

    @Test
    void q4PrefillUsesBatchedKernelAndPreservesAutoregressiveState() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(config.numLayers(), config.contextLength(), config.kvDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(config.numLayers(), config.contextLength(), config.kvDim()));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(batched.usesBatchedPrefill()).isTrue();
      assertThat(actual).containsExactly(expected);
      assertThat(batched.forward(3, tokens.length))
          .containsExactly(sequential.forward(3, tokens.length));
    }

    @Test
    void transientForwardReusesLogitStorageWithoutChangingForwardSnapshots() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaForwardPass forwardPass =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(config.numLayers(), config.contextLength(), config.kvDim()));

      float[] snapshot = forwardPass.forward(5, 0);
      float[] snapshotValues = snapshot.clone();
      float[] firstTransient = forwardPass.forwardTransient(7, 1);
      float[] firstTransientValues = firstTransient.clone();
      float[] secondTransient = forwardPass.forwardTransient(11, 2);

      assertThat(secondTransient).isSameAs(firstTransient);
      assertThat(secondTransient).isNotSameAs(snapshot);
      assertThat(firstTransient).isNotEqualTo(firstTransientValues);
      assertThat(snapshot).containsExactly(snapshotValues);
    }

    @Test
    void differentPositionsProduceDifferentLogits() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      float[] logits0 = forwardPass.forward(5, 0);
      float[] logits1 = forwardPass.forward(5, 1);

      assertThat(logits0).isNotEqualTo(logits1);
    }

    @Test
    void rejectsNonSequentialPositions() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      assertThatThrownBy(() -> forwardPass.forward(5, 1))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("expected 0");
    }

    @Test
    void cacheGrowthAndResetPreserveSequenceResults() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);
      int[] tokens = new int[20];
      for (int i = 0; i < tokens.length; i++) {
        tokens[i] = i % VOCAB_SIZE;
      }

      float[] first = runSequence(forwardPass, tokens);
      assertThat(cache.allocatedSequenceCapacity()).isEqualTo(32);

      forwardPass.reset();
      float[] second = runSequence(forwardPass, tokens);

      assertThat(second).containsExactly(first);
    }
  }

  private static float[] runSequence(LlamaForwardPass forwardPass, int[] tokens) {
    float[] logits = null;
    for (int position = 0; position < tokens.length; position++) {
      logits = forwardPass.forward(tokens[position], position);
    }
    return logits;
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

  private static byte[] randomQ4(Random rng, int valueCount) {
    if (valueCount % 32 != 0) {
      throw new IllegalArgumentException("Q4_0 value count must be a multiple of 32");
    }
    byte[] data = new byte[(valueCount / 32) * 18];
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < valueCount / 32; block++) {
      int offset = block * 18;
      buffer.putShort(offset, Float.floatToFloat16(0.01f));
      for (int packed = 0; packed < 16; packed++) {
        data[offset + 2 + packed] = (byte) rng.nextInt(256);
      }
    }
    return data;
  }
}
