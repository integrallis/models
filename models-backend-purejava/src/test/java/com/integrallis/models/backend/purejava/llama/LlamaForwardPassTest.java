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
