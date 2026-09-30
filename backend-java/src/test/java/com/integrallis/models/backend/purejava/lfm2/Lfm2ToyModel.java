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
package com.integrallis.models.backend.purejava.lfm2;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * The toy LFM2 model the LFM2 tests share.
 *
 * <p>Three layers in the order convolution, attention, convolution, so both mixers and the boundary
 * between them are exercised. Two seeds are parameters so a single tensor can be varied at a time:
 * a hybrid that quietly ran the wrong mixer would still produce finite, plausible logits.
 *
 * <p>Public because the decoder adapter lives in another package and needs the same model. One
 * fixture rather than two: a second copy would drift, and then the adapter would be tested against
 * a model the forward pass is not.
 */
public final class Lfm2ToyModel {

  static final int DIM = 4;
  static final int LAYERS = 3;
  static final int HEADS = 2;
  static final int KV_HEADS = 1;
  static final int HEAD_DIM = DIM / HEADS;
  static final int HIDDEN = 6;
  static final int VOCAB = 4;
  static final int L_CACHE = 3;
  static final List<Integer> ATTENTION_LAYERS = List.of(1);

  private Lfm2ToyModel() {}

  /** The toy model as a parsed GGUF. */
  public static GgufFile file(int convKernelSeed, int attentionKeySeed) {
    return build(convKernelSeed, attentionKeySeed);
  }

  /** The toy model's bytes, for tests that need it written to a real file. */
  public static byte[] bytes(int convKernelSeed, int attentionKeySeed) {
    return builder(convKernelSeed, attentionKeySeed).build();
  }

  static GgufFile build(int convKernelSeed, int attentionKeySeed) {
    byte[] data = builder(convKernelSeed, attentionKeySeed).build();
    MemorySegment segment = Arena.ofAuto().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  private static SyntheticGgufBuilder builder(int convKernelSeed, int attentionKeySeed) {
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "lfm2")
            .addUint32("lfm2.embedding_length", DIM)
            .addUint32("lfm2.block_count", LAYERS)
            .addUint32("lfm2.attention.head_count", HEADS)
            .addUint32("lfm2.context_length", 64)
            .addUint32("lfm2.feed_forward_length", HIDDEN)
            .addUint32("lfm2.vocab_size", VOCAB)
            .addUint32("lfm2.shortconv.l_cache", L_CACHE)
            .addFloat32("lfm2.attention.layer_norm_rms_epsilon", 1.0e-5f)
            .addFloat32("lfm2.rope.freq_base", 1_000_000.0f);
    List<Integer> kvHeads = new ArrayList<>();
    for (int layer = 0; layer < LAYERS; layer++) {
      kvHeads.add(ATTENTION_LAYERS.contains(layer) ? KV_HEADS : 0);
    }
    builder.addInt32Array("lfm2.attention.head_count_kv", kvHeads);

    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB},
        f32(matrix(VOCAB, DIM, 5)));
    // Named token_embd_norm but used as the FINAL norm; see Lfm2Weights.
    builder.addTensor("token_embd_norm.weight", GgufTensorType.F32, new long[] {DIM}, f32(norm(9)));

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder.addTensor(
          prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, f32(norm(11 + layer)));
      builder.addTensor(
          prefix + "ffn_norm.weight", GgufTensorType.F32, new long[] {DIM}, f32(norm(21 + layer)));
      builder.addTensor(
          prefix + "ffn_gate.weight",
          GgufTensorType.F32,
          new long[] {DIM, HIDDEN},
          f32(matrix(HIDDEN, DIM, 31 + layer)));
      builder.addTensor(
          prefix + "ffn_up.weight",
          GgufTensorType.F32,
          new long[] {DIM, HIDDEN},
          f32(matrix(HIDDEN, DIM, 41 + layer)));
      builder.addTensor(
          prefix + "ffn_down.weight",
          GgufTensorType.F32,
          new long[] {HIDDEN, DIM},
          f32(matrix(DIM, HIDDEN, 51 + layer)));
      if (ATTENTION_LAYERS.contains(layer)) {
        builder.addTensor(
            prefix + "attn_q.weight",
            GgufTensorType.F32,
            new long[] {DIM, HEADS * HEAD_DIM},
            f32(matrix(HEADS * HEAD_DIM, DIM, 61)));
        builder.addTensor(
            prefix + "attn_k.weight",
            GgufTensorType.F32,
            new long[] {DIM, KV_HEADS * HEAD_DIM},
            f32(matrix(KV_HEADS * HEAD_DIM, DIM, attentionKeySeed)));
        builder.addTensor(
            prefix + "attn_v.weight",
            GgufTensorType.F32,
            new long[] {DIM, KV_HEADS * HEAD_DIM},
            f32(matrix(KV_HEADS * HEAD_DIM, DIM, 71)));
        builder.addTensor(
            prefix + "attn_output.weight",
            GgufTensorType.F32,
            new long[] {HEADS * HEAD_DIM, DIM},
            f32(matrix(DIM, HEADS * HEAD_DIM, 81)));
        builder.addTensor(
            prefix + "attn_q_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIM},
            f32(norm(91, HEAD_DIM)));
        builder.addTensor(
            prefix + "attn_k_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIM},
            f32(norm(101, HEAD_DIM)));
      } else {
        builder.addTensor(
            prefix + "shortconv.in_proj.weight",
            GgufTensorType.F32,
            new long[] {DIM, 3 * DIM},
            f32(matrix(3 * DIM, DIM, 111 + layer)));
        builder.addTensor(
            prefix + "shortconv.conv.weight",
            GgufTensorType.F32,
            new long[] {L_CACHE, DIM},
            f32(matrix(DIM, L_CACHE, convKernelSeed + layer)));
        builder.addTensor(
            prefix + "shortconv.out_proj.weight",
            GgufTensorType.F32,
            new long[] {DIM, DIM},
            f32(matrix(DIM, DIM, 121 + layer)));
      }
    }

    return builder;
  }

  static float[] norm(int seed) {
    return norm(seed, DIM);
  }

  static float[] norm(int seed, int length) {
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = 0.8f + ((seed + index * 3) % 7) * 0.05f;
    }
    return values;
  }

  static float[] matrix(int rows, int columns, int seed) {
    float[] values = new float[rows * columns];
    for (int index = 0; index < values.length; index++) {
      values[index] = (((seed + index * 11) % 19) - 9) * 0.045f;
    }
    return values;
  }

  static byte[] f32(float[] values) {
    byte[] data = new byte[values.length * 4];
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < values.length; index++) {
      buffer.putFloat(index * 4, values[index]);
    }
    return data;
  }
}
