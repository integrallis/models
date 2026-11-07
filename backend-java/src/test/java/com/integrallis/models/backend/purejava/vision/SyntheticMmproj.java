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
package com.integrallis.models.backend.purejava.vision;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/** Shared helpers for creating synthetic mmproj GGUF fixtures in vision tests. */
final class SyntheticMmproj {

  static final int DEFAULT_IMAGE_SIZE = 224;
  static final int DEFAULT_PATCH_SIZE = 14;
  static final int DEFAULT_HIDDEN_SIZE = 384;
  static final int DEFAULT_NUM_LAYERS = 6;
  static final int DEFAULT_NUM_HEADS = 6;
  static final int DEFAULT_INTERMEDIATE_SIZE = 1536;

  private SyntheticMmproj() {}

  /** Writes a minimal synthetic mmproj file to the given directory. */
  static Path write(
      Path directory,
      int imageSize,
      int patchSize,
      int hiddenSize,
      int numLayers,
      int numHeads,
      int intermediateSize)
      throws Exception {
    Random random = new Random(42);
    int patchPixels = 3 * patchSize * patchSize;
    int numPositions = (imageSize / patchSize) * (imageSize / patchSize) + 1;

    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "clip")
            .addString("general.type", "mmproj")
            .addInt32("clip.vision.embedding_length", hiddenSize)
            .addInt32("clip.vision.block_count", numLayers)
            .addInt32("clip.vision.attention.head_count", numHeads)
            .addInt32("clip.vision.patch_size", patchSize)
            .addInt32("clip.vision.image_size", imageSize)
            .addInt32("clip.vision.feed_forward_length", intermediateSize)
            .addInt32("clip.vision.num_position_embeddings", numPositions)
            .addFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f);

    builder
        .addTensor(
            "v.patch_embd.weight",
            GgufTensorType.F32,
            new long[] {patchPixels, hiddenSize},
            randomFloats(random, hiddenSize * patchPixels))
        .addTensor(
            "v.patch_embd.bias", GgufTensorType.F32, new long[] {hiddenSize}, zeros(hiddenSize));
    builder.addTensor(
        "v.position_embd.weight",
        GgufTensorType.F32,
        new long[] {hiddenSize, numPositions},
        randomFloats(random, numPositions * hiddenSize));
    builder
        .addTensor("v.pre_ln.weight", GgufTensorType.F32, new long[] {hiddenSize}, ones(hiddenSize))
        .addTensor("v.pre_ln.bias", GgufTensorType.F32, new long[] {hiddenSize}, zeros(hiddenSize))
        .addTensor(
            "v.class_embd.weight",
            GgufTensorType.F32,
            new long[] {hiddenSize},
            randomFloats(random, hiddenSize))
        .addTensor(
            "v.post_ln.weight", GgufTensorType.F32, new long[] {hiddenSize}, ones(hiddenSize))
        .addTensor(
            "v.post_ln.bias", GgufTensorType.F32, new long[] {hiddenSize}, zeros(hiddenSize));

    for (int layer = 0; layer < numLayers; layer++) {
      String p = "v.blk." + layer + ".";
      builder
          .addTensor(
              p + "ln1.weight", GgufTensorType.F32, new long[] {hiddenSize}, ones(hiddenSize))
          .addTensor(p + "ln1.bias", GgufTensorType.F32, new long[] {hiddenSize}, zeros(hiddenSize))
          .addTensor(
              p + "attn_qkv.weight",
              GgufTensorType.F32,
              new long[] {hiddenSize, hiddenSize * 3},
              randomFloats(random, hiddenSize * 3 * hiddenSize))
          .addTensor(
              p + "attn_qkv.bias",
              GgufTensorType.F32,
              new long[] {hiddenSize * 3},
              zeros(hiddenSize * 3))
          .addTensor(
              p + "attn_out.weight",
              GgufTensorType.F32,
              new long[] {hiddenSize, hiddenSize},
              randomFloats(random, hiddenSize * hiddenSize))
          .addTensor(
              p + "attn_out.bias", GgufTensorType.F32, new long[] {hiddenSize}, zeros(hiddenSize))
          .addTensor(
              p + "ln2.weight", GgufTensorType.F32, new long[] {hiddenSize}, ones(hiddenSize))
          .addTensor(p + "ln2.bias", GgufTensorType.F32, new long[] {hiddenSize}, zeros(hiddenSize))
          .addTensor(
              p + "ffn_up.weight",
              GgufTensorType.F32,
              new long[] {hiddenSize, intermediateSize},
              randomFloats(random, intermediateSize * hiddenSize))
          .addTensor(
              p + "ffn_up.bias",
              GgufTensorType.F32,
              new long[] {intermediateSize},
              zeros(intermediateSize))
          .addTensor(
              p + "ffn_down.weight",
              GgufTensorType.F32,
              new long[] {intermediateSize, hiddenSize},
              randomFloats(random, hiddenSize * intermediateSize))
          .addTensor(
              p + "ffn_down.bias", GgufTensorType.F32, new long[] {hiddenSize}, zeros(hiddenSize));
    }

    Path out = directory.resolve("mmproj.gguf");
    Files.write(out, builder.build());
    return out;
  }

  /** Writes a synthetic mmproj with default small parameters. */
  static Path writeDefault(Path directory) throws Exception {
    return write(
        directory,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_PATCH_SIZE,
        DEFAULT_HIDDEN_SIZE,
        DEFAULT_NUM_LAYERS,
        DEFAULT_NUM_HEADS,
        DEFAULT_INTERMEDIATE_SIZE);
  }

  private static byte[] randomFloats(Random random, int count) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] = (random.nextFloat() - 0.5f) * 0.2f;
    }
    ByteBuffer buf =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : values) {
      buf.putFloat(v);
    }
    return buf.array();
  }

  private static byte[] ones(int count) {
    float[] values = new float[count];
    java.util.Arrays.fill(values, 1.0f);
    ByteBuffer buf =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : values) {
      buf.putFloat(v);
    }
    return buf.array();
  }

  private static byte[] zeros(int count) {
    return new byte[count * Float.BYTES];
  }
}
