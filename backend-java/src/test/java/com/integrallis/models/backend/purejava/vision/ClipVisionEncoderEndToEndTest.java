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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Synthetic encoder shape checks; real image-to-answer evidence is recorded by the vision harness.
 */
@Tag("unit")
class ClipVisionEncoderEndToEndTest {

  private static final int IMAGE_SIZE = 16;
  private static final int PATCH_SIZE = 4;
  private static final int HIDDEN_SIZE = 32;
  private static final int NUM_LAYERS = 2;
  private static final int NUM_HEADS = 4;
  private static final int INTERMEDIATE_SIZE = 64;
  private static final int NUM_POSITIONS = 17;

  @Test
  void endToEndVisionEncoderProducesEmbeddingsForTextDecoder(@TempDir Path directory)
      throws Exception {
    Path mmproj = writeSyntheticMmproj(directory);

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(mmproj, arena);
      var config = ClipVisionConfig.fromGguf(file);
      var weights = ClipVisionWeights.fromGgufFile(file, config);
      var encoder = new ClipVisionEncoder(config, weights, null /* no kernel */);

      // Preprocess a synthetic image (16x16 red, 3 channels)
      float[] image = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
      for (int i = 0; i < image.length; i++) {
        image[i] = (i % 3 == 0) ? 1.0f : 0.0f; // Red channel = 1.0
      }

      float[] embeddings = encoder.encode(image);

      int expectedTokens = config.numTokens(); // includes CLS token
      int expectedDim = HIDDEN_SIZE;
      assertThat(embeddings).hasSize(expectedTokens * expectedDim);
      assertThat(allFinite(embeddings)).isTrue();
    }
  }

  @Test
  void configLoadedFromGgufHasSensibleDefaults(@TempDir Path directory) throws Exception {
    Path mmproj = writeSyntheticMmproj(directory);

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(mmproj, arena);
      var config = ClipVisionConfig.fromGguf(file);

      assertThat(config.hiddenSize()).isEqualTo(HIDDEN_SIZE);
      assertThat(config.numLayers()).isEqualTo(NUM_LAYERS);
      assertThat(config.numHeads()).isEqualTo(NUM_HEADS);
      assertThat(config.patchSize()).isEqualTo(PATCH_SIZE);
      assertThat(config.imageSize()).isEqualTo(IMAGE_SIZE);
      assertThat(config.numTokens())
          .isEqualTo((IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE) + 1); // +CLS
    }
  }

  private static Path writeSyntheticMmproj(Path directory) throws Exception {
    Random random = new Random(42);
    int patchPixels = 3 * PATCH_SIZE * PATCH_SIZE;
    int numPositions = NUM_POSITIONS;

    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "clip")
            .addString("general.type", "mmproj")
            .addString("general.name", "Synthetic CLIP mmproj")
            .addInt32("clip.vision.embedding_length", HIDDEN_SIZE)
            .addInt32("clip.vision.block_count", NUM_LAYERS)
            .addInt32("clip.vision.attention.head_count", NUM_HEADS)
            .addInt32("clip.vision.patch_size", PATCH_SIZE)
            .addInt32("clip.vision.image_size", IMAGE_SIZE)
            .addInt32("clip.vision.feed_forward_length", INTERMEDIATE_SIZE)
            .addInt32("clip.vision.num_position_embeddings", NUM_POSITIONS)
            .addFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f);

    // Patch embedding
    builder
        .addTensor(
            "v.patch_embd.weight",
            GgufTensorType.F32,
            new long[] {patchPixels, HIDDEN_SIZE},
            randomFloats(random, HIDDEN_SIZE * patchPixels))
        .addTensor(
            "v.patch_embd.bias", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, zeros(HIDDEN_SIZE));

    // Position embedding
    builder.addTensor(
        "v.position_embd.weight",
        GgufTensorType.F32,
        new long[] {HIDDEN_SIZE, NUM_POSITIONS},
        randomFloats(random, NUM_POSITIONS * HIDDEN_SIZE));

    // Pre-transformer norm
    builder
        .addTensor(
            "v.pre_ln.weight", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, ones(HIDDEN_SIZE))
        .addTensor(
            "v.pre_ln.bias", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, zeros(HIDDEN_SIZE))
        .addTensor(
            "v.class_embd.weight",
            GgufTensorType.F32,
            new long[] {HIDDEN_SIZE},
            randomFloats(random, HIDDEN_SIZE))
        .addTensor(
            "v.post_ln.weight", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, ones(HIDDEN_SIZE))
        .addTensor(
            "v.post_ln.bias", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, zeros(HIDDEN_SIZE));

    // Transformer blocks
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
      String p = "v.blk." + layer + ".";
      builder
          .addTensor(
              p + "ln1.weight", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, ones(HIDDEN_SIZE))
          .addTensor(
              p + "ln1.bias", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, zeros(HIDDEN_SIZE))
          .addTensor(
              p + "attn_qkv.weight",
              GgufTensorType.F32,
              new long[] {HIDDEN_SIZE, HIDDEN_SIZE * 3},
              randomFloats(random, HIDDEN_SIZE * 3 * HIDDEN_SIZE))
          .addTensor(
              p + "attn_qkv.bias",
              GgufTensorType.F32,
              new long[] {HIDDEN_SIZE * 3},
              zeros(HIDDEN_SIZE * 3))
          .addTensor(
              p + "attn_out.weight",
              GgufTensorType.F32,
              new long[] {HIDDEN_SIZE, HIDDEN_SIZE},
              randomFloats(random, HIDDEN_SIZE * HIDDEN_SIZE))
          .addTensor(
              p + "attn_out.bias", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, zeros(HIDDEN_SIZE))
          .addTensor(
              p + "ln2.weight", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, ones(HIDDEN_SIZE))
          .addTensor(
              p + "ln2.bias", GgufTensorType.F32, new long[] {HIDDEN_SIZE}, zeros(HIDDEN_SIZE))
          .addTensor(
              p + "ffn_up.weight",
              GgufTensorType.F32,
              new long[] {HIDDEN_SIZE, INTERMEDIATE_SIZE},
              randomFloats(random, INTERMEDIATE_SIZE * HIDDEN_SIZE))
          .addTensor(
              p + "ffn_up.bias",
              GgufTensorType.F32,
              new long[] {INTERMEDIATE_SIZE},
              zeros(INTERMEDIATE_SIZE))
          .addTensor(
              p + "ffn_down.weight",
              GgufTensorType.F32,
              new long[] {INTERMEDIATE_SIZE, HIDDEN_SIZE},
              randomFloats(random, HIDDEN_SIZE * INTERMEDIATE_SIZE))
          .addTensor(
              p + "ffn_down.bias",
              GgufTensorType.F32,
              new long[] {HIDDEN_SIZE},
              zeros(HIDDEN_SIZE));
    }

    Path out = directory.resolve("synthetic-mmproj.gguf");
    Files.write(out, builder.build());
    return out;
  }

  private static byte[] randomFloats(Random random, int count) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] = (random.nextFloat() - 0.5f) * 0.2f;
    }
    return floats(values);
  }

  private static byte[] ones(int count) {
    float[] values = new float[count];
    java.util.Arrays.fill(values, 1.0f);
    return floats(values);
  }

  private static byte[] zeros(int count) {
    float[] values = new float[count];
    return floats(values);
  }

  private static byte[] floats(float... values) {
    ByteBuffer buf =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : values) buf.putFloat(v);
    return buf.array();
  }

  private static boolean allFinite(float[] values) {
    for (float v : values) {
      if (!Float.isFinite(v)) return false;
    }
    return true;
  }
}
