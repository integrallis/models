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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 * A complete synthetic LLaVA 1.5 projector (23-block CLIP feature extractor plus the two-layer MLP)
 * whose output width matches the nano text model, with seeded random weights so different images
 * project to different rows. Four image rows: a 4x4 image in 2x2 patches.
 */
final class SyntheticLlava {
  static final int IMAGE_ROWS = 4;
  private static final int HIDDEN = 8;
  private static final int FFN = 16;

  private SyntheticLlava() {}

  /** Writes a projector whose output rows are {@code outputDim} wide. */
  static Path writeProjector(Path output, int outputDim, long seed) throws Exception {
    Random random = new Random(seed);
    SyntheticGgufBuilder b =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "clip")
            .addString("clip.projector_type", "mlp")
            .addBool("clip.has_llava_projector", true)
            .addBool("clip.use_gelu", false)
            .addUint32("clip.vision.embedding_length", HIDDEN)
            .addUint32("clip.vision.block_count", 23)
            .addUint32("clip.vision.attention.head_count", 2)
            .addUint32("clip.vision.patch_size", 2)
            .addUint32("clip.vision.image_size", 4)
            .addUint32("clip.vision.feed_forward_length", FFN)
            .addFloat32Array("clip.vision.image_mean", List.of(.5f, .5f, .5f))
            .addFloat32Array("clip.vision.image_std", List.of(.5f, .5f, .5f));
    tensor(
        b, "v.patch_embd.weight", new long[] {2, 2, 3, HIDDEN}, random(random, 12 * HIDDEN, .5f));
    tensor(b, "v.class_embd", new long[] {HIDDEN}, random(random, HIDDEN, .1f));
    tensor(b, "v.position_embd.weight", new long[] {HIDDEN, 5}, random(random, HIDDEN * 5, .1f));
    tensor(b, "v.pre_ln.weight", new long[] {HIDDEN}, fill(HIDDEN, 1));
    tensor(b, "v.pre_ln.bias", new long[] {HIDDEN}, fill(HIDDEN, 0));
    for (int i = 0; i < 23; i++) {
      String p = "v.blk." + i + ".";
      for (String name : List.of("attn_q", "attn_k", "attn_v", "attn_out")) {
        tensor(
            b,
            p + name + ".weight",
            new long[] {HIDDEN, HIDDEN},
            random(random, HIDDEN * HIDDEN, .05f));
        tensor(b, p + name + ".bias", new long[] {HIDDEN}, fill(HIDDEN, 0));
      }
      for (String name : List.of("ln1", "ln2")) {
        tensor(b, p + name + ".weight", new long[] {HIDDEN}, fill(HIDDEN, 1));
        tensor(b, p + name + ".bias", new long[] {HIDDEN}, fill(HIDDEN, 0));
      }
      tensor(b, p + "ffn_up.weight", new long[] {HIDDEN, FFN}, random(random, HIDDEN * FFN, .05f));
      tensor(b, p + "ffn_up.bias", new long[] {FFN}, fill(FFN, 0));
      tensor(
          b, p + "ffn_down.weight", new long[] {FFN, HIDDEN}, random(random, FFN * HIDDEN, .05f));
      tensor(b, p + "ffn_down.bias", new long[] {HIDDEN}, fill(HIDDEN, 0));
    }
    tensor(b, "mm.0.weight", new long[] {HIDDEN, HIDDEN}, random(random, HIDDEN * HIDDEN, .5f));
    tensor(b, "mm.0.bias", new long[] {HIDDEN}, fill(HIDDEN, 0));
    tensor(
        b, "mm.2.weight", new long[] {HIDDEN, outputDim}, random(random, HIDDEN * outputDim, .5f));
    tensor(b, "mm.2.bias", new long[] {outputDim}, fill(outputDim, 0));
    Files.write(output, b.build());
    return output;
  }

  /** A 4x4 PNG whose pixels are a pure function of {@code seed}. */
  static byte[] image(int seed) throws Exception {
    Random random = new Random(seed);
    BufferedImage image = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
    for (int y = 0; y < 4; y++) {
      for (int x = 0; x < 4; x++) image.setRGB(x, y, random.nextInt(0x1000000));
    }
    var out = new ByteArrayOutputStream();
    ImageIO.write(image, "png", out);
    return out.toByteArray();
  }

  private static float[] random(Random random, int count, float scale) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) values[i] = (random.nextFloat() - .5f) * 2 * scale;
    return values;
  }

  private static float[] fill(int count, float value) {
    float[] values = new float[count];
    java.util.Arrays.fill(values, value);
    return values;
  }

  private static void tensor(SyntheticGgufBuilder b, String name, long[] shape, float[] values) {
    ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) bytes.putFloat(value);
    b.addTensor(name, GgufTensorType.F32, shape, bytes.array());
  }
}
