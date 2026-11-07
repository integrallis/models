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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class LlavaVisionProjectorTest {
  @Test
  void realTensorLayoutPreservesResidualsAndProjectsPatches(@TempDir Path dir) throws Exception {
    Path model = write(dir.resolve("canonical.gguf"), false, "mlp");
    float[] pixels = new float[48];
    Arrays.fill(pixels, 1);
    // Analytical oracle: pre-LN([1..8]) + 23 * 0.1, then exact GELU and +0.25.
    // Both attention and FFN weights are zero; the FFN bias tests the residual path.
    float[] expectedRow = {
      0.852596178f,
      1.321883385f,
      1.813163286f,
      2.292892170f,
      2.753366443f,
      3.200028707f,
      3.639908048f,
      4.077276062f
    };
    try (var projector = LlavaVisionProjector.load(model)) {
      assertThat(projector.tokenCount()).isEqualTo(4);
      assertThat(projector.embeddingDimension()).isEqualTo(8);
      float[] result = projector.encodePreprocessed(pixels);
      for (int row = 0; row < 4; row++) {
        assertThat(Arrays.copyOfRange(result, row * 8, row * 8 + 8))
            .containsExactly(expectedRow, within(0.00002f));
      }
      assertThat(projector.encodePreprocessed(pixels)).containsExactly(result);
      pixels[0] = Float.NaN;
      assertThatThrownBy(() -> projector.encodePreprocessed(pixels))
          .isInstanceOf(IllegalArgumentException.class);
    }
    try (var canonical = LlavaVisionProjector.load(model);
        var legacy = LlavaVisionProjector.load(write(dir.resolve("legacy.gguf"), true, "mlp"))) {
      Arrays.fill(pixels, 1);
      assertThat(legacy.encodePreprocessed(pixels))
          .containsExactly(canonical.encodePreprocessed(pixels));
    }
  }

  @Test
  void rejectsDifferentArchitecturesAndClosedUse(@TempDir Path dir) throws Exception {
    Path unsupported = write(dir.resolve("qwen.gguf"), false, "qwen3vl_merger");
    assertThatThrownBy(() -> LlavaVisionProjector.load(unsupported))
        .isInstanceOf(IllegalArgumentException.class);
    var projector = LlavaVisionProjector.load(write(dir.resolve("valid.gguf"), false, "mlp"));
    projector.close();
    projector.close();
    assertThatThrownBy(() -> projector.encodePreprocessed(new float[48]))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void imageNormalizationUsesRgbChwAndSquarePadding() throws Exception {
    BufferedImage image = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
    image.setRGB(0, 0, 0xff0000);
    image.setRGB(1, 0, 0x0000ff);
    float[] mean = {0.5f, 0.5f, 0.5f};
    float[] std = {0.5f, 0.5f, 0.5f};
    ByteArrayOutputStream encoded = new ByteArrayOutputStream();
    ImageIO.write(image, "png", encoded);
    float[] pixels = ClipImagePreprocessor.preprocess(encoded.toByteArray(), 2, mean, std);
    assertThat(Arrays.copyOf(pixels, 2)).containsExactly(1f, -1f);
    assertThat(Arrays.copyOfRange(pixels, 4, 6)).containsExactly(-1f, -1f);
    assertThat(Arrays.copyOfRange(pixels, 8, 10)).containsExactly(-1f, 1f);
    assertThat(pixels[2]).isCloseTo((127f / 255f - .5f) / .5f, within(1e-7f));
    assertThatThrownBy(() -> ClipImagePreprocessor.preprocess(new byte[] {1, 2}, 2, mean, std))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> ClipImagePreprocessor.preprocess(image, 2, mean, new float[] {1, 0, 1}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void bicubicResizeMatchesIndependentPillowRgbReference() {
    BufferedImage image = new BufferedImage(5, 3, BufferedImage.TYPE_INT_RGB);
    for (int y = 0; y < 3; y++) {
      for (int x = 0; x < 5; x++)
        image.setRGB(x, y, ((x * 53 + y * 7) % 256 << 16) | (y * 83 << 8) | x * 41);
    }
    // Pillow 12.0.0: RGB pad to 5x5 with black, then BICUBIC resize to 4x4.
    int[][] expected = {
      {1, 0, 1},
      {13, 0, 11},
      {26, 0, 20},
      {38, 0, 30},
      {10, 27, 6},
      {77, 27, 58},
      {147, 27, 111},
      {214, 27, 163},
      {19, 144, 6},
      {86, 144, 58},
      {157, 144, 111},
      {224, 144, 163},
      {4, 36, 1},
      {17, 36, 11},
      {29, 36, 20},
      {42, 36, 30}
    };
    float[] actual =
        ClipImagePreprocessor.preprocess(image, 4, new float[] {0, 0, 0}, new float[] {1, 1, 1});
    for (int pixel = 0; pixel < 16; pixel++) {
      for (int channel = 0; channel < 3; channel++)
        assertThat(actual[channel * 16 + pixel]).isEqualTo(expected[pixel][channel] / 255f);
    }
  }

  @Test
  void rejectsInvalidDimensions() {
    assertThatThrownBy(() -> new ClipVisionConfig(8, 2, 3, 2, 4, 16, 5, 1e-5f))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new ClipVisionConfig(8, 2, 2, 3, 4, 16, 5, 1e-5f))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new ClipVisionConfig(8, 2, 2, 2, 4, 16, 4, 1e-5f))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static Path write(Path output, boolean swapped, String projector) throws Exception {
    SyntheticGgufBuilder b =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "clip")
            .addString("clip.projector_type", projector)
            .addBool("clip.has_llava_projector", true)
            .addBool("clip.use_gelu", false)
            .addUint32("clip.vision.embedding_length", 8)
            .addUint32("clip.vision.block_count", 23)
            .addUint32("clip.vision.attention.head_count", 2)
            .addUint32("clip.vision.patch_size", 2)
            .addUint32("clip.vision.image_size", 4)
            .addUint32("clip.vision.feed_forward_length", 16)
            .addFloat32Array("clip.vision.image_mean", List.of(.5f, .5f, .5f))
            .addFloat32Array("clip.vision.image_std", List.of(.5f, .5f, .5f));
    float[] patch = new float[8 * 12];
    for (int i = 0; i < 8; i++) patch[i * 12] = i + 1;
    tensor(b, "v.patch_embd.weight", new long[] {2, 2, 3, 8}, patch);
    vector(b, "v.class_embd", 8, .3f);
    matrix(b, "v.position_embd.weight", 8, 5, false);
    vector(b, "v.pre_ln.weight", 8, 1);
    vector(b, "v.pre_ln.bias", 8, 0);
    for (int i = 0; i < 23; i++) {
      String prefix = "v.blk." + i + ".";
      for (String name : List.of("attn_q", "attn_k", "attn_v", "attn_out")) {
        matrix(b, prefix + name + ".weight", 8, 8, false);
        vector(b, prefix + name + ".bias", 8, 0);
      }
      for (String name : List.of("ln1", "ln2")) {
        vector(b, prefix + name + ".weight", 8, 1);
        vector(b, prefix + name + ".bias", 8, 0);
      }
      String up = prefix + (swapped ? "ffn_down" : "ffn_up");
      String down = prefix + (swapped ? "ffn_up" : "ffn_down");
      matrix(b, up + ".weight", 8, 16, false);
      vector(b, up + ".bias", 16, 0);
      matrix(b, down + ".weight", 16, 8, false);
      vector(b, down + ".bias", 8, .1f);
    }
    for (String name : List.of("mm.0", "mm.2")) {
      matrix(b, name + ".weight", 8, 8, true);
      vector(b, name + ".bias", 8, name.equals("mm.2") ? .25f : 0);
    }
    Files.write(output, b.build());
    return output;
  }

  private static void vector(SyntheticGgufBuilder b, String name, int size, float value) {
    float[] values = new float[size];
    Arrays.fill(values, value);
    tensor(b, name, new long[] {size}, values);
  }

  private static void matrix(
      SyntheticGgufBuilder b, String name, int columns, int rows, boolean identity) {
    float[] values = new float[columns * rows];
    if (identity) for (int i = 0; i < Math.min(columns, rows); i++) values[i * columns + i] = 1;
    tensor(b, name, new long[] {columns, rows}, values);
  }

  private static void tensor(SyntheticGgufBuilder b, String name, long[] shape, float[] values) {
    ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) bytes.putFloat(value);
    b.addTensor(name, GgufTensorType.F32, shape, bytes.array());
  }
}
