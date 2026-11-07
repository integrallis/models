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

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Objects;
import javax.imageio.ImageIO;

/** LLaVA 1.5 RGB square padding, antialiased bicubic resize, and CLIP normalization. */
public final class ClipImagePreprocessor {
  private ClipImagePreprocessor() {}

  public static float[] preprocess(byte[] encoded, int size, float[] mean, float[] std) {
    Objects.requireNonNull(encoded, "encoded");
    try {
      BufferedImage image = ImageIO.read(new ByteArrayInputStream(encoded));
      if (image == null) throw new IllegalArgumentException("Unsupported or corrupt image");
      return preprocess(image, size, mean, std);
    } catch (IOException e) {
      throw new IllegalArgumentException("Cannot decode image", e);
    }
  }

  public static float[] preprocess(BufferedImage image, int size, float[] mean, float[] std) {
    Objects.requireNonNull(image, "image");
    if (size <= 0 || mean.length != 3 || std.length != 3) {
      throw new IllegalArgumentException(
          "Expected positive image size and three normalization channels");
    }
    for (int c = 0; c < 3; c++) {
      if (!Float.isFinite(mean[c])
          || mean[c] < 0
          || mean[c] > 1
          || !Float.isFinite(std[c])
          || std[c] <= 0) {
        throw new IllegalArgumentException("Invalid image normalization");
      }
    }
    int side = Math.max(image.getWidth(), image.getHeight());
    int left = (side - image.getWidth()) / 2;
    int top = (side - image.getHeight()) / 2;
    int[] rgb = new int[Math.multiplyExact(Math.multiplyExact(side, side), 3)];
    for (int y = 0; y < side; y++) {
      for (int x = 0; x < side; x++) {
        boolean inside =
            x >= left && x < left + image.getWidth() && y >= top && y < top + image.getHeight();
        int pixel = inside ? image.getRGB(x - left, y - top) : 0;
        for (int c = 0; c < 3; c++) {
          rgb[(y * side + x) * 3 + c] =
              inside ? (pixel >> (16 - c * 8)) & 255 : (int) (mean[c] * 255);
        }
      }
    }
    // Pillow's RGB bicubic resampling rounds/clamps after each separable pass.
    int[] horizontal = new int[Math.multiplyExact(Math.multiplyExact(side, size), 3)];
    double scale = (double) side / size;
    double filterScale = Math.max(1.0, scale);
    for (int x = 0; x < size; x++) {
      double center = (x + 0.5) * scale;
      int first = Math.max(0, (int) (center - 2 * filterScale + 0.5));
      int last = Math.min(side, (int) (center + 2 * filterScale + 0.5));
      int[] weights = weights(first, last, center, filterScale);
      for (int y = 0; y < side; y++) {
        for (int c = 0; c < 3; c++) {
          long sum = 1 << 21;
          for (int k = first; k < last; k++)
            sum += (long) weights[k - first] * rgb[(y * side + k) * 3 + c];
          horizontal[(y * size + x) * 3 + c] = pixel(sum);
        }
      }
    }
    float[] out = new float[Math.multiplyExact(Math.multiplyExact(size, size), 3)];
    for (int y = 0; y < size; y++) {
      double center = (y + 0.5) * scale;
      int first = Math.max(0, (int) (center - 2 * filterScale + 0.5));
      int last = Math.min(side, (int) (center + 2 * filterScale + 0.5));
      int[] weights = weights(first, last, center, filterScale);
      for (int x = 0; x < size; x++) {
        for (int c = 0; c < 3; c++) {
          long sum = 1 << 21;
          for (int k = first; k < last; k++)
            sum += (long) weights[k - first] * horizontal[(k * size + x) * 3 + c];
          out[c * size * size + y * size + x] = (pixel(sum) / 255.0f - mean[c]) / std[c];
        }
      }
    }
    return out;
  }

  private static int pixel(long value) {
    return Math.clamp((int) (value >> 22), 0, 255);
  }

  private static int[] weights(int first, int last, double center, double scale) {
    double[] result = new double[last - first];
    double sum = 0;
    for (int i = first; i < last; i++) {
      double x = Math.abs((i - center + 0.5) / scale);
      double w =
          x < 1 ? ((1.5 * x - 2.5) * x) * x + 1 : x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : 0;
      result[i - first] = w;
      sum += w;
    }
    // Pillow quantizes normalized bicubic coefficients to 22 fractional bits.
    int[] fixed = new int[result.length];
    for (int i = 0; i < result.length; i++) {
      double coefficient = result[i] / sum * (1 << 22);
      fixed[i] = (int) (coefficient + (coefficient < 0 ? -0.5 : 0.5));
    }
    return fixed;
  }
}
