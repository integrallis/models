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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.Objects;

/** Configuration for a CLIP-style vision encoder as stored in an mmproj GGUF file. */
public final class ClipVisionConfig {

  private final int hiddenSize;
  private final int numLayers;
  private final int numHeads;
  private final int patchSize;
  private final int imageSize;
  private final int intermediateSize;
  private final int numPositions;
  private final float layerNormEps;
  private boolean quickGelu;

  public ClipVisionConfig(
      int hiddenSize,
      int numLayers,
      int numHeads,
      int patchSize,
      int imageSize,
      int intermediateSize,
      int numPositions,
      float layerNormEps) {
    if (hiddenSize <= 0
        || numLayers <= 0
        || numHeads <= 0
        || hiddenSize % numHeads != 0
        || patchSize <= 0
        || imageSize <= 0
        || imageSize % patchSize != 0
        || intermediateSize <= 0
        || !Float.isFinite(layerNormEps)
        || layerNormEps <= 0) {
      throw new IllegalArgumentException("Invalid CLIP dimensions or normalization epsilon");
    }
    int patches = Math.multiplyExact(imageSize / patchSize, imageSize / patchSize);
    if (numPositions != Math.addExact(patches, 1)) {
      throw new IllegalArgumentException("CLIP position count must equal patch count plus CLS");
    }
    if ((long) imageSize * imageSize > Integer.MAX_VALUE / 3) {
      throw new IllegalArgumentException("Image is too large for a CHW array");
    }
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.numHeads = numHeads;
    this.patchSize = patchSize;
    this.imageSize = imageSize;
    this.intermediateSize = intermediateSize;
    this.numPositions = numPositions;
    this.layerNormEps = layerNormEps;
  }

  public int hiddenSize() {
    return hiddenSize;
  }

  public int numLayers() {
    return numLayers;
  }

  public int numHeads() {
    return numHeads;
  }

  public int patchSize() {
    return patchSize;
  }

  public int imageSize() {
    return imageSize;
  }

  public int intermediateSize() {
    return intermediateSize;
  }

  public int numPositions() {
    return numPositions;
  }

  public float layerNormEps() {
    return layerNormEps;
  }

  public boolean quickGelu() {
    return quickGelu;
  }

  public int headDim() {
    return hiddenSize / numHeads;
  }

  /**
   * Returns the number of vision tokens including the class token.
   *
   * <p>CLIP ViT prepends a learnable class token to the patch tokens, so the total is {@code
   * patches + 1}. The position embedding table has the same count.
   */
  public int numTokens() {
    int patches = (imageSize / patchSize) * (imageSize / patchSize);
    return patches + 1; // +1 for CLS token
  }

  private static int integer(GgufMetadata metadata, String key) {
    return metadata
        .getUint32(key)
        .or(() -> metadata.getInt32(key))
        .orElseThrow(() -> new IllegalArgumentException("Missing integer metadata: " + key));
  }

  /** Loads config from mmproj GGUF metadata. */
  public static ClipVisionConfig fromGguf(GgufFile file) {
    Objects.requireNonNull(file, "file");
    GgufMetadata meta = file.metadata();

    int hiddenSize = integer(meta, "clip.vision.embedding_length");
    int numLayers = integer(meta, "clip.vision.block_count");
    int numHeads = integer(meta, "clip.vision.attention.head_count");
    int patchSize = integer(meta, "clip.vision.patch_size");
    int imageSize = integer(meta, "clip.vision.image_size");
    int intermediateSize = integer(meta, "clip.vision.feed_forward_length");
    if (patchSize <= 0 || imageSize <= 0 || imageSize % patchSize != 0) {
      throw new IllegalArgumentException("Image size must be a positive multiple of patch size");
    }
    int expectedTokens =
        Math.addExact(Math.multiplyExact(imageSize / patchSize, imageSize / patchSize), 1);
    int numPositions =
        meta.getUint32("clip.vision.num_position_embeddings")
            .or(() -> meta.getInt32("clip.vision.num_position_embeddings"))
            .orElse(expectedTokens);
    if (numPositions != expectedTokens) {
      throw new IllegalArgumentException(
          "numPosition_embeddings ("
              + numPositions
              + ") must equal expected tokens ("
              + expectedTokens
              + ")");
    }
    float layerNormEps = meta.getFloat32("clip.vision.attention.layer_norm_epsilon").orElse(1e-5f);

    ClipVisionConfig config =
        new ClipVisionConfig(
            hiddenSize,
            numLayers,
            numHeads,
            patchSize,
            imageSize,
            intermediateSize,
            numPositions,
            layerNormEps);
    config.quickGelu = !meta.getBool("clip.use_gelu").orElse(true);
    return config;
  }
}
