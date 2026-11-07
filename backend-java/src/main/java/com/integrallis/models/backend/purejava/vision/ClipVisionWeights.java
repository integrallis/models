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

import static com.integrallis.models.backend.purejava.gguf.GgufTensorValues.toFloatArray;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

final class ClipVisionWeights {

  record Matrix(MemorySegment data, GgufTensorType type, int rows, int columns) {
    Matrix {
      Objects.requireNonNull(data, "data");
      Objects.requireNonNull(type, "type");
    }
  }

  record Layer(
      float[] ln1Weight,
      float[] ln1Bias,
      Matrix attnQkv,
      float[] attnQkvBias,
      Matrix attnOut,
      float[] attnOutBias,
      float[] ln2Weight,
      float[] ln2Bias,
      Matrix ffnUp,
      float[] ffnUpBias,
      Matrix ffnDown,
      float[] ffnDownBias) {}

  private final Matrix patchEmbed;
  private final float[] patchEmbedBias;
  private final Matrix positionEmbed;
  private final float[] positionEmbedF32;
  private final float[] preLnWeight;
  private final float[] preLnBias;
  private final float[] classEmb;
  private final float[] postLnWeight;
  private final float[] postLnBias;
  private final Layer[] layers;

  private ClipVisionWeights(
      Matrix patchEmbed,
      float[] patchEmbedBias,
      Matrix positionEmbed,
      float[] positionEmbedF32,
      float[] preLnWeight,
      float[] preLnBias,
      float[] classEmb,
      float[] postLnWeight,
      float[] postLnBias,
      Layer[] layers) {
    this.patchEmbed = patchEmbed;
    this.patchEmbedBias = patchEmbedBias;
    this.positionEmbed = positionEmbed;
    this.positionEmbedF32 = positionEmbedF32;
    this.preLnWeight = preLnWeight;
    this.preLnBias = preLnBias;
    this.classEmb = classEmb;
    this.postLnWeight = postLnWeight;
    this.postLnBias = postLnBias;
    this.layers = layers;
  }

  Matrix patchEmbed() {
    return patchEmbed;
  }

  float[] patchEmbedBias() {
    return patchEmbedBias;
  }

  Matrix positionEmbed() {
    return positionEmbed;
  }

  float[] positionEmbedF32() {
    return positionEmbedF32;
  }

  float[] preLnWeight() {
    return preLnWeight;
  }

  float[] preLnBias() {
    return preLnBias;
  }

  float[] classEmb() {
    return classEmb;
  }

  float[] postLnWeight() {
    return postLnWeight;
  }

  float[] postLnBias() {
    return postLnBias;
  }

  Layer layer(int index) {
    return layers[index];
  }

  static ClipVisionWeights fromGgufFile(GgufFile file, ClipVisionConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");

    int dim = config.hiddenSize();
    int intermediateSize = config.intermediateSize();
    int numPositions = config.numPositions();

    int patchPixels = 3 * config.patchSize() * config.patchSize();
    GgufTensorData patch = file.getTensor("v.patch_embd.weight");
    long[] patchShape = patch.shape();
    if (!Arrays.equals(patchShape, new long[] {patchPixels, dim})
        && !Arrays.equals(
            patchShape, new long[] {config.patchSize(), config.patchSize(), 3, dim})) {
      throw new IllegalArgumentException(
          "Invalid CLIP patch convolution shape: " + Arrays.toString(patchShape));
    }
    Matrix patchEmbed = new Matrix(patch.dataSegment(), patch.type(), dim, patchPixels);
    float[] patchEmbedBias = vectorOrZero(file, "v.patch_embd.bias", dim);

    Matrix positionEmbed = matrix(file, "v.position_embd.weight", numPositions, dim);
    float[] positionEmbedF32 = toFloatArray(file.getTensor("v.position_embd.weight"));

    float[] preLnWeight = optionalVector(file, "v.pre_ln.weight", dim);
    float[] preLnBias = optionalVector(file, "v.pre_ln.bias", dim);
    float[] classEmb =
        vector(file, file.hasTensor("v.class_embd") ? "v.class_embd" : "v.class_embd.weight", dim);
    float[] postLnWeight = optionalVector(file, "v.post_ln.weight", dim);
    float[] postLnBias = vectorOrZero(file, "v.post_ln.bias", dim);

    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String p = "v.blk." + layer + ".";
      Matrix qkv;
      float[] qkvValues;
      float[] qkvBias;
      if (file.hasTensor(p + "attn_qkv.weight")) {
        qkv = matrix(file, p + "attn_qkv.weight", dim * 3, dim);
        qkvValues = toFloatArray(file.getTensor(p + "attn_qkv.weight"));
        qkvBias = vectorOrZero(file, p + "attn_qkv.bias", dim * 3);
      } else {
        qkvValues = new float[Math.multiplyExact(3, Math.multiplyExact(dim, dim))];
        qkvBias = new float[3 * dim];
        String[] names = {"attn_q", "attn_k", "attn_v"};
        for (int i = 0; i < names.length; i++) {
          matrix(file, p + names[i] + ".weight", dim, dim);
          System.arraycopy(
              toFloatArray(file.getTensor(p + names[i] + ".weight")),
              0,
              qkvValues,
              i * dim * dim,
              dim * dim);
          System.arraycopy(
              vectorOrZero(file, p + names[i] + ".bias", dim), 0, qkvBias, i * dim, dim);
        }
        qkv = new Matrix(MemorySegment.ofArray(qkvValues), GgufTensorType.F32, 3 * dim, dim);
      }
      // Early LLaVA converters reversed the FFN tensor names. Shapes identify that layout.
      boolean swapped =
          dim != intermediateSize
              && Arrays.equals(
                  file.getTensor(p + "ffn_down.weight").shape(),
                  new long[] {dim, intermediateSize});
      String up = p + (swapped ? "ffn_down" : "ffn_up");
      String down = p + (swapped ? "ffn_up" : "ffn_down");
      layers[layer] =
          new Layer(
              vector(file, p + "ln1.weight", dim),
              vector(file, p + "ln1.bias", dim),
              qkv,
              qkvBias,
              matrix(file, p + "attn_out.weight", dim, dim),
              vector(file, p + "attn_out.bias", dim),
              vector(file, p + "ln2.weight", dim),
              vector(file, p + "ln2.bias", dim),
              matrix(file, up + ".weight", intermediateSize, dim),
              vector(file, up + ".bias", intermediateSize),
              matrix(file, down + ".weight", dim, intermediateSize),
              vector(file, down + ".bias", dim));
    }

    return new ClipVisionWeights(
        patchEmbed,
        patchEmbedBias,
        positionEmbed,
        positionEmbedF32,
        preLnWeight,
        preLnBias,
        classEmb,
        postLnWeight,
        postLnBias,
        layers);
  }

  static Matrix matrix(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = file.getTensor(name);
    long[] expected = {columns, rows};
    if (!Arrays.equals(tensor.shape(), expected)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expected)
              + ": "
              + Arrays.toString(tensor.shape()));
    }
    return new Matrix(tensor.dataSegment(), tensor.type(), rows, columns);
  }

  private static float[] vectorOrZero(GgufFile file, String name, int size) {
    float[] values = optionalVector(file, name, size);
    return values == null ? new float[size] : values;
  }

  private static float[] optionalVector(GgufFile file, String name, int size) {
    if (!file.hasTensor(name)) return null;
    GgufTensorData tensor = file.getTensor(name);
    long[] expected = {size};
    if (!Arrays.equals(tensor.shape(), expected)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expected)
              + ": "
              + Arrays.toString(tensor.shape()));
    }
    return toFloatArray(tensor);
  }

  static float[] vector(GgufFile file, String name, int size) {
    GgufTensorData tensor = file.getTensor(name);
    long[] expected = {size};
    if (!Arrays.equals(tensor.shape(), expected)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expected)
              + ": "
              + Arrays.toString(tensor.shape()));
    }
    return toFloatArray(tensor);
  }
}
