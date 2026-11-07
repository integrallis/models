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
import com.integrallis.models.backend.purejava.quant.Q8_0Quantizer;
import java.lang.foreign.Arena;
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
      float[] attnQkvF32,
      MemorySegment attnQkvQ8_0,
      float[] attnQkvBias,
      Matrix attnOut,
      float[] attnOutF32,
      MemorySegment attnOutQ8_0,
      float[] attnOutBias,
      float[] ln2Weight,
      float[] ln2Bias,
      Matrix ffnUp,
      float[] ffnUpF32,
      MemorySegment ffnUpQ8_0,
      float[] ffnUpBias,
      Matrix ffnDown,
      float[] ffnDownF32,
      MemorySegment ffnDownQ8_0,
      float[] ffnDownBias) {}

  private final Matrix patchEmbed;
  private final float[] patchEmbedF32;
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
      float[] patchEmbedF32,
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
    this.patchEmbedF32 = patchEmbedF32;
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

  float[] patchEmbedF32() {
    return patchEmbedF32;
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

  /**
   * Returns a new weights instance with Q8_0 quantized projection matrices. The Q8_0 segments are
   * allocated in the provided arena and can be passed to native kernels via {@code
   * GgufBatchedMatrixKernel}.
   *
   * @param arena arena for allocating native Q8_0 segments
   * @return new weights with Q8_0 caches populated
   */
  ClipVisionWeights withQ8_0(Arena arena) {
    Layer[] q8Layers = new Layer[layers.length];
    for (int i = 0; i < layers.length; i++) {
      Layer l = layers[i];
      q8Layers[i] =
          new Layer(
              l.ln1Weight,
              l.ln1Bias,
              l.attnQkv,
              l.attnQkvF32,
              Q8_0Quantizer.quantize(l.attnQkvF32, l.attnQkv.rows(), l.attnQkv.columns(), arena),
              l.attnQkvBias,
              l.attnOut,
              l.attnOutF32,
              Q8_0Quantizer.quantize(l.attnOutF32, l.attnOut.rows(), l.attnOut.columns(), arena),
              l.attnOutBias,
              l.ln2Weight,
              l.ln2Bias,
              l.ffnUp,
              l.ffnUpF32,
              Q8_0Quantizer.quantize(l.ffnUpF32, l.ffnUp.rows(), l.ffnUp.columns(), arena),
              l.ffnUpBias,
              l.ffnDown,
              l.ffnDownF32,
              Q8_0Quantizer.quantize(l.ffnDownF32, l.ffnDown.rows(), l.ffnDown.columns(), arena),
              l.ffnDownBias);
    }
    return new ClipVisionWeights(
        patchEmbed,
        patchEmbedF32,
        patchEmbedBias,
        positionEmbed,
        positionEmbedF32,
        preLnWeight,
        preLnBias,
        classEmb,
        postLnWeight,
        postLnBias,
        q8Layers);
  }

  static ClipVisionWeights fromGgufFile(GgufFile file, ClipVisionConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");

    int dim = config.hiddenSize();
    int intermediateSize = config.intermediateSize();
    int numPositions = config.numPositions();

    int patchPixels = 3 * config.patchSize() * config.patchSize();
    Matrix patchEmbed = matrix(file, "v.patch_embd.weight", dim, patchPixels);
    float[] patchEmbedF32 = toFloatArray(file.getTensor("v.patch_embd.weight"));
    float[] patchEmbedBias = vector(file, "v.patch_embd.bias", dim);

    Matrix positionEmbed = matrix(file, "v.position_embd.weight", numPositions, dim);
    float[] positionEmbedF32 = toFloatArray(file.getTensor("v.position_embd.weight"));

    float[] preLnWeight = optionalVector(file, "v.pre_ln.weight", dim);
    float[] preLnBias = optionalVector(file, "v.pre_ln.bias", dim);
    float[] classEmb = optionalVector(file, "v.class_embd.weight", dim);
    float[] postLnWeight = vector(file, "v.post_ln.weight", dim);
    float[] postLnBias = vector(file, "v.post_ln.bias", dim);

    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String p = "v.blk." + layer + ".";
      layers[layer] =
          new Layer(
              vector(file, p + "ln1.weight", dim),
              vector(file, p + "ln1.bias", dim),
              matrix(file, p + "attn_qkv.weight", dim * 3, dim),
              toFloatArray(file.getTensor(p + "attn_qkv.weight")),
              null, // Q8_0 populated later via withQ8_0()
              vector(file, p + "attn_qkv.bias", dim * 3),
              matrix(file, p + "attn_out.weight", dim, dim),
              toFloatArray(file.getTensor(p + "attn_out.weight")),
              null,
              vector(file, p + "attn_out.bias", dim),
              vector(file, p + "ln2.weight", dim),
              vector(file, p + "ln2.bias", dim),
              matrix(file, p + "ffn_up.weight", intermediateSize, dim),
              toFloatArray(file.getTensor(p + "ffn_up.weight")),
              null,
              vector(file, p + "ffn_up.bias", intermediateSize),
              matrix(file, p + "ffn_down.weight", dim, intermediateSize),
              toFloatArray(file.getTensor(p + "ffn_down.weight")),
              null,
              vector(file, p + "ffn_down.bias", dim));
    }

    return new ClipVisionWeights(
        patchEmbed,
        patchEmbedF32,
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

  private static Matrix matrix(GgufFile file, String name, int rows, int columns) {
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

  private static float[] optionalVector(GgufFile file, String name, int size) {
    GgufTensorData tensor = file.getTensor(name);
    if (tensor == null) {
      return null;
    }
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

  private static float[] vector(GgufFile file, String name, int size) {
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
