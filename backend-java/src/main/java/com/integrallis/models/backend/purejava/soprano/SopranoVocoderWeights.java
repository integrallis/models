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
package com.integrallis.models.backend.purejava.soprano;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Mapped projection tensors and decoded small vectors for Soprano's ConvNeXt vocoder. */
final class SopranoVocoderWeights {

  record Matrix(MemorySegment data, GgufTensorType type, int rows, int columns) {
    Matrix {
      Objects.requireNonNull(data, "data");
      Objects.requireNonNull(type, "type");
      if (rows <= 0 || columns <= 0) {
        throw new IllegalArgumentException("matrix dimensions must be positive");
      }
    }
  }

  record Block(
      float[] depthwiseWeight,
      float[] depthwiseBias,
      float[] normWeight,
      float[] normBias,
      Matrix pointwiseUp,
      float[] pointwiseUpBias,
      Matrix pointwiseDown,
      float[] pointwiseDownBias,
      float[] gamma) {}

  private final float[] embedding;
  private final float[] embeddingBias;
  private final float[] inputNormWeight;
  private final float[] inputNormBias;
  private final Block[] layers;
  private final float[] finalNormWeight;
  private final float[] finalNormBias;
  private final Matrix head;
  private final float[] headBias;
  private final float[] window;

  private SopranoVocoderWeights(
      float[] embedding,
      float[] embeddingBias,
      float[] inputNormWeight,
      float[] inputNormBias,
      Block[] layers,
      float[] finalNormWeight,
      float[] finalNormBias,
      Matrix head,
      float[] headBias,
      float[] window) {
    this.embedding = embedding;
    this.embeddingBias = embeddingBias;
    this.inputNormWeight = inputNormWeight;
    this.inputNormBias = inputNormBias;
    this.layers = layers;
    this.finalNormWeight = finalNormWeight;
    this.finalNormBias = finalNormBias;
    this.head = head;
    this.headBias = headBias;
    this.window = window;
  }

  static SopranoVocoderWeights load(GgufFile file, SopranoConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    int inputDim = config.hiddenSize();
    int dim = config.decoderDim();
    int intermediate = config.decoderIntermediateSize();

    Block[] layers = new Block[config.decoderLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "decoder.convnext." + layer;
      layers[layer] =
          new Block(
              values(file, prefix + ".dwconv.weight", config.decoderKernelSize(), 1, dim),
              vector(file, prefix + ".dwconv.bias", dim),
              vector(file, prefix + ".norm.weight", dim),
              vector(file, prefix + ".norm.bias", dim),
              projection(file, prefix + ".pwconv1.weight", intermediate, dim),
              vector(file, prefix + ".pwconv1.bias", intermediate),
              projection(file, prefix + ".pwconv2.weight", dim, intermediate),
              vector(file, prefix + ".pwconv2.bias", dim),
              vector(file, prefix + ".gamma", dim));
    }

    return new SopranoVocoderWeights(
        values(file, "decoder.embed.weight", 1, inputDim, dim),
        vector(file, "decoder.embed.bias", dim),
        vector(file, "decoder.norm.weight", dim),
        vector(file, "decoder.norm.bias", dim),
        layers,
        vector(file, "decoder.final_layer_norm.weight", dim),
        vector(file, "decoder.final_layer_norm.bias", dim),
        projection(file, "decoder.head.out.weight", config.fftSize() + 2, dim),
        vector(file, "decoder.head.out.bias", config.fftSize() + 2),
        vector(file, "decoder.head.istft.window", config.fftSize()));
  }

  float[] embedding() {
    return embedding;
  }

  float[] embeddingBias() {
    return embeddingBias;
  }

  float[] inputNormWeight() {
    return inputNormWeight;
  }

  float[] inputNormBias() {
    return inputNormBias;
  }

  int layerCount() {
    return layers.length;
  }

  Block layer(int index) {
    return layers[index];
  }

  float[] finalNormWeight() {
    return finalNormWeight;
  }

  float[] finalNormBias() {
    return finalNormBias;
  }

  Matrix head() {
    return head;
  }

  float[] headBias() {
    return headBias;
  }

  float[] window() {
    return window;
  }

  private static Matrix projection(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, columns, rows);
    if (!TensorOps.supportsBatchedMatmul(tensor.type())) {
      throw new IllegalArgumentException(
          name + " uses unsupported batched projection type " + tensor.type());
    }
    return new Matrix(tensor.dataSegment(), tensor.type(), rows, columns);
  }

  private static float[] vector(GgufFile file, String name, int size) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, size);
    return GgufTensorValues.toFloatArray(tensor);
  }

  private static float[] values(GgufFile file, String name, long... shape) {
    GgufTensorData tensor = file.getTensor(name);
    requireShape(tensor, shape);
    return GgufTensorValues.toFloatArray(tensor);
  }

  private static void requireShape(GgufTensorData tensor, long... expected) {
    long[] actual = tensor.shape();
    if (actual.length < expected.length) {
      throw shapeMismatch(tensor.name(), expected, actual);
    }
    for (int index = 0; index < expected.length; index++) {
      if (actual[index] != expected[index]) {
        throw shapeMismatch(tensor.name(), expected, actual);
      }
    }
    for (int index = expected.length; index < actual.length; index++) {
      if (actual[index] != 1) {
        throw shapeMismatch(tensor.name(), expected, actual);
      }
    }
  }

  private static IllegalArgumentException shapeMismatch(
      String name, long[] expected, long[] actual) {
    return new IllegalArgumentException(
        name
            + " shape must be "
            + Arrays.toString(expected)
            + " with optional singleton dimensions: "
            + Arrays.toString(actual));
  }
}
