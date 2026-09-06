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
package com.integrallis.models.backend.purejava.deberta;

import com.integrallis.models.backend.purejava.tensor.TensorSource;
import com.integrallis.models.backend.purejava.tensor.TensorStorage;
import com.integrallis.models.backend.purejava.tensor.TensorView;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/** DeBERTa-v2 reranker weights expanded once from immutable F16 Safetensors storage. */
final class DebertaV2Weights {

  static final class Matrix {
    private final float[] values;
    private final int rows;
    private final int columns;

    private Matrix(float[] values, int rows, int columns) {
      this.values = values;
      this.rows = rows;
      this.columns = columns;
    }

    float[] values() {
      return values;
    }

    int rows() {
      return rows;
    }

    int columns() {
      return columns;
    }
  }

  record Layer(
      Matrix query,
      float[] queryBias,
      Matrix key,
      float[] keyBias,
      Matrix value,
      float[] valueBias,
      Matrix attentionOutput,
      float[] attentionOutputBias,
      float[] attentionNormWeight,
      float[] attentionNormBias,
      Matrix feedForwardUp,
      float[] feedForwardUpBias,
      Matrix feedForwardDown,
      float[] feedForwardDownBias,
      float[] outputNormWeight,
      float[] outputNormBias) {}

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final Matrix wordEmbeddings;
  private final float[] embeddingNormWeight;
  private final float[] embeddingNormBias;
  private final Matrix relativeEmbeddings;
  private final float[] relativeNormWeight;
  private final float[] relativeNormBias;
  private final Layer[] layers;
  private final Matrix pooler;
  private final float[] poolerBias;
  private final float[] classifierWeight;
  private final float[] classifierBias;

  private DebertaV2Weights(
      Matrix wordEmbeddings,
      float[] embeddingNormWeight,
      float[] embeddingNormBias,
      Matrix relativeEmbeddings,
      float[] relativeNormWeight,
      float[] relativeNormBias,
      Layer[] layers,
      Matrix pooler,
      float[] poolerBias,
      float[] classifierWeight,
      float[] classifierBias) {
    this.wordEmbeddings = wordEmbeddings;
    this.embeddingNormWeight = embeddingNormWeight;
    this.embeddingNormBias = embeddingNormBias;
    this.relativeEmbeddings = relativeEmbeddings;
    this.relativeNormWeight = relativeNormWeight;
    this.relativeNormBias = relativeNormBias;
    this.layers = layers;
    this.pooler = pooler;
    this.poolerBias = poolerBias;
    this.classifierWeight = classifierWeight;
    this.classifierBias = classifierBias;
  }

  static DebertaV2Weights load(TensorSource source, DebertaV2Config config) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(config, "config");
    int dim = config.hiddenSize();
    int relativeRows = Math.multiplyExact(config.positionBuckets(), 2);
    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "deberta.encoder.layer." + layer + ".";
      layers[layer] =
          new Layer(
              matrix(source, prefix + "attention.self.query_proj.weight", dim, dim),
              vector(source, prefix + "attention.self.query_proj.bias", dim),
              matrix(source, prefix + "attention.self.key_proj.weight", dim, dim),
              vector(source, prefix + "attention.self.key_proj.bias", dim),
              matrix(source, prefix + "attention.self.value_proj.weight", dim, dim),
              vector(source, prefix + "attention.self.value_proj.bias", dim),
              matrix(source, prefix + "attention.output.dense.weight", dim, dim),
              vector(source, prefix + "attention.output.dense.bias", dim),
              vector(source, prefix + "attention.output.LayerNorm.weight", dim),
              vector(source, prefix + "attention.output.LayerNorm.bias", dim),
              matrix(source, prefix + "intermediate.dense.weight", config.intermediateSize(), dim),
              vector(source, prefix + "intermediate.dense.bias", config.intermediateSize()),
              matrix(source, prefix + "output.dense.weight", dim, config.intermediateSize()),
              vector(source, prefix + "output.dense.bias", dim),
              vector(source, prefix + "output.LayerNorm.weight", dim),
              vector(source, prefix + "output.LayerNorm.bias", dim));
    }
    return new DebertaV2Weights(
        matrix(source, "deberta.embeddings.word_embeddings.weight", config.vocabSize(), dim),
        vector(source, "deberta.embeddings.LayerNorm.weight", dim),
        vector(source, "deberta.embeddings.LayerNorm.bias", dim),
        matrix(source, "deberta.encoder.rel_embeddings.weight", relativeRows, dim),
        vector(source, "deberta.encoder.LayerNorm.weight", dim),
        vector(source, "deberta.encoder.LayerNorm.bias", dim),
        layers,
        matrix(source, "pooler.dense.weight", config.poolerHiddenSize(), dim),
        vector(source, "pooler.dense.bias", config.poolerHiddenSize()),
        matrix(source, "classifier.weight", 1, config.poolerHiddenSize()).values(),
        vector(source, "classifier.bias", 1));
  }

  Matrix wordEmbeddings() {
    return wordEmbeddings;
  }

  float[] embeddingNormWeight() {
    return embeddingNormWeight;
  }

  float[] embeddingNormBias() {
    return embeddingNormBias;
  }

  Matrix relativeEmbeddings() {
    return relativeEmbeddings;
  }

  float[] relativeNormWeight() {
    return relativeNormWeight;
  }

  float[] relativeNormBias() {
    return relativeNormBias;
  }

  int numLayers() {
    return layers.length;
  }

  Layer layer(int index) {
    return layers[index];
  }

  Matrix pooler() {
    return pooler;
  }

  float[] poolerBias() {
    return poolerBias;
  }

  float[] classifierWeight() {
    return classifierWeight;
  }

  float[] classifierBias() {
    return classifierBias;
  }

  void wordEmbedding(int token, float[] output) {
    if (token < 0 || token >= wordEmbeddings.rows()) {
      throw new IllegalArgumentException("token is outside the embedding table: " + token);
    }
    System.arraycopy(
        wordEmbeddings.values(), token * wordEmbeddings.columns(), output, 0, output.length);
  }

  private static Matrix matrix(TensorSource source, String name, int rows, int columns) {
    return new Matrix(values(source, name, rows, columns), rows, columns);
  }

  private static float[] vector(TensorSource source, String name, int length) {
    return values(source, name, length);
  }

  private static float[] values(TensorSource source, String name, long... shape) {
    TensorView tensor = source.tensor(name);
    if (!Arrays.equals(tensor.shape(), shape)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(shape)
              + "; got "
              + Arrays.toString(tensor.shape()));
    }
    TensorStorage expected = new TensorStorage("safetensors", "F16", 1, Short.BYTES);
    if (!expected.equals(tensor.storage())) {
      throw new IllegalArgumentException(
          name + " must use Safetensors F16; got " + tensor.storage());
    }
    int elements = 1;
    for (long dimension : shape) {
      elements = Math.multiplyExact(elements, Math.toIntExact(dimension));
    }
    float[] values = new float[elements];
    MemorySegment data = tensor.data();
    for (int index = 0; index < elements; index++) {
      values[index] = Float.float16ToFloat(data.get(LE_SHORT, (long) index * Short.BYTES));
    }
    return values;
  }
}
