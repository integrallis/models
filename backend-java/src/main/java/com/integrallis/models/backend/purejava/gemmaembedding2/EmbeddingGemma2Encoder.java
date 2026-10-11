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
package com.integrallis.models.backend.purejava.gemmaembedding2;

import com.integrallis.models.backend.purejava.SequenceEncoder;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.util.Arrays;
import java.util.Objects;

/**
 * Whole-sequence EmbeddingGemma 2 text execution. Local attention is symmetric, including both
 * endpoints at half the declared window. The projected per-layer inputs come from the initial
 * scaled embeddings; this architecture has no per-layer token table. Pooling includes all tokens.
 */
public final class EmbeddingGemma2Encoder implements SequenceEncoder {
  private record Matrix(GgufTensorData tensor, int rows, int columns) {}

  private record Layer(
      Matrix query,
      Matrix key,
      Matrix value,
      Matrix attentionOutput,
      Matrix gate,
      Matrix up,
      Matrix down,
      Matrix inputGate,
      Matrix projection,
      float[] attentionNorm,
      float[] queryNorm,
      float[] keyNorm,
      float[] attentionPostNorm,
      float[] ffnNorm,
      float[] ffnPostNorm,
      float[] projectionNorm,
      float scale) {}

  private static final float[] EMPTY_SCRATCH = new float[0];
  private final EmbeddingGemma2Config config;
  private final int contextCapacity;
  private final GgufTensorData embedding;
  private final Matrix perLayerProjection;
  private final float[] perLayerNorm;
  private final float[] outputNorm;
  private final Matrix output;
  private final Layer[] layers;
  private final RotaryTable fullRotary;
  private final RotaryTable slidingRotary;
  private final float[] fullOnes;
  private final float[] slidingOnes;

  private float[] hidden = new float[0];
  private float[] normalized = new float[0];
  private float[] projected = new float[0];
  private float[] perLayer = new float[0];
  private float[] queries = new float[0];
  private float[] keys = new float[0];
  private float[] values = new float[0];
  private float[] attention = new float[0];
  private float[] gate = new float[0];
  private float[] up = new float[0];
  private float[] inputGate = new float[0];
  private float[] outputTokens = new float[0];
  private float[] scores = new float[0];
  private byte[] quantized = new byte[0];
  private float[] scales = new float[0];
  private int[] zeroPoints = new int[0];
  private short[] sums = new short[0];
  private int capacity;

  /** Loads only supported matrix encodings, validates all shapes before any inference. */
  public static EmbeddingGemma2Encoder fromGgufFile(
      GgufFile file, EmbeddingGemma2Config config, int contextCapacity) {
    return new EmbeddingGemma2Encoder(file, config, contextCapacity);
  }

  private EmbeddingGemma2Encoder(GgufFile file, EmbeddingGemma2Config config, int contextCapacity) {
    this.config = Objects.requireNonNull(config, "config");
    if (contextCapacity <= 0 || contextCapacity > config.contextLength()) {
      throw new IllegalArgumentException("context capacity must be within the model context");
    }
    this.contextCapacity = contextCapacity;
    int dim = config.embeddingDim();
    embedding = tensor(file, "token_embd.weight", dim, config.vocabSize());
    perLayerProjection =
        matrix(file, "per_layer_model_proj.weight", config.perLayerDim() * config.numLayers(), dim);
    perLayerNorm = vector(file, "per_layer_proj_norm.weight", config.perLayerDim());
    outputNorm = vector(file, "output_norm.weight", dim);
    output = matrix(file, "output.weight", config.outputDim(), dim);
    layers = new Layer[config.numLayers()];
    for (int index = 0; index < layers.length; index++) {
      String p = "blk." + index + ".";
      int head = config.headDim(index);
      int keyDim = head * config.kvHeads().get(index);
      layers[index] =
          new Layer(
              matrix(file, p + "attn_q.weight", head * config.numHeads(), dim),
              matrix(file, p + "attn_k.weight", keyDim, dim),
              matrix(file, p + "attn_v.weight", keyDim, dim),
              matrix(file, p + "attn_output.weight", dim, head * config.numHeads()),
              matrix(file, p + "ffn_gate.weight", config.hiddenDim(), dim),
              matrix(file, p + "ffn_up.weight", config.hiddenDim(), dim),
              matrix(file, p + "ffn_down.weight", dim, config.hiddenDim()),
              matrix(file, p + "inp_gate.weight", config.perLayerDim(), dim),
              matrix(file, p + "proj.weight", dim, config.perLayerDim()),
              vector(file, p + "attn_norm.weight", dim),
              vector(file, p + "attn_q_norm.weight", head),
              vector(file, p + "attn_k_norm.weight", head),
              vector(file, p + "post_attention_norm.weight", dim),
              vector(file, p + "ffn_norm.weight", dim),
              vector(file, p + "post_ffw_norm.weight", dim),
              vector(file, p + "post_norm.weight", dim),
              vector(file, p + "layer_output_scale.weight", 1)[0]);
    }
    fullRotary = new RotaryTable(config.fullHeadDim(), config.fullRopeTheta(), 1.0f);
    slidingRotary = new RotaryTable(config.slidingHeadDim(), config.slidingRopeTheta(), 1.0f);
    fullOnes = new float[config.fullHeadDim()];
    slidingOnes = new float[config.slidingHeadDim()];
    Arrays.fill(fullOnes, 1.0f);
    Arrays.fill(slidingOnes, 1.0f);
  }

  @Override
  public synchronized float[] encode(int[] tokens) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0 || tokens.length > contextCapacity) {
      throw new IllegalArgumentException("sequence length must be in [1, " + contextCapacity + "]");
    }
    for (int token : tokens) {
      if (token < 0 || token >= config.vocabSize())
        throw new IllegalArgumentException("token out of vocabulary: " + token);
    }
    int count = tokens.length;
    ensureCapacity(count);
    int dim = config.embeddingDim();
    float scale = (float) Math.sqrt(dim);
    float[] row = new float[dim];
    for (int position = 0; position < count; position++) {
      GgufTensorValues.dequantizeRow(
          embedding.dataSegment(), embedding.type(), tokens[position], dim, row);
      for (int i = 0; i < dim; i++) hidden[position * dim + i] = row[i] * scale;
    }
    project(perLayer, hidden, count, perLayerProjection);
    int perLayerElements = count * config.numLayers() * config.perLayerDim();
    float inverse = 1.0f / scale;
    for (int i = 0; i < perLayerElements; i++) perLayer[i] *= inverse;
    normalize(perLayer, perLayer, count * config.numLayers(), perLayerNorm);
    for (int layer = 0; layer < layers.length; layer++) executeLayer(layer, count);
    normalize(normalized, hidden, count, outputNorm);
    project(outputTokens, normalized, count, output);
    float[] pooled = new float[config.outputDim()];
    for (int i = 0; i < pooled.length; i++) {
      double sum = 0;
      for (int position = 0; position < count; position++)
        sum += outputTokens[position * pooled.length + i];
      pooled[i] = (float) (sum / count);
    }
    return pooled;
  }

  private void executeLayer(int index, int count) {
    Layer layer = layers[index];
    int dim = config.embeddingDim();
    normalize(normalized, hidden, count, layer.attentionNorm());
    project(queries, normalized, count, layer.query());
    project(keys, normalized, count, layer.key());
    project(values, normalized, count, layer.value());
    int headDim = config.headDim(index);
    int heads = config.numHeads();
    int kvHeads = config.kvHeads().get(index);
    boolean sliding = config.slidingLayers().get(index);
    normalize(queries, queries, count * heads, layer.queryNorm());
    normalize(keys, keys, count * kvHeads, layer.keyNorm());
    normalize(values, values, count * kvHeads, sliding ? slidingOnes : fullOnes);
    RotaryTable rotary = sliding ? slidingRotary : fullRotary;
    for (int position = 0; position < count; position++) {
      rotary.prepare(position);
      for (int head = 0; head < heads; head++)
        rotary.apply(queries, (position * heads + head) * headDim, true);
      for (int head = 0; head < kvHeads; head++)
        rotary.apply(keys, (position * kvHeads + head) * headDim, true);
    }
    attend(
        queries,
        keys,
        values,
        attention,
        scores,
        count,
        heads,
        kvHeads,
        headDim,
        sliding ? config.slidingWindow() : 0);
    project(projected, attention, count, layer.attentionOutput());
    normalize(projected, projected, count, layer.attentionPostNorm());
    add(hidden, projected, count * dim);
    normalize(normalized, hidden, count, layer.ffnNorm());
    project(gate, normalized, count, layer.gate());
    project(up, normalized, count, layer.up());
    TensorOps.geluGlu(gate, gate, up, count * config.hiddenDim());
    project(projected, gate, count, layer.down());
    normalize(projected, projected, count, layer.ffnPostNorm());
    add(hidden, projected, count * dim);
    project(inputGate, hidden, count, layer.inputGate());
    int perDim = config.perLayerDim();
    TensorOps.gelu(inputGate, 0, inputGate, 0, count * perDim);
    for (int position = 0; position < count; position++) {
      int source = (position * layers.length + index) * perDim;
      for (int i = 0; i < perDim; i++) inputGate[position * perDim + i] *= perLayer[source + i];
    }
    project(projected, inputGate, count, layer.projection());
    normalize(projected, projected, count, layer.projectionNorm());
    for (int i = 0; i < count * dim; i++) hidden[i] = (hidden[i] + projected[i]) * layer.scale();
  }

  /** Attention uses scale 1: the learned query norm already supplies the scale. */
  static void attend(
      float[] queries,
      float[] keys,
      float[] values,
      float[] out,
      float[] scores,
      int count,
      int heads,
      int kvHeads,
      int headDim,
      int window) {
    int queryDim = heads * headDim;
    int keyDim = kvHeads * headDim;
    Arrays.fill(out, 0, count * queryDim, 0);
    for (int position = 0; position < count; position++) {
      int start = window == 0 ? 0 : Math.max(0, position - window / 2);
      int end = window == 0 ? count : Math.min(count, position + window / 2 + 1);
      for (int head = 0; head < heads; head++) {
        int kvHead = head / (heads / kvHeads);
        int queryOffset = position * queryDim + head * headDim;
        for (int keyPosition = start; keyPosition < end; keyPosition++) {
          scores[keyPosition] =
              PinnedReduction.dot(
                  queries, queryOffset, keys, keyPosition * keyDim + kvHead * headDim, headDim);
        }
        TensorOps.softmax(scores, start, end - start);
        for (int keyPosition = start; keyPosition < end; keyPosition++) {
          float weight = scores[keyPosition];
          int valueOffset = keyPosition * keyDim + kvHead * headDim;
          for (int i = 0; i < headDim; i++)
            out[queryOffset + i] += weight * values[valueOffset + i];
        }
      }
    }
  }

  private void normalize(float[] out, float[] input, int rows, float[] norm) {
    for (int row = 0; row < rows; row++) {
      int offset = row * norm.length;
      TensorOps.rmsNorm(out, offset, input, offset, norm, norm.length, config.epsilon());
    }
  }

  private void project(float[] out, float[] input, int count, Matrix matrix) {
    TensorOps.ggufBatchedMatmul(
        out,
        input,
        matrix.tensor().dataSegment(),
        matrix.tensor().type(),
        count,
        matrix.rows(),
        matrix.columns(),
        quantized,
        scales,
        zeroPoints,
        sums,
        EMPTY_SCRATCH,
        GgufQ4Kernel.WIDENED);
  }

  private static void add(float[] out, float[] addend, int count) {
    for (int i = 0; i < count; i++) out[i] += addend[i];
  }

  private void ensureCapacity(int count) {
    if (capacity >= count) return;
    int dim = config.embeddingDim();
    hidden = buffer(count, dim);
    normalized = buffer(count, dim);
    projected = buffer(count, dim);
    perLayer = buffer(count, Math.multiplyExact(config.perLayerDim(), config.numLayers()));
    int queryDim =
        Math.multiplyExact(
            config.numHeads(), Math.max(config.fullHeadDim(), config.slidingHeadDim()));
    queries = buffer(count, queryDim);
    keys = buffer(count, queryDim);
    values = buffer(count, queryDim);
    attention = buffer(count, queryDim);
    gate = buffer(count, config.hiddenDim());
    up = buffer(count, config.hiddenDim());
    inputGate = buffer(count, config.perLayerDim());
    outputTokens = buffer(count, config.outputDim());
    scores = new float[count];
    int width =
        Math.max(Math.max(dim, queryDim), Math.max(config.hiddenDim(), config.perLayerDim()));
    quantized = new byte[Math.multiplyExact(count, width)];
    scales = buffer(count, (width + 31) / 32);
    zeroPoints = new int[Math.multiplyExact(count, (width + 3) / 4)];
    sums = new short[Math.multiplyExact(count, (width + 15) / 16)];
    capacity = count;
  }

  private static float[] buffer(int count, int width) {
    return new float[Math.multiplyExact(count, width)];
  }

  private static GgufTensorData tensor(GgufFile file, String name, long... shape) {
    GgufTensorData tensor = file.getTensor(name);
    if (!Arrays.equals(tensor.shape(), shape)) {
      throw new IllegalArgumentException(
          "Unexpected shape for " + name + ": " + Arrays.toString(tensor.shape()));
    }
    return tensor;
  }

  private static Matrix matrix(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = tensor(file, name, columns, rows);
    // These are the exact-artifact encodings validated for this encoder. Other quantizations
    // need their own fixture and oracle run, especially kernels with different scratch contracts.
    switch (tensor.type()) {
      case BF16, F16, Q8_0 -> {}
      default ->
          throw new UnsupportedOperationException("EmbeddingGemma 2 matrix type: " + tensor.type());
    }
    return new Matrix(tensor, rows, columns);
  }

  private static float[] vector(GgufFile file, String name, int size) {
    float[] values = GgufTensorValues.toFloatArray(tensor(file, name, size));
    for (float value : values) {
      if (!Float.isFinite(value)) throw new IllegalArgumentException("Nonfinite weight in " + name);
    }
    return values;
  }
}
