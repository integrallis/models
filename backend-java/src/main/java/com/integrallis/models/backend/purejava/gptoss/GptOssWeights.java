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
package com.integrallis.models.backend.purejava.gptoss;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.tensor.TensorSource;
import com.integrallis.models.backend.purejava.tensor.TensorStorage;
import com.integrallis.models.backend.purejava.tensor.TensorView;
import com.integrallis.vectors.core.BFloat16Matrix;
import java.util.Arrays;
import java.util.Objects;

/** Zero-copy mapped dense and expert weights for a GPT-OSS Safetensors checkpoint. */
final class GptOssWeights {

  record Layer(
      float[] attentionNorm,
      GptOssProjection query,
      float[] queryBias,
      GptOssProjection key,
      float[] keyBias,
      GptOssProjection value,
      float[] valueBias,
      GptOssProjection output,
      float[] outputBias,
      float[] sinks,
      float[] postAttentionNorm,
      GptOssProjection router,
      float[] routerBias,
      GptOssMxfp4ExpertWeights experts) {

    Layer {
      attentionNorm = Objects.requireNonNull(attentionNorm, "attentionNorm").clone();
      Objects.requireNonNull(query, "query");
      queryBias = Objects.requireNonNull(queryBias, "queryBias").clone();
      Objects.requireNonNull(key, "key");
      keyBias = Objects.requireNonNull(keyBias, "keyBias").clone();
      Objects.requireNonNull(value, "value");
      valueBias = Objects.requireNonNull(valueBias, "valueBias").clone();
      Objects.requireNonNull(output, "output");
      outputBias = Objects.requireNonNull(outputBias, "outputBias").clone();
      sinks = Objects.requireNonNull(sinks, "sinks").clone();
      postAttentionNorm = Objects.requireNonNull(postAttentionNorm, "postAttentionNorm").clone();
      Objects.requireNonNull(router, "router");
      routerBias = Objects.requireNonNull(routerBias, "routerBias").clone();
      Objects.requireNonNull(experts, "experts");
    }

    @Override
    public float[] attentionNorm() {
      return attentionNorm.clone();
    }

    @Override
    public float[] queryBias() {
      return queryBias.clone();
    }

    @Override
    public float[] keyBias() {
      return keyBias.clone();
    }

    @Override
    public float[] valueBias() {
      return valueBias.clone();
    }

    @Override
    public float[] outputBias() {
      return outputBias.clone();
    }

    @Override
    public float[] sinks() {
      return sinks.clone();
    }

    @Override
    public float[] postAttentionNorm() {
      return postAttentionNorm.clone();
    }

    @Override
    public float[] routerBias() {
      return routerBias.clone();
    }
  }

  private final GptOssProjection tokenEmbedding;
  private final float[] outputNorm;
  private final GptOssProjection output;
  private final Layer[] layers;

  private GptOssWeights(
      GptOssProjection tokenEmbedding,
      float[] outputNorm,
      GptOssProjection output,
      Layer[] layers) {
    this.tokenEmbedding = Objects.requireNonNull(tokenEmbedding, "tokenEmbedding");
    this.outputNorm = Objects.requireNonNull(outputNorm, "outputNorm").clone();
    this.output = Objects.requireNonNull(output, "output");
    this.layers = Objects.requireNonNull(layers, "layers").clone();
  }

  /**
   * Maps a complete GPT-OSS decoder from a GGUF file.
   *
   * <p>The same weights as {@link #load}, differently stored: attention matrices are K-quants
   * rather than BF16, the experts arrive as three stacked tensors rather than fused blocks and
   * scales, and the head is its own tensor rather than optionally tied. {@link GptOssProjection} is
   * what lets the graph be shared across both.
   */
  static GptOssWeights fromGguf(GgufFile file, GptOssConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");

    GptOssProjection tokenEmbedding =
        ggufMatrix(file, "token_embd.weight", config.vocabSize(), config.hiddenSize());
    GptOssProjection output =
        file.hasTensor("output.weight")
            ? ggufMatrix(file, "output.weight", config.vocabSize(), config.hiddenSize())
            : tokenEmbedding;
    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "blk." + layer + ".";
      layers[layer] =
          new Layer(
              ggufVector(file, prefix + "attn_norm.weight", config.hiddenSize()),
              ggufMatrix(
                  file, prefix + "attn_q.weight", config.queryDimension(), config.hiddenSize()),
              ggufVector(file, prefix + "attn_q.bias", config.queryDimension()),
              ggufMatrix(
                  file, prefix + "attn_k.weight", config.keyValueDimension(), config.hiddenSize()),
              ggufVector(file, prefix + "attn_k.bias", config.keyValueDimension()),
              ggufMatrix(
                  file, prefix + "attn_v.weight", config.keyValueDimension(), config.hiddenSize()),
              ggufVector(file, prefix + "attn_v.bias", config.keyValueDimension()),
              ggufMatrix(
                  file,
                  prefix + "attn_output.weight",
                  config.hiddenSize(),
                  config.queryDimension()),
              ggufVector(file, prefix + "attn_output.bias", config.hiddenSize()),
              // One learned logit per query head, added to the softmax denominator.
              ggufVector(file, prefix + "attn_sinks.weight", config.numHeads()),
              ggufVector(file, prefix + "post_attention_norm.weight", config.hiddenSize()),
              ggufMatrix(
                  file, prefix + "ffn_gate_inp.weight", config.numExperts(), config.hiddenSize()),
              ggufVector(file, prefix + "ffn_gate_inp.bias", config.numExperts()),
              GptOssMxfp4ExpertWeights.fromGguf(
                  file,
                  layer,
                  config.numExperts(),
                  config.hiddenSize(),
                  config.intermediateSize()));
    }
    return new GptOssWeights(
        tokenEmbedding,
        ggufVector(file, "output_norm.weight", config.hiddenSize()),
        output,
        layers);
  }

  private static GptOssProjection ggufMatrix(GgufFile file, String name, int rows, int columns) {
    GgufTensorData tensor = file.getTensor(name);
    long[] expected = {columns, rows};
    if (!Arrays.equals(expected, tensor.shape())) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expected)
              + ", found "
              + Arrays.toString(tensor.shape()));
    }
    return GptOssProjection.ofGguf(tensor.dataSegment(), tensor.type(), rows, columns);
  }

  private static float[] ggufVector(GgufFile file, String name, int length) {
    GgufTensorData tensor = file.getTensor(name);
    long[] expected = {length};
    if (!Arrays.equals(expected, tensor.shape())) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expected)
              + ", found "
              + Arrays.toString(tensor.shape()));
    }
    return GgufTensorValues.toFloatArray(tensor);
  }

  static GptOssWeights load(TensorSource source, GptOssConfig config) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(config, "config");
    if (!"safetensors".equals(source.format())) {
      throw new IllegalArgumentException(
          "GPT-OSS weights require Safetensors; got " + source.format());
    }

    GptOssProjection tokenEmbedding =
        matrix(source, "model.embed_tokens.weight", config.vocabSize(), config.hiddenSize());
    GptOssProjection output =
        config.tieWordEmbeddings()
            ? tokenEmbedding
            : matrix(source, "lm_head.weight", config.vocabSize(), config.hiddenSize());
    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "model.layers." + layer + ".";
      layers[layer] =
          new Layer(
              vector(source, prefix + "input_layernorm.weight", config.hiddenSize()),
              matrix(
                  source,
                  prefix + "self_attn.q_proj.weight",
                  config.queryDimension(),
                  config.hiddenSize()),
              attentionBias(
                  source,
                  prefix + "self_attn.q_proj.bias",
                  config.queryDimension(),
                  config.attentionBias()),
              matrix(
                  source,
                  prefix + "self_attn.k_proj.weight",
                  config.keyValueDimension(),
                  config.hiddenSize()),
              attentionBias(
                  source,
                  prefix + "self_attn.k_proj.bias",
                  config.keyValueDimension(),
                  config.attentionBias()),
              matrix(
                  source,
                  prefix + "self_attn.v_proj.weight",
                  config.keyValueDimension(),
                  config.hiddenSize()),
              attentionBias(
                  source,
                  prefix + "self_attn.v_proj.bias",
                  config.keyValueDimension(),
                  config.attentionBias()),
              matrix(
                  source,
                  prefix + "self_attn.o_proj.weight",
                  config.hiddenSize(),
                  config.queryDimension()),
              attentionBias(
                  source,
                  prefix + "self_attn.o_proj.bias",
                  config.hiddenSize(),
                  config.attentionBias()),
              vector(source, prefix + "self_attn.sinks", config.numHeads()),
              vector(source, prefix + "post_attention_layernorm.weight", config.hiddenSize()),
              matrix(
                  source, prefix + "mlp.router.weight", config.numExperts(), config.hiddenSize()),
              vector(source, prefix + "mlp.router.bias", config.numExperts()),
              GptOssMxfp4ExpertWeights.load(
                  source,
                  layer,
                  config.numExperts(),
                  config.hiddenSize(),
                  config.intermediateSize()));
    }
    return new GptOssWeights(
        tokenEmbedding, vector(source, "model.norm.weight", config.hiddenSize()), output, layers);
  }

  GptOssProjection tokenEmbedding() {
    return tokenEmbedding;
  }

  float[] outputNorm() {
    return outputNorm.clone();
  }

  GptOssProjection output() {
    return output;
  }

  Layer layer(int index) {
    if (index < 0 || index >= layers.length) {
      throw new IndexOutOfBoundsException(
          "layer index is outside [0, " + layers.length + "): " + index);
    }
    return layers[index];
  }

  private static float[] attentionBias(
      TensorSource source, String name, int length, boolean enabled) {
    return enabled ? vector(source, name, length) : new float[length];
  }

  private static GptOssProjection matrix(TensorSource source, String name, int rows, int columns) {
    TensorView tensor = requireBf16(source, name, rows, columns);
    return GptOssProjection.ofBFloat16(BFloat16Matrix.of(tensor.data(), rows, columns));
  }

  private static float[] vector(TensorSource source, String name, int length) {
    TensorView tensor = requireBf16(source, name, length);
    float[] values = new float[length];
    GgufTensorValues.dequantizeRow(tensor.data(), GgufTensorType.BF16, 0, length, values);
    return values;
  }

  private static TensorView requireBf16(TensorSource source, String name, long... expectedShape) {
    TensorView tensor = source.tensor(name);
    if (!Arrays.equals(tensor.shape(), expectedShape)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expectedShape)
              + "; got "
              + Arrays.toString(tensor.shape()));
    }
    TensorStorage expected = new TensorStorage("safetensors", "BF16", 1, Short.BYTES);
    if (!expected.equals(tensor.storage())) {
      throw new IllegalArgumentException(
          name + " storage must be " + expected + "; got " + tensor.storage());
    }
    return tensor;
  }
}
