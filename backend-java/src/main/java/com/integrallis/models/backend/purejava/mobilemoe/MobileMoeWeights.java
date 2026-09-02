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
package com.integrallis.models.backend.purejava.mobilemoe;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.tensor.TensorSource;
import com.integrallis.models.backend.purejava.tensor.TensorStorage;
import com.integrallis.models.backend.purejava.tensor.TensorView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Complete zero-copy MobileMoE QAT weights over a mapped Safetensors source. */
final class MobileMoeWeights {

  record SharedExpert(
      MobileMoePackedInt4Matrix gate,
      MobileMoePackedInt4Matrix up,
      MobileMoePackedInt4Matrix down) {
    SharedExpert {
      Objects.requireNonNull(gate, "gate");
      Objects.requireNonNull(up, "up");
      Objects.requireNonNull(down, "down");
    }
  }

  record Layer(
      float[] attentionNorm,
      MobileMoePackedInt4Matrix query,
      MobileMoePackedInt4Matrix key,
      MobileMoePackedInt4Matrix value,
      MobileMoePackedInt4Matrix output,
      float[] postAttentionNorm,
      MobileMoeFloat32Matrix router,
      float[] expertBias,
      MobileMoeExperts experts,
      SharedExpert sharedExpert) {
    Layer {
      attentionNorm = Objects.requireNonNull(attentionNorm, "attentionNorm").clone();
      Objects.requireNonNull(query, "query");
      Objects.requireNonNull(key, "key");
      Objects.requireNonNull(value, "value");
      Objects.requireNonNull(output, "output");
      postAttentionNorm = Objects.requireNonNull(postAttentionNorm, "postAttentionNorm").clone();
      Objects.requireNonNull(router, "router");
      expertBias = Objects.requireNonNull(expertBias, "expertBias").clone();
      Objects.requireNonNull(experts, "experts");
      Objects.requireNonNull(sharedExpert, "sharedExpert");
    }

    @Override
    public float[] attentionNorm() {
      return attentionNorm.clone();
    }

    @Override
    public float[] postAttentionNorm() {
      return postAttentionNorm.clone();
    }

    @Override
    public float[] expertBias() {
      return expertBias.clone();
    }
  }

  private final MobileMoePackedInt4Matrix embedding;
  private final float[] outputNorm;
  private final Layer[] layers;

  private MobileMoeWeights(
      MobileMoePackedInt4Matrix embedding, float[] outputNorm, Layer[] layers) {
    this.embedding = Objects.requireNonNull(embedding, "embedding");
    this.outputNorm = Objects.requireNonNull(outputNorm, "outputNorm").clone();
    this.layers = Objects.requireNonNull(layers, "layers").clone();
  }

  static MobileMoeWeights load(TensorSource source, MobileMoeHuggingFaceConfig config) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(config, "config");
    if (!"safetensors".equals(source.format())) {
      throw new IllegalArgumentException("MobileMoE weights require Safetensors");
    }
    int groupSize =
        config
            .quantization()
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "MobileMoE Java path currently requires the packed QAT checkpoint"))
            .groupSize();
    MobileMoePackedInt4Matrix embedding =
        linear(source, "model.embed_tokens", config.vocabSize(), config.hiddenSize(), groupSize);
    Layer[] layers = new Layer[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      String prefix = "model.layers." + layer + ".";
      String attention = prefix + "self_attn.";
      String feedForward = prefix + "feed_forward.";
      layers[layer] =
          new Layer(
              bf16Vector(source, prefix + "input_layernorm.weight", config.hiddenSize()),
              linear(
                  source,
                  attention + "q_proj",
                  config.queryDimension(),
                  config.hiddenSize(),
                  groupSize),
              linear(
                  source,
                  attention + "k_proj",
                  config.keyValueDimension(),
                  config.hiddenSize(),
                  groupSize),
              linear(
                  source,
                  attention + "v_proj",
                  config.keyValueDimension(),
                  config.hiddenSize(),
                  groupSize),
              linear(
                  source,
                  attention + "o_proj",
                  config.hiddenSize(),
                  config.queryDimension(),
                  groupSize),
              bf16Vector(source, prefix + "post_attention_layernorm.weight", config.hiddenSize()),
              f32Matrix(
                  source, feedForward + "router.weight", config.numExperts(), config.hiddenSize()),
              bf16Vector(source, feedForward + "expert_bias", config.numExperts()),
              experts(source, feedForward + "experts.", config, groupSize),
              new SharedExpert(
                  linear(
                      source,
                      feedForward + "shared_expert.gate_proj",
                      config.sharedIntermediateSize(),
                      config.hiddenSize(),
                      groupSize),
                  linear(
                      source,
                      feedForward + "shared_expert.up_proj",
                      config.sharedIntermediateSize(),
                      config.hiddenSize(),
                      groupSize),
                  linear(
                      source,
                      feedForward + "shared_expert.down_proj",
                      config.hiddenSize(),
                      config.sharedIntermediateSize(),
                      groupSize)));
    }
    return new MobileMoeWeights(
        embedding, bf16Vector(source, "model.norm.weight", config.hiddenSize()), layers);
  }

  MobileMoePackedInt4Matrix embedding() {
    return embedding;
  }

  float[] outputNorm() {
    return outputNorm.clone();
  }

  Layer layer(int index) {
    if (index < 0 || index >= layers.length) {
      throw new IndexOutOfBoundsException("layer index out of range: " + index);
    }
    return layers[index];
  }

  private static MobileMoePackedInt4Matrix linear(
      TensorSource source, String base, int rows, int columns, int groupSize) {
    TensorView packed = require(source, base + ".qweight", "U8", 1, 1, rows, columns / 2L);
    TensorView scales =
        require(
            source,
            base + ".weight_scale",
            "F16",
            1,
            Short.BYTES,
            rows,
            columns / (long) groupSize);
    return MobileMoePackedInt4Matrix.of(packed.data(), scales.data(), rows, columns, groupSize);
  }

  private static MobileMoeFloat32Matrix f32Matrix(
      TensorSource source, String name, int rows, int columns) {
    TensorView tensor = require(source, name, "F32", 1, Float.BYTES, rows, columns);
    return MobileMoeFloat32Matrix.of(tensor.data(), rows, columns);
  }

  private static MobileMoeExperts experts(
      TensorSource source, String prefix, MobileMoeHuggingFaceConfig config, int groupSize) {
    int count = config.numExperts();
    int hidden = config.hiddenSize();
    int expertHidden = config.intermediateSize();
    int gateUpOutput = Math.multiplyExact(2, expertHidden);
    TensorView gateUp =
        require(source, prefix + "gate_up_qweight", "U8", 1, 1, count, hidden, gateUpOutput / 2L);
    TensorView gateUpScale =
        require(
            source,
            prefix + "gate_up_scale",
            "F16",
            1,
            Short.BYTES,
            count,
            hidden,
            gateUpOutput / (long) groupSize);
    TensorView down =
        require(source, prefix + "down_qweight", "U8", 1, 1, count, expertHidden, hidden / 2L);
    TensorView downScale =
        require(
            source,
            prefix + "down_scale",
            "F16",
            1,
            Short.BYTES,
            count,
            expertHidden,
            hidden / (long) groupSize);
    long gateBytes = Math.multiplyExact((long) hidden, gateUpOutput / 2L);
    long gateScaleBytes =
        Math.multiplyExact(
            Math.multiplyExact((long) hidden, gateUpOutput / groupSize), Short.BYTES);
    long downBytes = Math.multiplyExact((long) expertHidden, hidden / 2L);
    long downScaleBytes =
        Math.multiplyExact(
            Math.multiplyExact((long) expertHidden, hidden / groupSize), Short.BYTES);
    MobileMoeExperts.Expert[] experts = new MobileMoeExperts.Expert[count];
    for (int expert = 0; expert < count; expert++) {
      experts[expert] =
          new MobileMoeExperts.Expert(
              MobileMoePackedInt4RightMatrix.of(
                  slice(gateUp, expert, gateBytes),
                  slice(gateUpScale, expert, gateScaleBytes),
                  hidden,
                  gateUpOutput,
                  groupSize),
              MobileMoePackedInt4RightMatrix.of(
                  slice(down, expert, downBytes),
                  slice(downScale, expert, downScaleBytes),
                  expertHidden,
                  hidden,
                  groupSize));
    }
    return new MobileMoeExperts(experts);
  }

  private static float[] bf16Vector(TensorSource source, String name, int length) {
    TensorView tensor = require(source, name, "BF16", 1, Short.BYTES, length);
    float[] values = new float[length];
    GgufTensorValues.dequantizeRow(tensor.data(), GgufTensorType.BF16, 0, length, values);
    return values;
  }

  private static TensorView require(
      TensorSource source,
      String name,
      String type,
      int blockElements,
      int blockBytes,
      long... shape) {
    TensorView tensor = source.tensor(name);
    if (!Arrays.equals(tensor.shape(), shape)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(shape)
              + "; got "
              + Arrays.toString(tensor.shape()));
    }
    TensorStorage expected = new TensorStorage("safetensors", type, blockElements, blockBytes);
    if (!expected.equals(tensor.storage())) {
      throw new IllegalArgumentException(
          name + " storage must be " + expected + "; got " + tensor.storage());
    }
    return tensor;
  }

  private static MemorySegment slice(TensorView tensor, int index, long bytes) {
    return tensor.data().asSlice(Math.multiplyExact((long) index, bytes), bytes);
  }
}
