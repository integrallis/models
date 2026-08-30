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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.tensor.TensorSource;
import com.integrallis.models.backend.purejava.tensor.TensorStorage;
import com.integrallis.models.backend.purejava.tensor.TensorView;
import com.integrallis.vectors.core.Mxfp4Matrix;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Strict zero-copy mapping of one GPT-OSS layer's Hugging Face MXFP4 expert tensors. */
final class GptOssMxfp4ExpertWeights {

  static final class Expert {
    private final Mxfp4Matrix gateUp;
    private final float[] gateUpBias;
    private final Mxfp4Matrix down;
    private final float[] downBias;

    private Expert(Mxfp4Matrix gateUp, float[] gateUpBias, Mxfp4Matrix down, float[] downBias) {
      this.gateUp = Objects.requireNonNull(gateUp, "gateUp");
      this.gateUpBias = Objects.requireNonNull(gateUpBias, "gateUpBias");
      this.down = Objects.requireNonNull(down, "down");
      this.downBias = Objects.requireNonNull(downBias, "downBias");
    }

    Mxfp4Matrix gateUp() {
      return gateUp;
    }

    float[] gateUpBias() {
      return gateUpBias;
    }

    Mxfp4Matrix down() {
      return down;
    }

    float[] downBias() {
      return downBias;
    }
  }

  private final Expert[] experts;

  private GptOssMxfp4ExpertWeights(Expert[] experts) {
    this.experts = experts;
  }

  static GptOssMxfp4ExpertWeights load(
      TensorSource source, int layer, int expertCount, int hiddenSize, int intermediateSize) {
    Objects.requireNonNull(source, "source");
    if (!"safetensors".equals(source.format())) {
      throw new IllegalArgumentException(
          "GPT-OSS weights require Safetensors; got " + source.format());
    }
    if (layer < 0) {
      throw new IllegalArgumentException("layer must not be negative: " + layer);
    }
    requirePositive(expertCount, "expertCount");
    requireMxfp4Dimension(hiddenSize, "hiddenSize");
    requireMxfp4Dimension(intermediateSize, "intermediateSize");

    String prefix = "model.layers." + layer + ".mlp.experts.";
    long gateUpRows = Math.multiplyExact(2L, intermediateSize);
    TensorView gateUpBlocks =
        require(
            source,
            prefix + "gate_up_proj_blocks",
            "U8",
            1,
            1,
            expertCount,
            gateUpRows,
            hiddenSize / 32L,
            16);
    TensorView gateUpScales =
        require(
            source,
            prefix + "gate_up_proj_scales",
            "U8",
            1,
            1,
            expertCount,
            gateUpRows,
            hiddenSize / 32L);
    TensorView gateUpBias =
        require(
            source, prefix + "gate_up_proj_bias", "BF16", 1, Short.BYTES, expertCount, gateUpRows);
    TensorView downBlocks =
        require(
            source,
            prefix + "down_proj_blocks",
            "U8",
            1,
            1,
            expertCount,
            hiddenSize,
            intermediateSize / 32L,
            16);
    TensorView downScales =
        require(
            source,
            prefix + "down_proj_scales",
            "U8",
            1,
            1,
            expertCount,
            hiddenSize,
            intermediateSize / 32L);
    TensorView downBias =
        require(source, prefix + "down_proj_bias", "BF16", 1, Short.BYTES, expertCount, hiddenSize);

    long gateUpBlockBytes = Math.multiplyExact(gateUpRows, hiddenSize / 2L);
    long gateUpScaleBytes = Math.multiplyExact(gateUpRows, hiddenSize / 32L);
    long downBlockBytes = Math.multiplyExact((long) hiddenSize, intermediateSize / 2L);
    long downScaleBytes = Math.multiplyExact((long) hiddenSize, intermediateSize / 32L);
    Expert[] experts = new Expert[expertCount];
    for (int expert = 0; expert < expertCount; expert++) {
      Mxfp4Matrix gateUp =
          Mxfp4Matrix.of(
              slice(gateUpBlocks, expert, gateUpBlockBytes),
              slice(gateUpScales, expert, gateUpScaleBytes),
              Math.multiplyExact(2, intermediateSize),
              hiddenSize);
      Mxfp4Matrix down =
          Mxfp4Matrix.of(
              slice(downBlocks, expert, downBlockBytes),
              slice(downScales, expert, downScaleBytes),
              hiddenSize,
              intermediateSize);
      experts[expert] =
          new Expert(
              gateUp,
              loadBf16Row(gateUpBias, expert, Math.multiplyExact(2, intermediateSize)),
              down,
              loadBf16Row(downBias, expert, hiddenSize));
    }
    return new GptOssMxfp4ExpertWeights(experts);
  }

  int expertCount() {
    return experts.length;
  }

  Expert expert(int index) {
    if (index < 0 || index >= experts.length) {
      throw new IndexOutOfBoundsException(
          "expert index " + index + " is outside [0, " + experts.length + ")");
    }
    return experts[index];
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

  private static MemorySegment slice(TensorView tensor, int index, long bytesPerExpert) {
    return tensor.data().asSlice(Math.multiplyExact((long) index, bytesPerExpert), bytesPerExpert);
  }

  private static float[] loadBf16Row(TensorView tensor, int row, int columns) {
    long bytes = Math.multiplyExact((long) columns, Short.BYTES);
    MemorySegment data = tensor.data().asSlice(Math.multiplyExact((long) row, bytes), bytes);
    float[] values = new float[columns];
    GgufTensorValues.dequantizeRow(data, GgufTensorType.BF16, 0, columns, values);
    return values;
  }

  private static void requireMxfp4Dimension(int value, String name) {
    requirePositive(value, name);
    if (value % 32 != 0) {
      throw new IllegalArgumentException(name + " must be a multiple of 32: " + value);
    }
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive: " + value);
    }
  }
}
