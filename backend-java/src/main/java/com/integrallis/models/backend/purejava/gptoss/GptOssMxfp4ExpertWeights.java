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
import com.integrallis.vectors.core.Mxfp4Matrix;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Strict zero-copy mapping of one GPT-OSS layer's Hugging Face MXFP4 expert tensors. */
final class GptOssMxfp4ExpertWeights {

  /**
   * One expert's feed-forward, in whichever shape its artifact provides.
   *
   * <p><b>Fused</b> is the safetensors shape: one {@code gate_up_proj} whose rows interleave gate
   * and up, and one bias of the same interleaved layout. <b>Split</b> is the GGUF shape: {@code
   * ffn_gate_exps} and {@code ffn_up_exps} as separate tensors with separate biases.
   *
   * <p>Two shapes rather than one canonical form because converting either way costs a copy of the
   * weights -- interleaving two tensors, or de-interleaving one -- and these are the largest
   * tensors in the model. The whole point of this class is that it maps them without copying.
   */
  static final class Expert {
    private final Mxfp4Matrix gateUp;
    private final float[] gateUpBias;
    private final GptOssProjection gate;
    private final float[] gateBias;
    private final GptOssProjection up;
    private final float[] upBias;
    private final Mxfp4Matrix down;
    private final GptOssProjection splitDown;
    private final float[] downBias;

    /** The fused safetensors shape. */
    Expert(Mxfp4Matrix gateUp, float[] gateUpBias, Mxfp4Matrix down, float[] downBias) {
      this.gateUp = Objects.requireNonNull(gateUp, "gateUp");
      this.gateUpBias = Objects.requireNonNull(gateUpBias, "gateUpBias");
      this.down = Objects.requireNonNull(down, "down");
      this.downBias = Objects.requireNonNull(downBias, "downBias");
      this.gate = null;
      this.gateBias = null;
      this.up = null;
      this.upBias = null;
      this.splitDown = null;
    }

    /** The split GGUF shape. */
    Expert(
        GptOssProjection gate,
        float[] gateBias,
        GptOssProjection up,
        float[] upBias,
        GptOssProjection down,
        float[] downBias) {
      this.gate = Objects.requireNonNull(gate, "gate");
      this.gateBias = Objects.requireNonNull(gateBias, "gateBias");
      this.up = Objects.requireNonNull(up, "up");
      this.upBias = Objects.requireNonNull(upBias, "upBias");
      this.splitDown = Objects.requireNonNull(down, "down");
      this.downBias = Objects.requireNonNull(downBias, "downBias");
      this.gateUp = null;
      this.gateUpBias = null;
      this.down = null;
    }

    boolean isFused() {
      return gateUp != null;
    }

    Mxfp4Matrix gateUp() {
      return gateUp;
    }

    float[] gateUpBias() {
      return gateUpBias;
    }

    GptOssProjection gate() {
      return gate;
    }

    float[] gateBias() {
      return gateBias;
    }

    GptOssProjection up() {
      return up;
    }

    float[] upBias() {
      return upBias;
    }

    Mxfp4Matrix down() {
      return down;
    }

    GptOssProjection splitDown() {
      return splitDown;
    }

    float[] downBias() {
      return downBias;
    }

    /** The output width, whichever shape this is. */
    int hiddenSize() {
      return isFused() ? gateUp.columns() : gate.columns();
    }

    /** The feed-forward width: half the fused rows, or the split gate's rows. */
    int intermediateSize() {
      return isFused() ? gateUp.rows() / 2 : gate.rows();
    }
  }

  private final Expert[] experts;

  private GptOssMxfp4ExpertWeights(Expert[] experts) {
    this.experts = experts.clone();
  }

  static GptOssMxfp4ExpertWeights of(Expert... experts) {
    Objects.requireNonNull(experts, "experts");
    if (experts.length == 0) {
      throw new IllegalArgumentException("experts must not be empty");
    }
    for (int index = 0; index < experts.length; index++) {
      Objects.requireNonNull(experts[index], "experts[" + index + "]");
    }
    return new GptOssMxfp4ExpertWeights(experts);
  }

  /**
   * Maps one layer's experts from a GGUF file, in the split shape.
   *
   * <p>GGUF stacks all experts into one tensor per projection -- {@code ffn_gate_exps}, {@code
   * ffn_up_exps}, {@code ffn_down_exps} -- and stores MXFP4 as a single interleaved block format
   * rather than the separate blocks and scales the safetensors release uses. So this slices the
   * stacks by expert and hands each slice over as a {@link GptOssProjection}, which routes to
   * {@code ggufMatmul} and its MXFP4 case. No copy, and no second implementation of the block
   * format.
   *
   * <p>The biases are stacked the same way, one row of {@code intermediateSize} (or {@code
   * hiddenSize} for the down projection) per expert.
   */
  static GptOssMxfp4ExpertWeights fromGguf(
      GgufFile file, int layer, int expertCount, int hiddenSize, int intermediateSize) {
    Objects.requireNonNull(file, "file");
    if (layer < 0) {
      throw new IllegalArgumentException("layer must not be negative: " + layer);
    }
    requirePositive(expertCount, "expertCount");
    requireMxfp4Dimension(hiddenSize, "hiddenSize");
    requireMxfp4Dimension(intermediateSize, "intermediateSize");

    String prefix = "blk." + layer + ".";
    GgufTensorData gate = file.getTensor(prefix + "ffn_gate_exps.weight");
    GgufTensorData up = file.getTensor(prefix + "ffn_up_exps.weight");
    GgufTensorData down = file.getTensor(prefix + "ffn_down_exps.weight");
    float[][] gateBias =
        stackedBias(file, prefix + "ffn_gate_exps.bias", expertCount, intermediateSize);
    float[][] upBias =
        stackedBias(file, prefix + "ffn_up_exps.bias", expertCount, intermediateSize);
    float[][] downBias = stackedBias(file, prefix + "ffn_down_exps.bias", expertCount, hiddenSize);

    Expert[] experts = new Expert[expertCount];
    GptOssProjection[] gateSlices =
        slices(gate, intermediateSize, hiddenSize, expertCount, prefix + "ffn_gate_exps.weight");
    GptOssProjection[] upSlices =
        slices(up, intermediateSize, hiddenSize, expertCount, prefix + "ffn_up_exps.weight");
    GptOssProjection[] downSlices =
        slices(down, hiddenSize, intermediateSize, expertCount, prefix + "ffn_down_exps.weight");
    for (int expert = 0; expert < expertCount; expert++) {
      experts[expert] =
          new Expert(
              gateSlices[expert],
              gateBias[expert],
              upSlices[expert],
              upBias[expert],
              downSlices[expert],
              downBias[expert]);
    }
    return of(experts);
  }

  /**
   * Slices a stacked expert tensor into one projection per expert.
   *
   * <p>The cut lands on whole rows, and a row is a whole number of quantization blocks, so no block
   * is straddled -- straddling one reinterprets a block's scale byte as weights and yields
   * plausible, wrong numbers rather than an error.
   */
  private static GptOssProjection[] slices(
      GgufTensorData tensor, int rows, int columns, int expertCount, String name) {
    GgufTensorType type = tensor.type();
    if (columns % type.blockSize() != 0) {
      throw new IllegalArgumentException(
          name
              + ": row length "
              + columns
              + " is not a multiple of the "
              + type.blockSize()
              + "-element "
              + type
              + " block, so experts cannot be sliced exactly");
    }
    long bytesPerRow = (long) (columns / type.blockSize()) * type.typeSize();
    long bytesPerExpert = bytesPerRow * rows;
    MemorySegment segment = tensor.dataSegment();
    long expected = bytesPerExpert * expertCount;
    if (segment.byteSize() != expected) {
      throw new IllegalArgumentException(
          name
              + " is "
              + segment.byteSize()
              + " bytes but "
              + expertCount
              + " experts of "
              + rows
              + "x"
              + columns
              + " need "
              + expected);
    }
    GptOssProjection[] result = new GptOssProjection[expertCount];
    for (int expert = 0; expert < expertCount; expert++) {
      result[expert] =
          GptOssProjection.ofGguf(
              segment.asSlice(bytesPerExpert * expert, bytesPerExpert), type, rows, columns);
    }
    return result;
  }

  /** One bias row per expert, from a stacked {@code [width, expertCount]} tensor. */
  private static float[][] stackedBias(GgufFile file, String name, int expertCount, int width) {
    GgufTensorData tensor = file.getTensor(name);
    float[] all = GgufTensorValues.toFloatArray(tensor);
    if (all.length != Math.multiplyExact(expertCount, width)) {
      throw new IllegalArgumentException(
          name + " has " + all.length + " values but needs " + expertCount * width);
    }
    float[][] rows = new float[expertCount][width];
    for (int expert = 0; expert < expertCount; expert++) {
      System.arraycopy(all, expert * width, rows[expert], 0, width);
    }
    return rows;
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
