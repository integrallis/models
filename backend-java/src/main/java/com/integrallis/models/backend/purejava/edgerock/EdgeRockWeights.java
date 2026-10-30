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
package com.integrallis.models.backend.purejava.edgerock;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Resolves EdgeRock's tensors from a GGUF file.
 *
 * <p>Two things this does that a looser loader would not.
 *
 * <p><strong>It refuses a tensor whose shape disagrees with the metadata.</strong> The header says
 * {@code embedding_length}, {@code head_count} and so on; the tensors say what is actually stored.
 * A projection built from the wrong width still multiplies and still returns numbers, so the two
 * are checked against each other at load.
 *
 * <p><strong>It requires the memory tensors on exactly the full-attention layers.</strong>
 * EdgeRock-4B carries {@code memory.*} on 8 of 32 layers, spaced by {@code
 * full_attention_interval}. A file that carried them elsewhere -- or omitted them -- would
 * otherwise load into a decoder that silently skipped a layer's memory.
 */
public final class EdgeRockWeights {

  private final GgufFile file;
  private final EdgeRockConfig config;

  private EdgeRockWeights(GgufFile file, EdgeRockConfig config) {
    this.file = Objects.requireNonNull(file, "file");
    this.config = Objects.requireNonNull(config, "config");
  }

  public static EdgeRockWeights load(GgufFile file, EdgeRockConfig config) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(config, "config");
    QuantizedProjection.requireNoHadamard(
        file.metadata().get("prism.hadamard.version").isPresent());
    EdgeRockWeights weights = new EdgeRockWeights(file, config);
    weights.verifyLayerLayout();
    return weights;
  }

  /**
   * Checks that memory tensors appear on exactly the layers the interval predicts.
   *
   * <p>Both directions matter. A missing bank means a layer decodes without its memory; a bank on a
   * recurrent layer means the file disagrees with {@code full_attention_interval}, and the decoder
   * would place attention on the wrong layers while loading cleanly.
   */
  private void verifyLayerLayout() {
    for (int layer = 0; layer < config.blockCount(); layer++) {
      boolean expected = config.isFullAttentionLayer(layer);
      boolean present = file.hasTensor("blk." + layer + ".memory.q_proj.weight");
      if (expected != present) {
        throw new IllegalArgumentException(
            "layer "
                + layer
                + (expected
                    ? " is a full-attention layer but carries no memory.q_proj.weight"
                    : " is recurrent but carries memory.q_proj.weight; the file disagrees with "
                        + "full_attention_interval "
                        + config.fullAttentionInterval()));
      }
      boolean attention = file.hasTensor("blk." + layer + ".attn_q.weight");
      if (expected != attention) {
        throw new IllegalArgumentException(
            "layer " + layer + " attention tensors disagree with full_attention_interval");
      }
    }
  }

  /** The memory layer weights for one full-attention layer. */
  public EdgeRockMemoryLayer.Weights memoryLayer(int layer) {
    if (!config.isFullAttentionLayer(layer)) {
      throw new IllegalArgumentException("layer " + layer + " carries no memory bank");
    }
    String p = "blk." + layer + ".memory.";
    int embedding = config.embeddingLength();
    int heads = config.headCount();
    int groups = config.headCountKv();
    int dim = config.headDim();
    int writer = config.memoryWriterIntermediate();
    int rank = config.memoryAdmissionRank();
    return new EdgeRockMemoryLayer.Weights(
        vector(p + "input_norm.weight", embedding),
        projection(p + "q_proj.weight", heads * dim, embedding),
        vector(p + "q_norm.weight", dim),
        projection(p + "k_proj.weight", groups * dim, embedding),
        vector(p + "k_norm.weight", dim),
        projection(p + "v_proj.weight", groups * dim, embedding),
        projection(p + "o_proj.weight", embedding, heads * dim),
        projection(p + "write_gate_proj.weight", writer, embedding),
        projection(p + "write_up_proj.weight", writer, embedding),
        projection(p + "write_down_proj.weight", embedding, writer),
        projection(p + "admission_down.weight", rank, embedding),
        projection(p + "admission_up.weight", groups, rank),
        rows(p + "read_gate_weight", heads, dim),
        vector(p + "read_gate_bias", heads),
        vector(p + "read_scale", heads));
  }

  /**
   * A projection, with its stored shape checked against the widths the metadata implies.
   *
   * <p>GGUF stores a matrix as {@code [columns, rows]}: the first dimension is the input width.
   */
  private QuantizedProjection projection(String name, int rows, int columns) {
    GgufTensorData tensor = require(name);
    long[] shape = tensor.info().shape();
    if (shape.length != 2 || shape[0] != columns || shape[1] != rows) {
      throw new IllegalArgumentException(
          name
              + " is "
              + java.util.Arrays.toString(shape)
              + " but the metadata implies ["
              + columns
              + ", "
              + rows
              + "]");
    }
    return new QuantizedProjection(tensor.dataSegment(), tensor.type(), rows, columns, name);
  }

  /** An F32 vector such as a norm weight or a per-head scale. */
  private float[] vector(String name, int length) {
    GgufTensorData tensor = require(name);
    if (tensor.type() != GgufTensorType.F32) {
      throw new IllegalArgumentException(name + " must be F32 but is " + tensor.type());
    }
    long[] shape = tensor.info().shape();
    long elements = 1;
    for (long dimension : shape) {
      elements *= dimension;
    }
    if (elements != length) {
      throw new IllegalArgumentException(
          name + " holds " + elements + " elements but the metadata implies " + length);
    }
    return copy(tensor.dataSegment(), length);
  }

  /** An F32 matrix read row-major into {@code [rows][columns]}, for the per-head read gate. */
  private float[][] rows(String name, int rows, int columns) {
    float[] flat = vector(name, rows * columns);
    float[][] out = new float[rows][columns];
    for (int row = 0; row < rows; row++) {
      System.arraycopy(flat, row * columns, out[row], 0, columns);
    }
    return out;
  }

  private GgufTensorData require(String name) {
    if (!file.hasTensor(name)) {
      throw new IllegalArgumentException("missing EdgeRock tensor " + name);
    }
    return file.getTensor(name);
  }

  private static float[] copy(MemorySegment segment, int length) {
    float[] out = new float[length];
    for (int i = 0; i < length; i++) {
      out[i] = segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
    }
    return out;
  }
}
