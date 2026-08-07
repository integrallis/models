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
package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.Optional;

/**
 * The pair of linear layers a sentence-transformer applies to its pooled vector.
 *
 * <p>Present only in GGUFs converted with {@code --sentence-transformers-dense-modules}. They are
 * part of the model, not an optional refinement: the published EmbeddingGemma vectors are the
 * output of this head, so skipping it yields a vector of the right width and the wrong direction —
 * the failure that retrieval quietly absorbs instead of reporting.
 *
 * <p>Two plain matrix multiplications, widening to {@code dense_2_feat_out} and back. No activation
 * between them and no bias tensors, matching llama.cpp's {@code build_dense_out}.
 */
public final class DenseProjectionHead {

  private final MemorySegment expandWeight;
  private final GgufTensorType expandType;
  private final MemorySegment contractWeight;
  private final GgufTensorType contractType;
  private final int inputDim;
  private final int innerDim;
  private final float[] expanded;

  private DenseProjectionHead(
      MemorySegment expandWeight,
      GgufTensorType expandType,
      MemorySegment contractWeight,
      GgufTensorType contractType,
      int inputDim,
      int innerDim) {
    this.expandWeight = expandWeight;
    this.expandType = expandType;
    this.contractWeight = contractWeight;
    this.contractType = contractType;
    this.inputDim = inputDim;
    this.innerDim = innerDim;
    this.expanded = new float[innerDim];
  }

  /**
   * Loads the head if the model carries one.
   *
   * <p>Both tensors must be present together. Applying only the first would change the vector's
   * width, and applying only the second is not a projection the model was ever trained with, so a
   * half-present head is rejected rather than partially honoured.
   *
   * @param file the parsed model
   * @param architecture the GGUF {@code general.architecture} value, which prefixes its keys
   * @param embeddingDim the model's hidden width, which the head must both accept and return
   * @return the head, or empty when this GGUF was converted without the dense modules
   */
  public static Optional<DenseProjectionHead> load(
      GgufFile file, String architecture, int embeddingDim) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(architecture, "architecture");
    GgufTensorData expand = optionalTensor(file, "dense_2.weight");
    GgufTensorData contract = optionalTensor(file, "dense_3.weight");
    if (expand == null && contract == null) {
      return Optional.empty();
    }
    if (expand == null || contract == null) {
      throw new IllegalArgumentException(
          "model declares only one of dense_2.weight and dense_3.weight; a dense projection head"
              + " must be complete to be applied");
    }

    GgufMetadata metadata = file.metadata();
    int featIn = requireDim(metadata, architecture, "dense_2_feat_in");
    int innerDim = requireDim(metadata, architecture, "dense_2_feat_out");
    int featOut = requireDim(metadata, architecture, "dense_3_feat_out");
    if (featIn != embeddingDim || featOut != embeddingDim) {
      throw new IllegalArgumentException(
          "dense projection head must accept and return "
              + embeddingDim
              + " dimensions, but takes "
              + featIn
              + " and returns "
              + featOut);
    }
    if (requireDim(metadata, architecture, "dense_3_feat_in") != innerDim) {
      throw new IllegalArgumentException(
          "dense_3_feat_in must match dense_2_feat_out (" + innerDim + ")");
    }

    return Optional.of(
        new DenseProjectionHead(
            expand.dataSegment(),
            expand.type(),
            contract.dataSegment(),
            contract.type(),
            embeddingDim,
            innerDim));
  }

  /** Width this head accepts and returns. */
  public int dimension() {
    return inputDim;
  }

  /**
   * Projects a pooled vector in place.
   *
   * @param pooled the pooled sentence vector, overwritten with the projection
   */
  public void project(float[] pooled) {
    Objects.requireNonNull(pooled, "pooled");
    if (pooled.length != inputDim) {
      throw new IllegalArgumentException(
          "pooled vector must have " + inputDim + " dimensions, got " + pooled.length);
    }
    TensorOps.ggufMatmul(expanded, pooled, expandWeight, expandType, innerDim, inputDim);
    TensorOps.ggufMatmul(pooled, expanded, contractWeight, contractType, inputDim, innerDim);
  }

  private static GgufTensorData optionalTensor(GgufFile file, String name) {
    try {
      return file.getTensor(name);
    } catch (IllegalArgumentException e) {
      return null;
    }
  }

  private static int requireDim(GgufMetadata metadata, String architecture, String key) {
    return metadata
        .getUint32(architecture + "." + key)
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    "model carries dense projection tensors but no "
                        + architecture
                        + "."
                        + key
                        + " to size them"));
  }
}
