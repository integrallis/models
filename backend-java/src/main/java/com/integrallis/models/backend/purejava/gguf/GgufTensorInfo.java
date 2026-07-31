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
package com.integrallis.models.backend.purejava.gguf;

import java.util.Objects;

/** Describes a single tensor's name, shape, type, and offset within the tensor data section. */
public record GgufTensorInfo(
    String name, int nDimensions, long[] shape, GgufTensorType type, long offset) {

  public GgufTensorInfo {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(shape, "shape");
    Objects.requireNonNull(type, "type");
    shape = shape.clone();
    if (nDimensions != shape.length) {
      throw new IllegalArgumentException(
          "nDimensions (" + nDimensions + ") != shape.length (" + shape.length + ")");
    }
    if (nDimensions < 1 || nDimensions > 4) {
      throw new IllegalArgumentException("nDimensions must be between 1 and 4: " + nDimensions);
    }
    for (long dimension : shape) {
      if (dimension <= 0) {
        throw new IllegalArgumentException("Tensor dimensions must be positive: " + dimension);
      }
    }
    if (offset < 0) {
      throw new IllegalArgumentException("Tensor offset must be non-negative: " + offset);
    }
  }

  @Override
  public long[] shape() {
    return shape.clone();
  }

  /** Computes the total number of elements in this tensor. */
  public long elementCount() {
    long count = 1;
    for (long dim : shape) {
      count = Math.multiplyExact(count, dim);
    }
    return count;
  }

  /** Computes the byte size of this tensor's data. */
  public long byteSize() {
    long elements = elementCount();
    if (elements % type.blockSize() != 0) {
      throw new IllegalArgumentException(
          "Tensor element count "
              + elements
              + " is not a multiple of "
              + type
              + " block size "
              + type.blockSize());
    }
    long blocks = elements / type.blockSize();
    return Math.multiplyExact(blocks, type.typeSize());
  }
}
