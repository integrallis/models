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
package com.integrallis.models.backend.purejava.tensor;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/** A logical row-major shape, storage description, and read-only zero-copy byte view. */
public record TensorView(String name, long[] shape, TensorStorage storage, MemorySegment data) {

  public TensorView {
    if (Objects.requireNonNull(name, "name").isBlank()) {
      throw new IllegalArgumentException("tensor name must not be blank");
    }
    shape = Objects.requireNonNull(shape, "shape").clone();
    Objects.requireNonNull(storage, "storage");
    data = Objects.requireNonNull(data, "data").asReadOnly();
    long elements = 1;
    for (long dimension : shape) {
      if (dimension < 0) {
        throw new IllegalArgumentException("tensor dimensions must not be negative");
      }
      elements = Math.multiplyExact(elements, dimension);
    }
    if (elements % storage.blockElements() != 0) {
      throw new IllegalArgumentException(
          "tensor element count is not divisible by its storage block size");
    }
    long expectedBytes =
        Math.multiplyExact(elements / storage.blockElements(), storage.blockBytes());
    if (data.byteSize() != expectedBytes) {
      throw new IllegalArgumentException(
          "tensor data requires " + expectedBytes + " bytes; got " + data.byteSize());
    }
  }

  @Override
  public long[] shape() {
    return shape.clone();
  }

  @Override
  public MemorySegment data() {
    return data.asReadOnly();
  }
}
