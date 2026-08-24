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
package com.integrallis.models.backend.purejava.safetensors;

import java.util.Objects;

/** Header entry for one row-major, little-endian Safetensors tensor. */
public record SafetensorsTensorInfo(
    String name, SafetensorsDtype dtype, long[] shape, long dataBegin, long dataEnd) {

  public SafetensorsTensorInfo {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(dtype, "dtype");
    shape = Objects.requireNonNull(shape, "shape").clone();
  }

  @Override
  public long[] shape() {
    return shape.clone();
  }

  /** Number of logical elements, where a rank-zero tensor is one scalar. */
  public long elementCount() {
    long elements = 1;
    for (long dimension : shape) {
      elements = Math.multiplyExact(elements, dimension);
    }
    return elements;
  }

  /** Number of bytes occupied in the file data buffer. */
  public long byteCount() {
    return Math.subtractExact(dataEnd, dataBegin);
  }

  long expectedByteCount() {
    long bits = Math.multiplyExact(elementCount(), dtype.bitWidth());
    if ((bits & 7) != 0) {
      throw new MalformedSafetensorsException(
          "tensor " + name + " bit count is not byte-aligned: " + bits);
    }
    return bits / 8;
  }
}
