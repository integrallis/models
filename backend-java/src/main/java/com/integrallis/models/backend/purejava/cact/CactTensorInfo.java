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
package com.integrallis.models.backend.purejava.cact;

import java.util.Objects;

/** Positional tensor metadata from a `.cact` directory record. */
public record CactTensorInfo(
    int index,
    CactTensorType type,
    long[] shape,
    long offset,
    long byteSize,
    int groupSize,
    int recordBits) {

  public CactTensorInfo {
    Objects.requireNonNull(type, "type");
    shape = shape.clone();
  }

  @Override
  public long[] shape() {
    return shape.clone();
  }

  /** Returns the number of logical elements. */
  public long elementCount() {
    long elements = 1;
    for (long dimension : shape) {
      elements = Math.multiplyExact(elements, dimension);
    }
    return elements;
  }

  /** Returns the packed-code prefix size for a CQ tensor. */
  public long packedCodeBytes() {
    requireCq();
    long paddedColumns = Math.multiplyExact(Math.ceilDiv(shape[1], groupSize), groupSize);
    int packedBits = recordBits == 5 ? 2 : recordBits;
    return Math.multiplyExact(shape[0], Math.multiplyExact(paddedColumns, packedBits) / Byte.SIZE);
  }

  /** Returns the FP16 norm suffix size for a CQ tensor. */
  public long normBytes() {
    requireCq();
    long groupsPerRow = Math.ceilDiv(shape[1], groupSize);
    return Math.multiplyExact(Math.multiplyExact(shape[0], groupsPerRow), Short.BYTES);
  }

  private void requireCq() {
    if (type != CactTensorType.CQ) {
      throw new IllegalStateException("tensor " + index + " is not CQ");
    }
  }
}
