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
package com.integrallis.models.api;

/** Data types supported for tensor storage and computation. */
public enum DType {
  F32(4),
  F16(2),
  BF16(2),
  Q8_0(1),
  Q4_0(1);

  private final int byteSize;

  DType(int byteSize) {
    this.byteSize = byteSize;
  }

  /** Returns the byte size per element (for quantized types, per element before blocking). */
  public int byteSize() {
    return byteSize;
  }
}
