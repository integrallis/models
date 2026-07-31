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
package com.integrallis.models.backend.purejava.quant;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class DequantizationChecks {

  private DequantizationChecks() {}

  static void validate(
      MemorySegment src,
      long srcOffset,
      float[] dst,
      int dstOffset,
      int count,
      int blockSize,
      int blockBytes) {
    Objects.requireNonNull(src, "src");
    Objects.requireNonNull(dst, "dst");
    if (count < 0) {
      throw new IllegalArgumentException("count must be non-negative: " + count);
    }
    if (count % blockSize != 0) {
      throw new IllegalArgumentException("count must be a multiple of " + blockSize + ": " + count);
    }
    Objects.checkFromIndexSize(dstOffset, count, dst.length);
    if (srcOffset < 0) {
      throw new IllegalArgumentException("srcOffset must be non-negative: " + srcOffset);
    }
    long requiredBytes;
    try {
      requiredBytes = Math.multiplyExact((long) count / blockSize, blockBytes);
    } catch (ArithmeticException overflow) {
      throw new IllegalArgumentException("source byte count overflow", overflow);
    }
    if (srcOffset > src.byteSize() || requiredBytes > src.byteSize() - srcOffset) {
      throw new IllegalArgumentException(
          "source range exceeds segment: offset="
              + srcOffset
              + ", bytes="
              + requiredBytes
              + ", segment="
              + src.byteSize());
    }
  }
}
