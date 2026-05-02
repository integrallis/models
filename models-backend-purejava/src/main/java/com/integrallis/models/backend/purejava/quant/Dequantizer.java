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

/** Interface for dequantizing quantized tensor data into float32 arrays. */
public interface Dequantizer {

  /**
   * Dequantizes {@code count} elements from {@code src} starting at {@code srcOffset} into {@code
   * dst} starting at {@code dstOffset}.
   */
  void dequantize(MemorySegment src, long srcOffset, float[] dst, int dstOffset, int count);
}
