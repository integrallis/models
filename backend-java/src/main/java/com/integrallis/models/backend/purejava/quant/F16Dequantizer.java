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
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/** Dequantizes F16 (half-precision) data to F32 element-by-element. */
public final class F16Dequantizer implements Dequantizer {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Override
  public void dequantize(MemorySegment src, long srcOffset, float[] dst, int dstOffset, int count) {
    for (int i = 0; i < count; i++) {
      short bits = src.get(LE_SHORT, srcOffset + (long) i * 2);
      dst[dstOffset + i] = Float16.toFloat(bits);
    }
  }
}
