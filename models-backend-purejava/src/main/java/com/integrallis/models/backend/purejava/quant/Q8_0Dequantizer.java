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

/**
 * Dequantizes Q8_0 blocks. Each block is 34 bytes: 2-byte f16 scale + 32 signed int8 quantized
 * values.
 */
public final class Q8_0Dequantizer implements Dequantizer {

  private static final int BLOCK_SIZE = 32;
  private static final int BLOCK_BYTES = 34;
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Override
  public void dequantize(MemorySegment src, long srcOffset, float[] dst, int dstOffset, int count) {
    int numBlocks = count / BLOCK_SIZE;

    for (int block = 0; block < numBlocks; block++) {
      long blockOffset = srcOffset + (long) block * BLOCK_BYTES;

      // Read f16 scale
      short scaleBits = src.get(LE_SHORT, blockOffset);
      float scale = Float16.toFloat(scaleBits);

      // Read 32 signed int8 quantized values
      long quantOffset = blockOffset + 2;
      int outIdx = dstOffset + block * BLOCK_SIZE;

      for (int i = 0; i < BLOCK_SIZE; i++) {
        byte quant = src.get(ValueLayout.JAVA_BYTE, quantOffset + i);
        dst[outIdx + i] = quant * scale;
      }
    }
  }
}
