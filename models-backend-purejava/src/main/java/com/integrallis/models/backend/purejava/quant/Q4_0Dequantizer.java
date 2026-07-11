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
 * Dequantizes Q4_0 blocks. Each block is 18 bytes: 2-byte f16 scale + 16 bytes of packed 4-bit
 * nibbles representing 32 values. Nibbles are stored unsigned (0-15), centered at 8 (subtract 8 for
 * signed range -8..7).
 */
public final class Q4_0Dequantizer implements Dequantizer {

  private static final int BLOCK_SIZE = 32;
  private static final int BLOCK_BYTES = 18;
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

      // Read 16 bytes of nibbles (32 4-bit values)
      long nibbleOffset = blockOffset + 2;
      int outIdx = dstOffset + block * BLOCK_SIZE;

      for (int i = 0; i < 16; i++) {
        byte packed = src.get(ValueLayout.JAVA_BYTE, nibbleOffset + i);
        int lo = (packed & 0x0F) - 8;
        int hi = ((packed >>> 4) & 0x0F) - 8;
        dst[outIdx + i] = lo * scale;
        dst[outIdx + i + 16] = hi * scale;
      }
    }
  }
}
