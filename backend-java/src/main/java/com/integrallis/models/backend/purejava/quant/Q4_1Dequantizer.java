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
 * Dequantizes GGUF {@code Q4_1}: four-bit quants with a per-block scale <em>and</em> minimum.
 *
 * <p>Transcribed from {@code block_q4_1} in llama.cpp {@code ggml/src/ggml-common.h:205-212} and
 * {@code dequantize_row_q4_1} in {@code ggml/src/ggml-quants.c}: {@code d} as f16, {@code m} as
 * f16, then sixteen nibble bytes. Twenty bytes for thirty-two weights.
 *
 * <p>The value is {@code q * d + m} with {@code q} unsigned in 0..15. That is the one thing to get
 * right and the one thing that fails silently: {@code Q4_0} centres its quants by subtracting
 * eight, and applying that bias here would shift every weight in the block by {@code -8 * d} while
 * still producing plausible text. The low nibble is element {@code j} and the high nibble is
 * element {@code j + 16}, matching the reference's split rather than interleaving pairs.
 *
 * <p>Needed because several published artifacts carry a handful of {@code Q4_1} tensors inside an
 * otherwise {@code Q4_0} or K-quant build -- {@code qwen2.5-math-1.5b Q4_0}, {@code phi-4-mini
 * Q4_0}, {@code fin-r1-7b Q4_0} and {@code huatuogpt-o1-7b Q4_0} each carry three or four, around
 * 3% of their weights -- and without this they fail at the first token with "GGUF matmul not
 * supported for: Q4_1". Four models across three capability gaps, blocked by three tensors each.
 */
public final class Q4_1Dequantizer implements Dequantizer {
  /** Weights per block, which a row length must be a multiple of. */
  public static final int BLOCK_SIZE = 32;

  /** Bytes per block: two f16 headers then sixteen nibble bytes. */
  public static final int BLOCK_BYTES = 20;

  private static final int HALF_BLOCK = BLOCK_SIZE / 2;
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Override
  public void dequantize(MemorySegment src, long srcOffset, float[] dst, int dstOffset, int count) {
    DequantizationChecks.validate(src, srcOffset, dst, dstOffset, count, BLOCK_SIZE, BLOCK_BYTES);
    int numBlocks = count / BLOCK_SIZE;
    for (int block = 0; block < numBlocks; block++) {
      long blockOffset = srcOffset + (long) block * BLOCK_BYTES;
      float scale = Float16.toFloat(src.get(LE_SHORT, blockOffset));
      float minimum = Float16.toFloat(src.get(LE_SHORT, blockOffset + 2));
      long nibbleOffset = blockOffset + 4;
      int outIdx = dstOffset + block * BLOCK_SIZE;
      for (int i = 0; i < HALF_BLOCK; i++) {
        int packed = Byte.toUnsignedInt(src.get(ValueLayout.JAVA_BYTE, nibbleOffset + i));
        int low = packed & 0x0F;
        int high = packed >>> 4;
        dst[outIdx + i] = low * scale + minimum;
        dst[outIdx + i + HALF_BLOCK] = high * scale + minimum;
      }
    }
  }
}
