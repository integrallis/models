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

import static org.assertj.core.api.Assertions.assertThat;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Test;

class PtqQ8WidthContractTest {
  @Test
  void integerGroupsAndScaleFmaOrderMatchIndependentUnpacking() {
    Random random = new Random(181);
    for (int blocks : new int[] {1, 3, 17})
      for (int offset : new int[] {0, 32, 96}) {
        byte[] packed = new byte[13 + blocks * 28], input = new byte[offset + blocks * 128];
        random.nextBytes(packed);
        random.nextBytes(input);
        MemorySegment source = MemorySegment.ofArray(packed);
        float[] scales = new float[offset / 32 + blocks * 4];
        for (int i = 0; i < scales.length; i++) scales[i] = random.nextFloat();
        for (int block = 0; block < blocks; block++)
          source.set(
              ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
              13 + block * 28L + 26,
              Float.floatToFloat16(.125f));
        float[] unpacked = new float[blocks * 128];
        TernaryDequantizer.dequantizePtq1_0(source, 13, unpacked, 0, unpacked.length);
        float expected = 0;
        for (int block = 0; block < blocks; block++) {
          float blockSum = 0;
          for (int group = 0; group < 4; group++) {
            int sum = 0;
            for (int j = 0; j < 32; j++) {
              int at = block * 128 + group * 32 + j;
              sum += (int) (unpacked[at] * 8) * input[offset + at];
            }
            blockSum = Math.fma(sum, scales[offset / 32 + block * 4 + group], blockSum);
          }
          expected = Math.fma(blockSum, .125f, expected);
        }
        float actual =
            TernaryDequantizer.dotProductPtq1_0Q8_0(
                source, 13, input, scales, offset, blocks * 128);
        assertThat(Float.floatToRawIntBits(actual)).isEqualTo(Float.floatToRawIntBits(expected));
      }
  }
}
