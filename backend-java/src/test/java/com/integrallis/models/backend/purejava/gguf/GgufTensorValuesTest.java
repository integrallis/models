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
package com.integrallis.models.backend.purejava.gguf;

import static org.assertj.core.api.Assertions.assertThat;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufTensorValuesTest {

  @Test
  void dequantizesOneBfloat16RowWithoutExpandingTheMatrix() {
    ByteBuffer bytes = ByteBuffer.allocate(6 * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int bits : new int[] {0x3f80, 0xc000, 0x3f00, 0x4040, 0xbf80, 0x4000}) {
      bytes.putShort((short) bits);
    }
    float[] row = new float[3];

    GgufTensorValues.dequantizeRow(
        MemorySegment.ofArray(bytes.array()), GgufTensorType.BF16, 1, 3, row);

    assertThat(row).containsExactly(3.0f, -1.0f, 2.0f);
  }
}
