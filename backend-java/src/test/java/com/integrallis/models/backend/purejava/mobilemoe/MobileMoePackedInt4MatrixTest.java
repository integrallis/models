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
package com.integrallis.models.backend.purejava.mobilemoe;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MobileMoePackedInt4MatrixTest {

  @Test
  void decodesLowNibbleFirstAndAppliesOneFp16ScalePerGroup() {
    int rows = 2;
    int columns = 32;
    byte[] packed = new byte[rows * columns / 2];
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column += 2) {
        int even = ((column + row) % 16) - 8;
        int odd = ((column + row + 1) % 16) - 8;
        packed[(row * columns + column) / 2] = pack(even, odd);
      }
    }
    MemorySegment scales = fp16(0.5f, 0.25f);
    MobileMoePackedInt4Matrix matrix =
        MobileMoePackedInt4Matrix.of(MemorySegment.ofArray(packed), scales, rows, columns, columns);

    assertThat(matrix.value(0, 0)).isEqualTo(-4.0f);
    assertThat(matrix.value(0, 1)).isEqualTo(-3.5f);
    assertThat(matrix.value(1, 0)).isEqualTo(-1.75f);
    assertThat(matrix.value(1, 15)).isEqualTo(-2.0f);

    float[] input = new float[columns];
    for (int index = 0; index < input.length; index++) {
      input[index] = (index - 8) * 0.125f;
    }
    float[] expected = reference(packed, new float[] {0.5f, 0.25f}, rows, columns, input);
    float[] actual = new float[rows];

    matrix.multiply(input, actual);

    assertThat(actual).containsExactly(expected, within(1.0e-6f));
  }

  @Test
  void rejectsMismatchedBuffersAndUnsupportedGroups() {
    assertThatThrownBy(
            () ->
                MobileMoePackedInt4Matrix.of(
                    MemorySegment.ofArray(new byte[16]), fp16(1.0f), 1, 31, 32))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("columns");
    assertThatThrownBy(
            () ->
                MobileMoePackedInt4Matrix.of(
                    MemorySegment.ofArray(new byte[16]), fp16(1.0f, 1.0f), 1, 32, 32))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("scale");
  }

  @Test
  void batchesIndependentActivationRowsWithoutChangingTheirResults() {
    int rows = 2;
    int columns = 32;
    byte[] packed = new byte[rows * columns / 2];
    for (int logical = 0; logical < rows * columns; logical += 2) {
      packed[logical / 2] = pack((logical % 16) - 8, ((logical + 1) % 16) - 8);
    }
    MobileMoePackedInt4Matrix matrix =
        MobileMoePackedInt4Matrix.of(
            MemorySegment.ofArray(packed), fp16(0.5f, 0.25f), rows, columns, columns);
    float[] input = new float[3 * columns];
    for (int index = 0; index < input.length; index++) {
      input[index] = (index - 17) * 0.03125f;
    }
    float[] actual = new float[3 * rows];

    matrix.multiplyBatch(input, 3, actual);

    for (int batch = 0; batch < 3; batch++) {
      float[] row = new float[columns];
      System.arraycopy(input, batch * columns, row, 0, columns);
      float[] expected = new float[rows];
      matrix.multiply(row, expected);
      assertThat(actual[batch * rows]).isCloseTo(expected[0], within(1.0e-6f));
      assertThat(actual[batch * rows + 1]).isCloseTo(expected[1], within(1.0e-6f));
    }
  }

  private static byte pack(int even, int odd) {
    return (byte) ((even & 0x0f) | ((odd & 0x0f) << 4));
  }

  private static MemorySegment fp16(float... values) {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putShort(Float.floatToFloat16(value));
    }
    return MemorySegment.ofArray(buffer.array());
  }

  private static float[] reference(
      byte[] packed, float[] scales, int rows, int columns, float[] input) {
    float[] output = new float[rows];
    for (int row = 0; row < rows; row++) {
      float sum = 0.0f;
      for (int column = 0; column < columns; column++) {
        int packedByte = Byte.toUnsignedInt(packed[(row * columns + column) / 2]);
        int nibble = (column & 1) == 0 ? packedByte & 0x0f : packedByte >>> 4;
        int quantized = nibble > 7 ? nibble - 16 : nibble;
        sum += quantized * scales[row] * input[column];
      }
      output[row] = sum;
    }
    return output;
  }
}
