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
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MobileMoePackedInt4RightMatrixTest {

  @Test
  void multipliesTheExpertInputByItsInputOutputTensorWithoutTransposingWeights() {
    int inputs = 3;
    int outputs = 32;
    byte[] packed = new byte[inputs * outputs / 2];
    float[] decoded = new float[inputs * outputs];
    for (int input = 0; input < inputs; input++) {
      float scale = 0.25f * (input + 1);
      for (int output = 0; output < outputs; output += 2) {
        int even = ((input + output) % 16) - 8;
        int odd = ((input + output + 1) % 16) - 8;
        packed[(input * outputs + output) / 2] = (byte) ((even & 0x0f) | ((odd & 0x0f) << 4));
        decoded[input * outputs + output] = even * scale;
        decoded[input * outputs + output + 1] = odd * scale;
      }
    }
    MobileMoePackedInt4RightMatrix matrix =
        MobileMoePackedInt4RightMatrix.of(
            MemorySegment.ofArray(packed), fp16(0.25f, 0.5f, 0.75f), inputs, outputs, 32);
    float[] activation = {0.5f, -1.25f, 0.75f};
    float[] expected = new float[outputs];
    for (int output = 0; output < outputs; output++) {
      for (int input = 0; input < inputs; input++) {
        expected[output] += activation[input] * decoded[input * outputs + output];
      }
    }
    float[] actual = new float[outputs];

    matrix.multiply(activation, actual);

    assertThat(actual).containsExactly(expected, within(1.0e-6f));
  }

  @Test
  void batchesIndependentExpertActivationRowsWithoutChangingTheirResults() {
    int inputs = 3;
    int outputs = 32;
    byte[] packed = new byte[inputs * outputs / 2];
    for (int logical = 0; logical < inputs * outputs; logical += 2) {
      packed[logical / 2] =
          (byte) (((logical % 16) - 8 & 0x0f) | ((((logical + 1) % 16) - 8 & 0x0f) << 4));
    }
    MobileMoePackedInt4RightMatrix matrix =
        MobileMoePackedInt4RightMatrix.of(
            MemorySegment.ofArray(packed), fp16(0.25f, 0.5f, 0.75f), inputs, outputs, 32);
    float[] input = {0.5f, -1.25f, 0.75f, -0.25f, 0.125f, 1.5f};
    float[] actual = new float[2 * outputs];

    matrix.multiplyBatch(input, 2, actual);

    for (int batch = 0; batch < 2; batch++) {
      float[] row = new float[inputs];
      System.arraycopy(input, batch * inputs, row, 0, inputs);
      float[] expected = new float[outputs];
      matrix.multiply(row, expected);
      for (int output = 0; output < outputs; output++) {
        assertThat(actual[batch * outputs + output]).isCloseTo(expected[output], within(1.0e-6f));
      }
    }
  }

  private static MemorySegment fp16(float... values) {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putShort(Float.floatToFloat16(value));
    }
    return MemorySegment.ofArray(buffer.array());
  }
}
