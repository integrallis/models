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
package com.integrallis.models.backend.purejava.ops;

import static org.assertj.core.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class GgmlSwiGluTest {
  @Test
  void matchesIndependentGgmlIncludingExtremeValuesRaggedTailsAndAliases() throws Exception {
    for (int size : new int[] {7, 32, 4096, 8193}) {
      float[] gate = fixture(size, "input"), up = fixture(size, "weight");
      float[] expected = fixture(size, "expected");
      float[] input = new float[size + 2];
      input[0] = 99f;
      input[size + 1] = 77f;
      System.arraycopy(gate, 0, input, 1, size);
      float[] output = input.clone();
      GgmlSwiGlu.apply(output, 1, input, 1, up, 0, size);
      assertThat(Arrays.copyOfRange(output, 1, size + 1)).containsExactly(expected);
      assertThat(output[0]).isEqualTo(99f);
      assertThat(output[size + 1]).isEqualTo(77f);
      GgmlSwiGlu.apply(input, 1, input, 1, up, 0, size);
      assertThat(input).containsExactly(output);
    }
  }

  @Test
  void rejectsInvalidRangesAndForwardAliasesBeforeWrites() {
    float[] values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float[] original = values.clone();
    assertThatThrownBy(() -> GgmlSwiGlu.apply(values, 1, values, 0, new float[8], 0, 8))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> GgmlSwiGlu.apply(values, 1, new float[8], 0, values, 0, 8))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> GgmlSwiGlu.apply(values, 0, new float[7], 0, values, 0, 8))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> GgmlSwiGlu.apply(values, -1, values, 0, values, 0, 8))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThat(values).containsExactly(original);
    GgmlSwiGlu.apply(values, 9, values, 9, values, 9, 0);
    assertThat(values).containsExactly(original);
  }

  private static float[] fixture(int size, String part) throws Exception {
    try (var stream =
        GgmlSwiGluTest.class.getResourceAsStream("/swiglu-ggml/" + size + "-" + part + ".f32")) {
      assertThat(stream).isNotNull();
      var values =
          ByteBuffer.wrap(stream.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
      float[] result = new float[values.remaining()];
      values.get(result);
      return result;
    }
  }
}
