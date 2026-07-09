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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TensorOpsTest {

  @Nested
  static class RmsNorm {

    @Test
    void identityWeightNormalizesCorrectly() {
      float[] x = {3.0f, 4.0f};
      float[] weight = {1.0f, 1.0f};
      float[] out = new float[2];

      TensorOps.rmsNorm(out, x, weight, 2, 1e-5f);

      // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
      float rms = (float) Math.sqrt(12.5f);
      assertThat(out[0]).isCloseTo(3.0f / rms, within(1e-4f));
      assertThat(out[1]).isCloseTo(4.0f / rms, within(1e-4f));
    }

    @Test
    void weightScalesOutput() {
      float[] x = {1.0f, 1.0f};
      float[] weight = {2.0f, 3.0f};
      float[] out = new float[2];

      TensorOps.rmsNorm(out, x, weight, 2, 1e-5f);

      // rms = sqrt((1+1)/2) = 1.0
      assertThat(out[0]).isCloseTo(2.0f, within(1e-4f));
      assertThat(out[1]).isCloseTo(3.0f, within(1e-4f));
    }
  }

  @Nested
  static class Matmul {

    @Test
    void identityMatrix() {
      // 2x2 identity * [1, 2] = [1, 2]
      float[] weight = {1, 0, 0, 1};
      float[] x = {1.0f, 2.0f};
      float[] out = new float[2];

      TensorOps.matmul(out, x, weight, 2, 2);

      assertThat(out[0]).isEqualTo(1.0f);
      assertThat(out[1]).isEqualTo(2.0f);
    }

    @Test
    void known3x2Product() {
      // [[1,2],[3,4],[5,6]] * [1,2] = [5, 11, 17]
      float[] weight = {1, 2, 3, 4, 5, 6};
      float[] x = {1.0f, 2.0f};
      float[] out = new float[3];

      TensorOps.matmul(out, x, weight, 3, 2);

      assertThat(out[0]).isEqualTo(5.0f);
      assertThat(out[1]).isEqualTo(11.0f);
      assertThat(out[2]).isEqualTo(17.0f);
    }

    @Test
    void delegatesToVectorsRowMajorGemvSemantics() {
      int rows = 5;
      int cols = 7;
      float[] x = {0.25f, -1.0f, 2.5f, 0.0f, 3.0f, -0.5f, 1.25f};
      float[] weight = new float[rows * cols];
      for (int i = 0; i < weight.length; i++) {
        weight[i] = (i % 9 - 4) * 0.125f;
      }
      float[] expected = new float[rows];
      float[] actual = new float[rows];

      VectorUtil.batchDotProduct(x, weight, rows, cols, expected);
      TensorOps.matmul(actual, x, weight, rows, cols);

      for (int row = 0; row < rows; row++) {
        assertThat(actual[row]).isCloseTo(expected[row], within(1e-5f));
      }
    }
  }

  @Nested
  static class QuantizedMatmul {

    @Test
    void q4_0MatchesVectorsFusedDotSemantics() {
      float[] x = repeatingQuery(32);
      byte[] row = q4Block(0.5f);
      float[] expected = new float[1];
      float[] actual = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, row);

        expected[0] = VectorUtil.ggufQ4_0DotProduct(x, qWeight, 0, 32);
        TensorOps.quantizedMatmul(actual, x, qWeight, GgufTensorType.Q4_0, 1, 32);

        assertThat(actual[0]).isCloseTo(expected[0], within(1e-5f));
      }
    }

    @Test
    void q8_0MatchesVectorsFusedDotSemantics() {
      float[] x = repeatingQuery(32);
      byte[] row = q8Block(0.25f);
      float[] expected = new float[1];
      float[] actual = new float[1];

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, row);

        expected[0] = VectorUtil.ggufQ8_0DotProduct(x, qWeight, 0, 32);
        TensorOps.quantizedMatmul(actual, x, qWeight, GgufTensorType.Q8_0, 1, 32);

        assertThat(actual[0]).isCloseTo(expected[0], within(1e-5f));
      }
    }

    @Test
    void rejectsNonBlockAlignedQ4_0Dimensions() {
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, q4Block(1.0f));

        assertThatThrownBy(
                () ->
                    TensorOps.quantizedMatmul(
                        new float[1], repeatingQuery(31), qWeight, GgufTensorType.Q4_0, 1, 31))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("multiple of 32");
      }
    }

    @Test
    void rejectsNonBlockAlignedQ8_0Dimensions() {
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment qWeight = copy(arena, q8Block(1.0f));

        assertThatThrownBy(
                () ->
                    TensorOps.quantizedMatmul(
                        new float[1], repeatingQuery(31), qWeight, GgufTensorType.Q8_0, 1, 31))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("multiple of 32");
      }
    }

    private static float[] repeatingQuery(int length) {
      float[] out = new float[length];
      for (int i = 0; i < length; i++) {
        out[i] = (i % 7) - 3.0f;
      }
      return out;
    }

    private static byte[] q4Block(float scale) {
      byte[] block = new byte[18];
      ByteBuffer.wrap(block)
          .order(ByteOrder.LITTLE_ENDIAN)
          .putShort(0, Float.floatToFloat16(scale));
      for (int i = 0; i < 16; i++) {
        int lo = i & 0x0F;
        int hi = 15 - i;
        block[2 + i] = (byte) (lo | (hi << 4));
      }
      return block;
    }

    private static byte[] q8Block(float scale) {
      byte[] block = new byte[34];
      ByteBuffer.wrap(block)
          .order(ByteOrder.LITTLE_ENDIAN)
          .putShort(0, Float.floatToFloat16(scale));
      for (int i = 0; i < 32; i++) {
        block[2 + i] = (byte) (i - 16);
      }
      return block;
    }

    private static MemorySegment copy(Arena arena, byte[] bytes) {
      MemorySegment segment = arena.allocate(bytes.length);
      MemorySegment.copy(bytes, 0, segment, ValueLayout.JAVA_BYTE, 0, bytes.length);
      return segment;
    }
  }

  @Nested
  static class Softmax {

    @Test
    void uniformInputProducesUniform() {
      float[] x = {1.0f, 1.0f, 1.0f, 1.0f};
      TensorOps.softmax(x, 0, 4);

      for (float v : x) {
        assertThat(v).isCloseTo(0.25f, within(1e-5f));
      }
    }

    @Test
    void sumsToOne() {
      float[] x = {1.0f, 2.0f, 3.0f};
      TensorOps.softmax(x, 0, 3);

      float sum = x[0] + x[1] + x[2];
      assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }

    @Test
    void numericallyStableWithLargeValues() {
      float[] x = {1000.0f, 1001.0f, 1002.0f};
      TensorOps.softmax(x, 0, 3);

      float sum = x[0] + x[1] + x[2];
      assertThat(sum).isCloseTo(1.0f, within(1e-5f));
      // All values should be finite
      for (float v : x) {
        assertThat(v).isFinite();
        assertThat(v).isGreaterThan(0.0f);
      }
    }

    @Test
    void respectsOffset() {
      float[] x = {99.0f, 1.0f, 2.0f, 3.0f, 99.0f};
      TensorOps.softmax(x, 1, 3);

      assertThat(x[0]).isEqualTo(99.0f); // untouched
      assertThat(x[4]).isEqualTo(99.0f); // untouched
      float sum = x[1] + x[2] + x[3];
      assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }
  }

  @Nested
  static class Rope {

    @Test
    void positionZeroIsIdentity() {
      float[] q = {1.0f, 2.0f, 3.0f, 4.0f};
      float[] k = {5.0f, 6.0f, 7.0f, 8.0f};
      float[] qOrig = q.clone();
      float[] kOrig = k.clone();

      TensorOps.rope(q, k, 0, 4, 10000.0f);

      // At position 0, angle=0, cos=1, sin=0, so values unchanged
      for (int i = 0; i < 4; i++) {
        assertThat(q[i]).isCloseTo(qOrig[i], within(1e-5f));
        assertThat(k[i]).isCloseTo(kOrig[i], within(1e-5f));
      }
    }

    @Test
    void preservesNorm() {
      float[] q = {1.0f, 2.0f, 3.0f, 4.0f};
      float[] k = {5.0f, 6.0f, 7.0f, 8.0f};

      float qNormBefore = norm(q);
      float kNormBefore = norm(k);

      TensorOps.rope(q, k, 7, 4, 10000.0f);

      assertThat(norm(q)).isCloseTo(qNormBefore, within(1e-4f));
      assertThat(norm(k)).isCloseTo(kNormBefore, within(1e-4f));
    }

    private float norm(float[] v) {
      float sum = 0;
      for (float x : v) sum += x * x;
      return (float) Math.sqrt(sum);
    }
  }

  @Nested
  static class SwiGlu {

    @Test
    void knownValues() {
      // silu(x) = x * sigmoid(x)
      // For x=0: silu(0) = 0, so out = 0*up = 0
      float[] gate = {0.0f};
      float[] up = {5.0f};
      float[] out = new float[1];

      TensorOps.swiGlu(out, gate, up, 1);
      assertThat(out[0]).isCloseTo(0.0f, within(1e-5f));
    }

    @Test
    void positiveGateScalesUp() {
      // For large positive gate, silu(x) ≈ x
      float[] gate = {10.0f};
      float[] up = {2.0f};
      float[] out = new float[1];

      TensorOps.swiGlu(out, gate, up, 1);
      // silu(10) ≈ 10, so out ≈ 20
      assertThat(out[0]).isCloseTo(20.0f, within(0.01f));
    }
  }
}
