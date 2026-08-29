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
package com.integrallis.models.accelerator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

class Q8ProjectionKernelTest {

  @Test
  void matchesVectorApiDequantizedProjectionAcrossBatches() {
    int batchSize = 3;
    int rows = 17;
    int cols = 96;
    byte[] weights = randomQ8Matrix(rows, cols, 23L);
    float[] input = randomFloats(batchSize * cols, 29L);
    float[] expected = vectorApiProjection(weights, input, batchSize, rows, cols);
    ByteArray deviceWeights = ByteArray.fromArray(weights);
    FloatArray deviceInput = FloatArray.fromArray(input);
    FloatArray deviceOutput = new FloatArray(batchSize * rows);

    Q8ProjectionKernel.multiply(deviceWeights, deviceInput, deviceOutput, batchSize, rows, cols);

    float[] actual = deviceOutput.toHeapArray();
    assertThat(actual).hasSameSizeAs(expected);
    for (int index = 0; index < expected.length; index++) {
      assertThat(actual[index]).isCloseTo(expected[index], within(2.0e-5f));
    }
  }

  @Test
  void rejectsColumnsThatAreNotQ8BlockAligned() {
    assertThatThrownBy(
            () ->
                Q8ProjectionKernel.validate(
                    new ByteArray(34), new FloatArray(31), new FloatArray(1), 1, 1, 31))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("multiple of 32");
  }

  private static float[] vectorApiProjection(
      byte[] weights, float[] input, int batchSize, int rows, int cols) {
    float[] output = new float[batchSize * rows];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);
    for (int batch = 0; batch < batchSize; batch++) {
      float[] query = new float[cols];
      float[] projected = new float[rows];
      System.arraycopy(input, batch * cols, query, 0, cols);
      VectorUtil.ggufQ8_0BatchDotProduct(query, weightSegment, rows, cols, projected);
      System.arraycopy(projected, 0, output, batch * rows, rows);
    }
    return output;
  }

  private static byte[] randomQ8Matrix(int rows, int cols, long seed) {
    Random random = new Random(seed);
    int blocks = rows * cols / 32;
    byte[] weights = new byte[blocks * 34];
    for (int block = 0; block < blocks; block++) {
      short scale = Float.floatToFloat16(0.001f + random.nextFloat() * 0.05f);
      int offset = block * 34;
      weights[offset] = (byte) scale;
      weights[offset + 1] = (byte) (scale >>> 8);
      for (int quant = 0; quant < 32; quant++) {
        weights[offset + 2 + quant] = (byte) (random.nextInt(255) - 127);
      }
    }
    return weights;
  }

  private static float[] randomFloats(int length, long seed) {
    Random random = new Random(seed);
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = random.nextFloat(-2.0f, 2.0f);
    }
    return values;
  }
}
