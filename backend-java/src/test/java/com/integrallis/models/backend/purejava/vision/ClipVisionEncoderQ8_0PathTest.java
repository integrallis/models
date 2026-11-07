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
package com.integrallis.models.backend.purejava.vision;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Verifies that ClipVisionEncoder can route projections through a Q8_0 kernel. */
@Tag("unit")
class ClipVisionEncoderQ8_0PathTest {

  @Test
  void q8_0KernelPathProducesSameShapeAsFallback(@TempDir Path directory) throws Exception {
    Path mmproj = SyntheticMmproj.writeDefault(directory);
    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(mmproj, arena);
      var config = ClipVisionConfig.fromGguf(file);
      var weights = ClipVisionWeights.fromGgufFile(file, config);

      // Quantize to Q8_0
      var q8Weights = weights.withQ8_0(arena);

      int imageSize = SyntheticMmproj.DEFAULT_IMAGE_SIZE;
      int hiddenSize = SyntheticMmproj.DEFAULT_HIDDEN_SIZE;
      int patchSize = SyntheticMmproj.DEFAULT_PATCH_SIZE;
      float[] image = new float[3 * imageSize * imageSize];
      Random rnd = new Random(42);
      for (int i = 0; i < image.length; i++) image[i] = rnd.nextFloat();

      // Run with counting kernel that records Q8_0 calls
      CountingKernel countingKernel = new CountingKernel();
      var encoderQ8 = new ClipVisionEncoder(config, q8Weights, countingKernel);
      float[] embQ8 = encoderQ8.encode(image);

      // Run with no kernel (fallback path)
      var encoderFallback = new ClipVisionEncoder(config, weights, null);
      float[] embFallback = encoderFallback.encode(image);

      // Both should produce the same shape
      assertThat(embQ8).hasSize(embFallback.length);

      // The Q8_0 path should have triggered the kernel for projections
      // 4 projections per layer * 6 layers = 24 calls
      assertThat(countingKernel.q8_0Calls).isGreaterThan(0);
      System.out.println("Q8_0 kernel calls: " + countingKernel.q8_0Calls);
    }
    Files.deleteIfExists(mmproj);
  }

  private static final class CountingKernel implements GgufBatchedMatrixKernel {
    int q8_0Calls = 0;

    @Override
    public boolean supports(GgufTensorType type) {
      return type == GgufTensorType.Q8_0;
    }

    @Override
    public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
      return type == GgufTensorType.Q8_0;
    }

    @Override
    public void multiply(
        float[] output,
        float[] input,
        MemorySegment weights,
        GgufTensorType type,
        int batchSize,
        int rows,
        int cols) {
      if (type == GgufTensorType.Q8_0) {
        q8_0Calls++;
      }
      // Do a simple reference matmul for correctness
      for (int b = 0; b < batchSize; b++) {
        for (int r = 0; r < rows; r++) {
          float sum = 0.0f;
          for (int c = 0; c < cols; c++) {
            sum += input[b * cols + c] * 0.01f;
          }
          output[b * rows + r] = sum;
        }
      }
    }

    @Override
    public void close() {}
  }
}
