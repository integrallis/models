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

import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Benchmark comparing pure-Java F32 vs Rust-accelerated Q8_0 vision encoding. */
@Tag("benchmark")
class ClipVisionEncoderNativeKernelBenchmark {

  @Test
  void compareF32JavaVsQ8_0Rust(@TempDir Path directory) throws Exception {
    String libraryPath = System.getProperty("models.native.kernels.library");
    boolean nativeAvailable = libraryPath != null && !libraryPath.isBlank();
    Path nativeSo = nativeAvailable ? Path.of(libraryPath) : null;

    Path mmproj = SyntheticMmproj.writeDefault(directory);
    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(mmproj, arena);
      var config = ClipVisionConfig.fromGguf(file);
      var weights = ClipVisionWeights.fromGgufFile(file, config);
      var q8Weights = weights.withQ8_0(arena);

      int imageSize = SyntheticMmproj.DEFAULT_IMAGE_SIZE;
      int hiddenSize = SyntheticMmproj.DEFAULT_HIDDEN_SIZE;
      int patchSize = SyntheticMmproj.DEFAULT_PATCH_SIZE;
      float[] image = new float[3 * imageSize * imageSize];
      Random rnd = new Random(42);
      for (int i = 0; i < image.length; i++) image[i] = rnd.nextFloat();

      int iterations = 10;
      int warmup = 5;

      // Pure-Java F32
      var encoderJava = new ClipVisionEncoder(config, weights, null);
      for (int i = 0; i < warmup; i++) encoderJava.encode(image);
      double[] javaTimes = new double[iterations];
      for (int i = 0; i < iterations; i++) {
        long t0 = System.nanoTime();
        float[] emb = encoderJava.encode(image);
        javaTimes[i] = (System.nanoTime() - t0) / 1_000_000.0;
        assertThat(emb).hasSize(config.numTokens() * hiddenSize);
      }

      System.out.println("\n=== ClipVisionEncoder Native Kernel Benchmark ===\n");
      System.out.printf(
          "Config: image=%dx%d patch=%d hidden=%d layers=%d heads=%d intermediate=%d%n",
          imageSize,
          imageSize,
          patchSize,
          hiddenSize,
          SyntheticMmproj.DEFAULT_NUM_LAYERS,
          SyntheticMmproj.DEFAULT_NUM_HEADS,
          SyntheticMmproj.DEFAULT_INTERMEDIATE_SIZE);
      System.out.printf("Native kernel available: %s%n", nativeAvailable);
      printStats("Pure-Java F32", javaTimes);

      if (nativeAvailable) {
        try (RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(nativeSo)) {
          var encoderRust = new ClipVisionEncoder(config, q8Weights, kernel);
          for (int i = 0; i < warmup; i++) encoderRust.encode(image);

          double[] rustTimes = new double[iterations];
          for (int i = 0; i < iterations; i++) {
            long t0 = System.nanoTime();
            float[] emb = encoderRust.encode(image);
            rustTimes[i] = (System.nanoTime() - t0) / 1_000_000.0;
            assertThat(emb).hasSize(config.numTokens() * hiddenSize);
          }
          printStats("Rust Q8_0  ", rustTimes);
          double javaMed = median(javaTimes);
          double rustMed = median(rustTimes);
          System.out.printf("Speedup: %.2fx (median)%n", javaMed / rustMed);
        }
      } else {
        System.out.println("Rust kernel not available — skipping native benchmark");
      }
      System.out.println();
    }
    Files.deleteIfExists(mmproj);
  }

  private static void printStats(String label, double[] times) {
    double avg = 0, min = Double.MAX_VALUE, max = 0;
    for (double t : times) {
      avg += t;
      if (t < min) min = t;
      if (t > max) max = t;
    }
    avg /= times.length;
    double var = 0;
    for (double t : times) var += (t - avg) * (t - avg);
    double std = Math.sqrt(var / times.length);
    System.out.printf(
        "%s: avg=%6.2f ms  min=%6.2f ms  max=%6.2f ms  std=%5.2f ms  (%.1f img/s)%n",
        label, avg, min, max, std, 1000.0 / avg);
  }

  private static double median(double[] values) {
    double[] sorted = values.clone();
    java.util.Arrays.sort(sorted);
    return sorted[sorted.length / 2];
  }
}
