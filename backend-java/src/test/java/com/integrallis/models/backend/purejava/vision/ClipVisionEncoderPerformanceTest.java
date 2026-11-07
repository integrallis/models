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
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Performance characterization for ClipVisionEncoder. */
@Tag("benchmark")
class ClipVisionEncoderPerformanceTest {

  @Test
  void reportEncodeLatencyAndFlops(@TempDir Path directory) throws Exception {
    System.out.println("\n=== ClipVisionEncoder Performance ===\n");

    int imageSize = SyntheticMmproj.DEFAULT_IMAGE_SIZE;
    int patchSize = SyntheticMmproj.DEFAULT_PATCH_SIZE;
    int hiddenSize = SyntheticMmproj.DEFAULT_HIDDEN_SIZE;
    int numLayers = SyntheticMmproj.DEFAULT_NUM_LAYERS;
    int numHeads = SyntheticMmproj.DEFAULT_NUM_HEADS;
    int intermediateSize = SyntheticMmproj.DEFAULT_INTERMEDIATE_SIZE;
    int tokens = (imageSize / patchSize) * (imageSize / patchSize);

    Path mmproj = SyntheticMmproj.writeDefault(directory);
    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(mmproj, arena);
      var config = ClipVisionConfig.fromGguf(file);
      var weights = ClipVisionWeights.fromGgufFile(file, config);
      var encoder = new ClipVisionEncoder(config, weights, null);

      float[] image = new float[3 * imageSize * imageSize];
      Random rnd = new Random(42);
      for (int i = 0; i < image.length; i++) image[i] = rnd.nextFloat();

      // Warmup
      for (int i = 0; i < 3; i++) encoder.encode(image);

      // Timed
      long[] times = new long[5];
      for (int i = 0; i < 5; i++) {
        long t0 = System.nanoTime();
        float[] emb = encoder.encode(image);
        times[i] = System.nanoTime() - t0;
        assertThat(emb).hasSize(config.numTokens() * hiddenSize);
      }

      double[] ms = new double[5];
      for (int i = 0; i < 5; i++) ms[i] = times[i] / 1_000_000.0;
      double avg = 0, std = 0;
      for (double v : ms) avg += v;
      avg /= ms.length;
      for (double v : ms) std += (v - avg) * (v - avg);
      std = Math.sqrt(std / ms.length);

      // FLOPs estimate
      long patchEmbed = (long) tokens * (3L * patchSize * patchSize) * hiddenSize;
      long qkv = (long) tokens * hiddenSize * (3L * hiddenSize);
      long attn = (long) tokens * tokens * hiddenSize;
      long outProj = (long) tokens * hiddenSize * hiddenSize;
      long ffn = (long) tokens * hiddenSize * intermediateSize * 2;
      long total = patchEmbed + numLayers * (qkv + attn + outProj + ffn);
      double gflops = total / 1e9;
      double gflopsPerSec = gflops / (avg / 1000.0);

      System.out.printf("Configuration:%n");
      System.out.printf(
          "  image_size=%d, patch_size=%d, hidden_size=%d, layers=%d, heads=%d%n",
          imageSize, patchSize, hiddenSize, numLayers, numHeads);
      System.out.printf("  tokens=%d, intermediate=%d%n", tokens, intermediateSize);
      System.out.printf("Latency:  avg=%.2f ms, std=%.2f ms%n", avg, std);
      System.out.printf("Throughput: %.1f images/sec%n", 1000.0 / avg);
      System.out.printf("FLOPs: %.3f GFLOPs (%.1f GFLOP/s effective)%n", gflops, gflopsPerSec);
      System.out.printf("Per-token: %.2f us/token%n", avg * 1000.0 / tokens);
      System.out.println();
    }
    Files.deleteIfExists(mmproj);
  }
}
