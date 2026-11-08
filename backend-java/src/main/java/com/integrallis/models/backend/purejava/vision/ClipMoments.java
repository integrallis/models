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

/**
 * Pairwise Welford moments, adapted from PyTorch v2.9.0's aten/src/ATen/native/cpu/moments_utils.h
 * (BSD license). Copyright holders and terms are retained in META-INF/LICENSE-PYTORCH.txt. Uses
 * eight lanes consistently, independently of the host vector width.
 */
final class ClipMoments {
  private ClipMoments() {}

  static float[] compute(float[] input, int offset, int size) {
    int vectors = size / 8;
    int chunks = (vectors + 15) / 16;
    int depth = Math.max(1, 32 - Integer.numberOfLeadingZeros(Math.max(1, chunks) - 1));
    int[] counts = new int[depth];
    float[] means = new float[depth * 8];
    float[] sums = new float[depth * 8];
    float[] chunkMean = new float[8];
    float[] chunkSum = new float[8];
    for (int chunk = 0; chunk < chunks; chunk++) {
      java.util.Arrays.fill(chunkMean, 0);
      java.util.Arrays.fill(chunkSum, 0);
      int count = Math.min(16, vectors - chunk * 16);
      for (int row = 0; row < count; row++) {
        float inverse = 1.0f / (row + 1);
        for (int lane = 0; lane < 8; lane++) {
          float value = input[offset + (chunk * 16 + row) * 8 + lane];
          float delta = value - chunkMean[lane];
          chunkMean[lane] = Math.fma(delta, inverse, chunkMean[lane]);
          chunkSum[lane] = Math.fma(delta, value - chunkMean[lane], chunkSum[lane]);
        }
      }
      merge(counts, means, sums, 0, count, chunkMean, chunkSum, 0);
      int mask = chunk + 1;
      for (int level = 1; level < depth && (mask & 1) == 0; level++, mask >>= 1) {
        merge(counts, means, sums, level, counts[level - 1], means, sums, (level - 1) * 8);
        counts[level - 1] = 0;
        java.util.Arrays.fill(means, (level - 1) * 8, level * 8, 0);
        java.util.Arrays.fill(sums, (level - 1) * 8, level * 8, 0);
      }
    }
    for (int level = 1; level < depth; level++) {
      merge(counts, means, sums, 0, counts[level], means, sums, level * 8);
    }
    int count = 0;
    float mean = 0;
    float sum = 0;
    for (int i = vectors * 8; i < size; i++) {
      float value = input[offset + i];
      float delta = value - mean;
      mean += delta / ++count;
      sum += delta * (value - mean);
    }
    for (int lane = 0; lane < 8; lane++) {
      int total = count + vectors;
      float weight = total == 0 ? 0 : (float) vectors / total;
      float delta = means[lane] - mean;
      mean = Math.fma(weight, delta, mean);
      sum += Math.fma(delta * delta * weight, count, sums[lane]);
      count = total;
    }
    return new float[] {mean, sum / size};
  }

  private static void merge(
      int[] counts,
      float[] means,
      float[] sums,
      int level,
      int addedCount,
      float[] addedMeans,
      float[] addedSums,
      int source) {
    int count = counts[level];
    int total = count + addedCount;
    float weight = total == 0 ? 0 : (float) addedCount / total;
    for (int lane = 0; lane < 8; lane++) {
      int target = level * 8 + lane;
      float delta = addedMeans[source + lane] - means[target];
      means[target] = Math.fma(weight, delta, means[target]);
      sums[target] += Math.fma(delta * delta * weight, count, addedSums[source + lane]);
    }
    counts[level] = total;
  }
}
