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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.*;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class NativeDebertaKernelsTest {
  @Test
  void denseAttentionMatchesScalarWithUnevenHeadsAndReusesScratch() {
    var kernels =
        NativeDebertaKernels.load(Path.of(System.getProperty("models.native.kernels.library")), 3);
    try {
      var random = new Random(7083);
      int dim = 15, heads = 3, width = 5;
      for (int length : new int[] {1, 7, 33, 3}) {
        float[] q = new float[length * dim], k = q.clone(), v = q.clone();
        for (float[] a : new float[][] {q, k, v})
          for (int i = 0; i < a.length; i++) a[i] = random.nextFloat() * 10 - 5;
        float[] actual = new float[length * dim + 3];
        Arrays.fill(actual, 99);
        assertThat(kernels.attention(actual, q, k, v, length, dim, heads)).isTrue();
        for (int i = 0; i < length; i++) {
          for (int h = 0; h < heads; h++) {
            double[] scores = new double[length];
            for (int j = 0; j < length; j++) {
              for (int c = 0; c < width; c++)
                scores[j] += (double) q[i * dim + h * width + c] * k[j * dim + h * width + c];
              scores[j] /= Math.sqrt(width);
            }
            double max = Arrays.stream(scores).max().orElseThrow(), sum = 0;
            for (int j = 0; j < length; j++) {
              scores[j] = Math.exp(scores[j] - max);
              sum += scores[j];
            }
            for (int c = 0; c < width; c++) {
              double expected = 0;
              for (int j = 0; j < length; j++)
                expected += scores[j] / sum * v[j * dim + h * width + c];
              assertThat((double) actual[i * dim + h * width + c])
                  .isCloseTo(expected, within(2e-5));
            }
          }
        }
        for (int i = length * dim; i < actual.length; i++) assertThat(actual[i]).isEqualTo(99f);
        float[] before = actual.clone();
        assertThatThrownBy(() -> kernels.attention(actual, q, k, v, length, dim, 4))
            .isInstanceOf(IllegalArgumentException.class);
        assertThat(actual).containsExactly(before);
      }
    } finally {
      kernels.close();
    }
    assertThatThrownBy(
            () ->
                kernels.attention(new float[1], new float[1], new float[1], new float[1], 1, 1, 1))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void disentangledAttentionMatchesIndependentScalarCalculationAndRejectsBadIndices() {
    try (var kernels =
        NativeDebertaKernels.load(
            Path.of(System.getProperty("models.native.kernels.library")), 3)) {
      var random = new Random(341);
      int dim = 15, heads = 3, width = 5, relativeRows = 18;
      for (int length : new int[] {1, 7, 33, 3}) {
        float[] q = new float[length * dim], k = q.clone(), v = q.clone();
        float[] rq = new float[relativeRows * dim], rk = rq.clone();
        for (float[] a : new float[][] {q, k, v, rq, rk}) {
          for (int i = 0; i < a.length; i++) a[i] = random.nextFloat() - .5f;
        }
        int[] indices = new int[length * length];
        for (int i = 0; i < length; i++)
          for (int j = 0; j < length; j++)
            indices[i * length + j] = Math.max(0, Math.min(relativeRows - 1, i - j + 8));
        float[] actual = new float[length * dim + 3];
        Arrays.fill(actual, 99f);
        kernels.attend(actual, q, k, v, rq, rk, indices, length, dim, heads, relativeRows);
        for (int i = 0; i < length; i++)
          for (int h = 0; h < heads; h++) {
            double[] scores = new double[length];
            for (int j = 0; j < length; j++) {
              int b = indices[i * length + j];
              for (int c = 0; c < width; c++) {
                int ho = h * width + c;
                scores[j] +=
                    (double) q[i * dim + ho] * k[j * dim + ho]
                        + (double) q[i * dim + ho] * rk[b * dim + ho]
                        + (double) k[j * dim + ho] * rq[b * dim + ho];
              }
              scores[j] /= Math.sqrt(width * 3.0);
            }
            double maximum = Arrays.stream(scores).max().orElseThrow();
            double sum = 0;
            for (int j = 0; j < length; j++) {
              scores[j] = Math.exp(scores[j] - maximum);
              sum += scores[j];
            }
            for (int c = 0; c < width; c++) {
              double expected = 0;
              for (int j = 0; j < length; j++)
                expected += scores[j] / sum * v[j * dim + h * width + c];
              assertThat((double) actual[i * dim + h * width + c])
                  .isCloseTo(expected, within(1e-5));
            }
          }
        for (int i = length * dim; i < actual.length; i++) assertThat(actual[i]).isEqualTo(99f);
        float[] before = actual.clone();
        indices[0] = -1;
        assertThatThrownBy(
                () ->
                    kernels.attend(
                        actual, q, k, v, rq, rk, indices, length, dim, heads, relativeRows))
            .isInstanceOf(IllegalStateException.class);
        assertThat(actual).containsExactly(before);
      }
      assertThat(kernels.attentionCalls()).isEqualTo(4);
    }
  }

  @Test
  void projectsUnevenBatchesAndReusesBuffersWithoutWritingPastOutput() throws Exception {
    var projection =
        NativeDebertaKernels.load(Path.of(System.getProperty("models.native.kernels.library")), 3);
    try {
      var random = new Random(78131);
      for (int batch : new int[] {1, 7, 33, 3}) {
        int inputs = 513, outputs = 19;
        float[] x = new float[batch * inputs], w = new float[outputs * inputs];
        float[] y = new float[batch * outputs + 7];
        Arrays.fill(y, 99);
        for (int i = 0; i < x.length; i++) x[i] = random.nextFloat() - 0.5f;
        for (int i = 0; i < w.length; i++) w[i] = random.nextFloat() - 0.5f;
        var prepared = projection.prepare(w, outputs, inputs);
        prepared.project(y, x, batch);
        for (int row = 0; row < batch; row++)
          for (int col = 0; col < outputs; col++) {
            double expected = 0;
            for (int k = 0; k < inputs; k++)
              expected += (double) x[row * inputs + k] * w[col * inputs + k];
            assertThat((double) y[row * outputs + col])
                .isCloseTo(expected, within(1e-4 + 1e-4 * Math.abs(expected)));
          }
        for (int i = batch * outputs; i < y.length; i++) assertThat(y[i]).isEqualTo(99f);
        float[] before = y.clone();
        assertThatThrownBy(() -> prepared.project(y, new float[x.length - 1], batch))
            .isInstanceOf(IllegalArgumentException.class);
        assertThat(y).containsExactly(before);
      }
      float[] mutableWeights = {11, 13};
      var shared = projection.prepare(mutableWeights, 1, 2);
      Arrays.fill(mutableWeights, 0); // Preparation owns an immutable copy.
      try (var pool = Executors.newFixedThreadPool(4)) {
        var tasks = new ArrayList<Callable<Void>>();
        for (int i = 0; i < 16; i++)
          tasks.add(
              () -> {
                float[] out = new float[2];
                shared.project(out, new float[] {2, -3, 5, 7}, 2);
                assertThat(out).containsExactly(-17, 146);
                return null;
              });
        for (var future : pool.invokeAll(tasks)) future.get();
      }
      projection.close();
      assertThatThrownBy(() -> shared.project(new float[1], new float[2], 1))
          .isInstanceOf(IllegalStateException.class);
    } finally {
      projection.close();
    }
    projection.close();
    assertThatThrownBy(() -> projection.prepare(new float[1], 1, 1))
        .isInstanceOf(IllegalStateException.class);
  }
}
