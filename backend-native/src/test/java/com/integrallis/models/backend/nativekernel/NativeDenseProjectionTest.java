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

class NativeDenseProjectionTest {
  @Test
  void projectsUnevenBatchesAndReusesBuffersWithoutWritingPastOutput() throws Exception {
    var projection =
        NativeDenseProjection.load(Path.of(System.getProperty("models.native.kernels.library")), 3);
    try {
      var random = new Random(78131);
      for (int batch : new int[] {1, 7, 33, 3}) {
        int inputs = 513, outputs = 19;
        float[] x = new float[batch * inputs], w = new float[outputs * inputs];
        float[] y = new float[batch * outputs + 7];
        Arrays.fill(y, 99);
        for (int i = 0; i < x.length; i++) x[i] = random.nextFloat() - 0.5f;
        for (int i = 0; i < w.length; i++) w[i] = random.nextFloat() - 0.5f;
        projection.project(y, x, batch, w, outputs, inputs);
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
        assertThatThrownBy(() -> projection.project(y, x, batch, w, outputs + 1, inputs))
            .isInstanceOf(IllegalArgumentException.class);
        assertThat(y).containsExactly(before);
      }
      try (var pool = Executors.newFixedThreadPool(4)) {
        var tasks = new ArrayList<Callable<Void>>();
        for (int i = 0; i < 16; i++)
          tasks.add(
              () -> {
                float[] out = new float[2];
                projection.project(out, new float[] {2, -3, 5, 7}, 2, new float[] {11, 13}, 1, 2);
                assertThat(out).containsExactly(-17, 146);
                return null;
              });
        for (var future : pool.invokeAll(tasks)) future.get();
      }
    } finally {
      projection.close();
    }
    projection.close();
    assertThatThrownBy(() -> projection.project(new float[1], new float[1], 1, new float[1], 1, 1))
        .isInstanceOf(IllegalStateException.class);
  }
}
