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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class PreparedActivationViewTest {
  @Test
  void preparedViewsMatchArrayAbiWithOffsetsAliasesAndRejectedRanges() {
    try (var kernel = RustGgufBatchedMatrixKernel.open(library())) {
      for (int size : new int[] {3, 4, 8, 32, 512, 4864}) {
        for (int alias = 0; alias < 4; alias++) {
          float[] gate = new float[size + 12];
          float[] up = new float[gate.length];
          for (int i = 0; i < gate.length; i++) {
            gate[i] = (i % 113 - 57) * .125f;
            up[i] = (i % 31 - 15) * .25f;
          }
          float[] out = alias == 1 ? gate : alias == 2 ? up : new float[gate.length];
          if (alias == 3) {
            up = gate;
            out = gate;
          }
          float[] eg = gate.clone(), eu = up == gate ? eg : up.clone();
          float[] expected = out == gate ? eg : out == up ? eu : out.clone();
          boolean accepted = kernel.trySwiGlu(expected, 4, eg, 4, eu, 4, size);
          var ov = kernel.prepareFloatArray(out);
          var gv = kernel.prepareFloatArray(gate);
          var uv = kernel.prepareFloatArray(up);
          assertThat(ov.values()).isSameAs(out);
          assertThat(gv.values()).isSameAs(gate);
          assertThat(uv.values()).isSameAs(up);
          assertThat(kernel.trySwiGlu(ov, 4, gv, 4, uv, 4, size)).isEqualTo(accepted);
          assertBits(expected, out);
          assertBits(eg, gate);
          assertBits(eu, up);
          float[] before = out.clone();
          assertThatThrownBy(() -> kernel.trySwiGlu(ov, -1, gv, 0, uv, 0, 4))
              .isInstanceOf(IndexOutOfBoundsException.class);
          assertThatThrownBy(() -> kernel.trySwiGlu(ov, 0, gv, 0, uv, 0, Integer.MAX_VALUE))
              .isInstanceOf(IndexOutOfBoundsException.class);
          assertBits(before, out);
        }
      }
    }
  }

  @Test
  void callerOwnedViewsWorkConcurrentlyAndRejectClosedProvider() throws Exception {
    var kernel = RustGgufBatchedMatrixKernel.open(library());
    float[] values = new float[8];
    var view = kernel.prepareFloatArray(values);
    try (kernel;
        var executor = Executors.newFixedThreadPool(4)) {
      var tasks = new java.util.ArrayList<java.util.concurrent.Future<?>>();
      for (int worker = 0; worker < 8; worker++) {
        tasks.add(
            executor.submit(
                () -> {
                  float[] out = new float[64], gate = new float[64], up = new float[64];
                  Arrays.fill(gate, .375f);
                  Arrays.fill(up, .75f);
                  float[] expected = new float[64];
                  assertThat(kernel.trySwiGlu(expected, 0, gate, 0, up, 0, 64)).isTrue();
                  var ov = kernel.prepareFloatArray(out);
                  var gv = kernel.prepareFloatArray(gate);
                  var uv = kernel.prepareFloatArray(up);
                  for (int i = 0; i < 100; i++) {
                    assertThat(kernel.trySwiGlu(ov, 0, gv, 0, uv, 0, 64)).isTrue();
                    assertBits(expected, out);
                  }
                }));
      }
      for (var task : tasks) task.get(30, TimeUnit.SECONDS);
    }
    assertThatThrownBy(() -> kernel.trySwiGlu(view, 0, view, 0, view, 0, 8))
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(() -> kernel.prepareFloatArray(values))
        .isInstanceOf(IllegalStateException.class);
  }

  private static void assertBits(float[] expected, float[] actual) {
    assertThat(actual.length).isEqualTo(expected.length);
    for (int i = 0; i < expected.length; i++) {
      assertThat(Float.floatToRawIntBits(actual[i]))
          .as("index %s", i)
          .isEqualTo(Float.floatToRawIntBits(expected[i]));
    }
  }

  private static Path library() {
    return Path.of(System.getProperty("models.native.kernels.library"));
  }
}
