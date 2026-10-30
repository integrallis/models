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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class FlintKernelRouteTest {
  @Test
  void invokesInjectedKernelAndClosesItWithBackend(@TempDir Path directory) throws Exception {
    var model = FlintGgufRouteTest.write(directory);
    var kernel = new RecordingKernel(false, false);
    try (var actual = PureJavaBackend.load(model, kernel);
        var expected = PureJavaBackend.load(model)) {
      assertThat(actual.prefill(new int[] {3, 4, 5}, 0))
          .containsExactly(expected.prefill(new int[] {3, 4, 5}, 0));
      assertThat(kernel.singleCalls).isPositive();
      assertThat(kernel.closeCalls).isZero();
    }
    assertThat(kernel.closeCalls).isEqualTo(1);
  }

  @Test
  void groupedProjectionsPreserveIndependentSessionAndRewindResults(@TempDir Path directory)
      throws Exception {
    var model = FlintGgufRouteTest.write(directory);
    var kernel = new RecordingKernel(true, false);
    try (var actual = PureJavaBackend.load(model, kernel);
        var expected = PureJavaBackend.load(model);
        var first = actual.openSession();
        var second = actual.openSession()) {
      actual.prefill(first, new int[] {3, 4, 5, 6, 7, 8}, 0);
      actual.prefill(second, new int[] {8, 7}, 0);
      actual.rewind(first, 2);
      assertThat(actual.prefill(first, new int[] {9, 10}, 2))
          .containsExactly(expected.prefill(new int[] {3, 4, 9, 10}, 0));
      expected.reset();
      assertThat(actual.forward(second, 6, 2))
          .containsExactly(expected.prefill(new int[] {8, 7, 6}, 0));
      assertThat(kernel.dualCalls).isPositive();
      assertThat(kernel.tripleCalls).isPositive();
      assertThat(kernel.groupCalls).isPositive();
      assertThat(kernel.independentCalls).isPositive();
    }
  }

  @Test
  void respectsShapeRejectionAndFallsBackToJava(@TempDir Path directory) throws Exception {
    var model = FlintGgufRouteTest.write(directory);
    var kernel = new RecordingKernel(false, true);
    try (var actual = PureJavaBackend.load(model, kernel);
        var expected = PureJavaBackend.load(model)) {
      assertThat(actual.prefill(new int[] {3, 4, 5}, 0))
          .containsExactly(expected.prefill(new int[] {3, 4, 5}, 0));
      assertThat(kernel.singleCalls).isZero();
    }
  }

  private static final class RecordingKernel implements GgufBatchedMatrixKernel {
    final boolean grouped;
    final boolean reject;
    int singleCalls, dualCalls, tripleCalls, groupCalls, independentCalls, closeCalls;

    RecordingKernel(boolean grouped, boolean reject) {
      this.grouped = grouped;
      this.reject = reject;
    }

    @Override
    public boolean supports(GgufTensorType type) {
      return type == GgufTensorType.F32;
    }

    @Override
    public boolean isEligible(GgufTensorType type, int batch, int rows, int cols) {
      return !reject && supports(type);
    }

    @Override
    public void multiply(
        float[] out,
        float[] in,
        MemorySegment weights,
        GgufTensorType type,
        int batch,
        int rows,
        int cols) {
      assertThat(batch).isEqualTo(1);
      singleCalls++;
      TensorOps.ggufMatmul(out, in, weights, type, rows, cols);
    }

    @Override
    public boolean isDualEligible(
        GgufTensorType a, int ar, GgufTensorType b, int br, int batch, int cols) {
      return grouped;
    }

    @Override
    public void multiplyDual(
        float[] ao,
        MemorySegment aw,
        GgufTensorType a,
        int ar,
        float[] bo,
        MemorySegment bw,
        GgufTensorType b,
        int br,
        float[] in,
        int batch,
        int cols) {
      dualCalls++;
      multiply(ao, in, aw, a, batch, ar, cols);
      multiply(bo, in, bw, b, batch, br, cols);
    }

    @Override
    public boolean isTripleEligible(
        GgufTensorType a,
        int ar,
        GgufTensorType b,
        int br,
        GgufTensorType c,
        int cr,
        int batch,
        int cols) {
      return grouped;
    }

    @Override
    public void multiplyTriple(
        float[] ao,
        MemorySegment aw,
        GgufTensorType a,
        int ar,
        float[] bo,
        MemorySegment bw,
        GgufTensorType b,
        int br,
        float[] co,
        MemorySegment cw,
        GgufTensorType c,
        int cr,
        float[] in,
        int batch,
        int cols) {
      tripleCalls++;
      multiply(ao, in, aw, a, batch, ar, cols);
      multiply(bo, in, bw, b, batch, br, cols);
      multiply(co, in, cw, c, batch, cr, cols);
    }

    @Override
    public boolean isGroupedEligible(
        GgufTensorType[] types, int[] rows, int count, int batch, int cols) {
      return grouped;
    }

    @Override
    public boolean isIndependentEligible(
        GgufTensorType[] types, int[] rows, int count, int batch, int cols) {
      return grouped;
    }

    @Override
    public void multiplyGrouped(
        float[][] outputs,
        MemorySegment[] weights,
        GgufTensorType[] types,
        int[] rows,
        int count,
        float[] input,
        int batch,
        int cols) {
      groupCalls++;
      for (int i = 0; i < count; i++)
        multiply(outputs[i], input, weights[i], types[i], batch, rows[i], cols);
    }

    @Override
    public void multiplyIndependent(
        float[][] outputs,
        MemorySegment[] weights,
        GgufTensorType[] types,
        int[] rows,
        int count,
        float[][] inputs,
        int batch,
        int cols) {
      independentCalls++;
      for (int i = 0; i < count; i++)
        multiply(outputs[i], inputs[i], weights[i], types[i], batch, rows[i], cols);
    }

    @Override
    public void close() {
      closeCalls++;
    }
  }
}
