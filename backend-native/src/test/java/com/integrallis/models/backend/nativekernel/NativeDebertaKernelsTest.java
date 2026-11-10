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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class NativeDebertaKernelsTest {
  @Test
  void orderedProjectionPreservesCancellationAndPreparedWeightOwnership() {
    var kernels =
        NativeDebertaKernels.load(Path.of(System.getProperty("models.native.kernels.library")), 3);
    var weights = new float[17 * 3];
    Arrays.fill(weights, 1f);
    var projection = kernels.prepareOrdered(weights, 17, 3);
    try {
      Arrays.fill(weights, 99f);
      float[] input = new float[7 * 3];
      for (int row = 0; row < 7; row++) {
        input[row * 3] = 16777216f;
        input[row * 3 + 1] = 1f;
        input[row * 3 + 2] = -16777216f;
      }
      float[] output = new float[7 * 17 + 1];
      output[output.length - 1] = 77f;
      projection.project(output, input, 7);
      // A reordered reduction gives 1. Sequential float32 FMA gives 0.
      assertThat(Arrays.copyOf(output, 7 * 17)).containsOnly(0f);
      assertThat(output[7 * 17]).isEqualTo(77f);
      assertThatThrownBy(() -> projection.project(output, new float[3], 7))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> kernels.prepareAffine(new float[6], new float[1], 2, 3))
          .isInstanceOf(IllegalArgumentException.class);
      assertThat(kernels.geluErf(new float[0])).isTrue();
    } finally {
      kernels.close();
    }
    assertThatThrownBy(() -> projection.project(new float[17], new float[3], 1))
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(() -> kernels.geluErf(new float[] {1}))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void finalPrefillRetainsTheFullBatchArithmeticAfterAppendingItsPrefix() {
    Path library = Path.of(System.getProperty("models.native.kernels.library"));
    try (var complete = new NativeCausalAttention(NativeDebertaKernels.load(library, 3));
        var pruned = new NativeCausalAttention(NativeDebertaKernels.load(library, 3))) {
      int batch = 64, width = 128, heads = 2, dim = width * heads, capacity = 128;
      int[] seed = {371};
      float[] q = fixtureValues(seed, batch * dim);
      float[] k = fixtureValues(seed, batch * dim);
      float[] v = fixtureValues(seed, batch * dim);
      float[] expected = new float[batch * dim];
      complete.attend(expected, q, k, v, 0, 0, batch, heads, heads, width, width, capacity, 0);
      assertThat(pruned.usesFloatRotaryRecurrence()).isTrue();
      assertThat(pruned.append(k, v, 0, 0, batch - 1, heads, heads, width, width, capacity, 0))
          .isTrue();
      float[] lastQ = Arrays.copyOfRange(q, (batch - 1) * dim, batch * dim);
      float[] lastK = Arrays.copyOfRange(k, (batch - 1) * dim, batch * dim);
      float[] lastV = Arrays.copyOfRange(v, (batch - 1) * dim, batch * dim);
      float[] actual = new float[dim];
      assertThatThrownBy(
              () ->
                  pruned.attendFinalPrefill(
                      actual, lastQ, lastK, lastV, 0, batch - 1, batch + 1, heads, heads, width,
                      width, capacity, 0))
          .isInstanceOf(IllegalArgumentException.class);
      pruned.attendFinalPrefill(
          actual, lastQ, lastK, lastV, 0, batch - 1, batch, heads, heads, width, width, capacity,
          0);
      assertThat(actual)
          .containsExactly(Arrays.copyOfRange(expected, (batch - 1) * dim, batch * dim));
      assertThat(pruned.isEligible(0, batch, 1, heads, heads, width, width, capacity, 0)).isTrue();
      assertThatThrownBy(() -> pruned.rewind(-1)).isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void causalAttentionRejectsAliasingBeforeNativeDispatch() {
    try (var kernels =
        NativeDebertaKernels.load(
            Path.of(System.getProperty("models.native.kernels.library")), 2)) {
      float[] query = {1, 2, 3, 4};
      float[] key = {4, 3, 2, 1};
      float[] value = {5, 6, 7, 8};
      for (float[] output : new float[][] {query, key, value}) {
        float[] original = output.clone();
        assertThatThrownBy(() -> kernels.causalAttention(output, query, key, value, 2, 2, 2, 1, 2))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("alias");
        assertThat(output).containsExactly(original);
      }
      float[] output = new float[4];
      kernels.causalAttention(output, query, key, value, 2, 2, 2, 1, 2);
      assertThat(output[0]).isEqualTo(value[0]);
      assertThat(output[1]).isEqualTo(value[1]);
    }
  }

  @Test
  void retainedAttentionRejectsOversizedShapesAndAliasingWithoutAdvancing() {
    try (var attention =
        new NativeCausalAttention(
            NativeDebertaKernels.load(
                Path.of(System.getProperty("models.native.kernels.library")), 2))) {
      assertThat(attention.isEligible(0, 0, 1, 65536, 65536, 65536, 65536, 2, 0)).isFalse();
      assertThat(attention.isEligible(0, 0, 1, 1, 1, 65536, 65536, 65536, 0)).isFalse();
      assertThat(attention.isEligible(0, 0, 1, 1, 1, 2, 2, Integer.MIN_VALUE, 0)).isFalse();
      float[] query = {1, 2, 3, 4};
      float[] key = {4, 3, 2, 1};
      float[] value = {5, 6, 7, 8};
      for (float[] output : new float[][] {query, key, value}) {
        assertThatThrownBy(
                () -> attention.attend(output, query, key, value, 0, 0, 2, 1, 1, 2, 2, 4, 0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("alias");
        assertThat(attention.isEligible(0, 0, 2, 1, 1, 2, 2, 4, 0)).isTrue();
      }
      attention.attend(new float[4], query, key, value, 0, 0, 2, 1, 1, 2, 2, 4, 0);
      assertThat(attention.isEligible(0, 2, 1, 1, 1, 2, 2, 4, 0)).isTrue();
    }
  }

  @Test
  void erfGeluMatchesIndependentOneDnnFixtureIncludingRaggedTail() throws IOException {
    float[] input = readFixture("/vision-arithmetic/gelu-input.f32");
    float[] expected = readFixture("/vision-arithmetic/gelu-torch.f32");
    try (var kernels =
        NativeDebertaKernels.load(
            Path.of(System.getProperty("models.native.kernels.library")), 3)) {
      assertThat(kernels.geluErf(input)).isTrue();
      assertThat(input).containsExactly(expected);
    }
  }

  @Test
  void affineProjectionMatchesIndependentTorchBlocksAndOwnsBias() throws IOException {
    int batch = 577, inputs = 1024, outputs = 16;
    int[] state = {0x12345678};
    float[] input = fixtureValues(state, batch * inputs);
    float[] weights = fixtureValues(state, outputs * inputs);
    float[] bias = fixtureValues(state, outputs);
    float[] expected = readFixture("/vision-arithmetic/affine-torch.f32");
    try (var kernels =
        NativeDebertaKernels.load(
            Path.of(System.getProperty("models.native.kernels.library")), 3)) {
      var projection = kernels.prepareAffine(weights, bias, outputs, inputs);
      Arrays.fill(weights, 99);
      Arrays.fill(bias, 99);
      float[] actual = new float[batch * outputs + 1];
      actual[actual.length - 1] = 77;
      projection.project(actual, input, batch);
      assertThat(Arrays.copyOf(actual, batch * outputs)).containsExactly(expected, within(1e-6f));
      assertThat(actual[actual.length - 1]).isEqualTo(77);
      assertThatThrownBy(() -> projection.project(new float[0], input, batch))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  private static float[] fixtureValues(int[] state, int size) {
    float[] values = new float[size];
    for (int i = 0; i < size; i++) {
      state[0] = 1664525 * state[0] + 1013904223;
      values[i] = (state[0] >>> 8) * (1f / 16777216) * 0.5f - 0.25f;
    }
    return values;
  }

  @Test
  void quickGeluMatchesUpstreamFloat32Graph() throws IOException {
    float[] values = readQuickGeluFixture("input.f32");
    float[] expected = readQuickGeluFixture("torch.f32");
    assertThat(values).hasSize(4097);
    assertThat(expected).hasSameSizeAs(values);
    try (var kernels =
        NativeDebertaKernels.load(
            Path.of(System.getProperty("models.native.kernels.library")), 3)) {
      assertThat(kernels.quickGelu(values)).isTrue();
      double squaredError = 0;
      for (int i = 0; i < values.length; i++) {
        double error = (double) values[i] - expected[i];
        squaredError += error * error;
        assertThat(Math.abs(error)).isLessThanOrEqualTo(5e-7);
      }
      assertThat(Math.sqrt(squaredError / values.length)).isLessThanOrEqualTo(2e-8);
    }
  }

  private static float[] readQuickGeluFixture(String name) throws IOException {
    return readFixture("/quick-gelu-torch/" + name);
  }

  private static float[] readFixture(String name) throws IOException {
    try (var input = NativeDebertaKernelsTest.class.getResourceAsStream(name)) {
      if (input == null) throw new IOException("Missing native oracle fixture: " + name);
      var buffer =
          ByteBuffer.wrap(input.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
      float[] values = new float[buffer.remaining()];
      buffer.get(values);
      return values;
    }
  }

  @Test
  void quickGeluMatchesDoublePrecisionOnRaggedAndParallelInputs() {
    try (var kernels =
        NativeDebertaKernels.load(
            Path.of(System.getProperty("models.native.kernels.library")), 3)) {
      for (int count : new int[] {1, 7, 33, 8193}) {
        float[] values = new float[count], expected = new float[count];
        for (int i = 0; i < count; i++) {
          values[i] = (i % 161 - 80) * .25f;
          expected[i] = (float) (values[i] / (1.0 + Math.exp(-1.702 * values[i])));
        }
        assertThat(kernels.quickGelu(values)).isTrue();
        assertThat(values).containsExactly(expected, within(2e-6f));
      }
    }
  }

  @org.junit.jupiter.params.ParameterizedTest
  @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
  void causalAttentionPreservesPrefixThroughChunksDecodeResetAndRewind(boolean appendPrefix)
      throws IOException {
    try (var attention =
        new NativeCausalAttention(
            NativeDebertaKernels.load(
                Path.of(System.getProperty("models.native.kernels.library")), 3))) {
      int dim = 15, heads = 3, width = 5, capacity = 16;
      var random = new Random(6081);
      float[] fullQ = new float[capacity * dim], fullK = fullQ.clone(), fullV = fullQ.clone();
      for (float[] a : new float[][] {fullQ, fullK, fullV})
        for (int i = 0; i < a.length; i++) a[i] = random.nextFloat() * 4 - 2;
      for (int repeat = 0; repeat < 2; repeat++) {
        attention.reset();
        int start = 0;
        for (int count : new int[] {3, 1, 7, 2}) {
          assertThat(
                  attention.isEligible(
                      0, start + 1, count, heads, heads, width, width, capacity, 0))
              .isFalse();
          float[] q = Arrays.copyOfRange(fullQ, start * dim, (start + count) * dim);
          float[] k = Arrays.copyOfRange(fullK, start * dim, (start + count) * dim);
          float[] v = Arrays.copyOfRange(fullV, start * dim, (start + count) * dim);
          if (appendPrefix && start == 0) {
            assertThat(attention.append(k, v, 0, 1, count, heads, heads, width, width, capacity, 0))
                .isFalse();
            assertThatThrownBy(
                    () ->
                        attention.append(
                            new float[0], v, 0, 0, count, heads, heads, width, width, capacity, 0))
                .isInstanceOf(IllegalArgumentException.class);
            assertThat(attention.append(k, v, 0, 0, count, heads, heads, width, width, capacity, 0))
                .isTrue();
            start += count;
            continue;
          }
          float[] output = new float[count * dim + 2];
          Arrays.fill(output, 99);
          attention.attend(
              output, q, k, v, 0, start, count, heads, heads, width, width, capacity, 0);
          float[] expected = readFixture("/vision-attention/chunk-" + start + ".f32");
          assertThat(Arrays.copyOf(output, count * dim)).containsExactly(expected, within(1e-6f));
          assertThat(output[count * dim]).isEqualTo(99);
          start += count;
        }
        attention.rewind(3);
        assertThat(attention.isEligible(0, 3, 1, heads, heads, width, width, capacity, 0)).isTrue();
        assertThat(attention.isEligible(0, 13, 1, heads, heads, width, width, capacity, 0))
            .isFalse();
      }
    }
  }

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
