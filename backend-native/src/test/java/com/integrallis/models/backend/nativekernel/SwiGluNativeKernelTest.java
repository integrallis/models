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

import com.integrallis.vectors.core.VectorUtil;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/** Actual provider/downcall checks against the same Java float SwiGLU contract. */
class SwiGluNativeKernelTest {
  @Test
  void actualProviderMatchesJavaForOffsetsTailsAndInPlaceAliases() {
    SplittableRandom random = new SplittableRandom(941772);
    try (var kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      for (int length : new int[] {4, 8, 12, 16, 32, 512, 14336}) {
        for (int alias = 0; alias < 4; alias++) {
          float[] gate = new float[length + 12];
          float[] up = new float[length + 12];
          for (int i = 0; i < gate.length; i++) {
            gate[i] = (float) random.nextDouble(-80, 80);
            up[i] = (float) random.nextDouble(-2, 2);
          }
          gate[4] = 0f;
          gate[5] = -0f;
          gate[6] = 80f;
          gate[7] = -80f;
          if (alias == 3) up = gate;
          float[] out = alias == 1 || alias == 3 ? gate : alias == 2 ? up : new float[gate.length];
          if (alias == 0) Arrays.fill(out, 42);
          float[] expected = out.clone();
          VectorUtil.swiGlu(expected, 4, gate.clone(), 4, up.clone(), 4, length);
          assertThat(kernel.trySwiGlu(out, 4, gate, 4, up, 4, length)).isTrue();
          equalBits(expected, out);
        }
      }
    }
  }

  @Test
  void backwardOverlapKeepsOriginalReadBeforeWriteBehavior() {
    try (var kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      for (boolean overlapGate : new boolean[] {true, false}) {
        float[] shared = new float[40];
        for (int i = 0; i < shared.length; i++) shared[i] = (i - 20) * .125f;
        float[] other = new float[40];
        Arrays.fill(other, .75f);
        float[] expected = shared.clone();
        VectorUtil.swiGlu(
            expected, 1, overlapGate ? expected : other, 4, overlapGate ? other : expected, 4, 32);
        assertThat(
                kernel.trySwiGlu(
                    shared,
                    1,
                    overlapGate ? shared : other,
                    4,
                    overlapGate ? other : shared,
                    4,
                    32))
            .isTrue();
        equalBits(expected, shared);
      }
    }
  }

  @Test
  void rejectedDomainsAndForwardAliasesLeaveEveryBufferUntouched() {
    try (var kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      for (float special :
          new float[] {
            Math.nextUp(80f),
            -Math.nextUp(80f),
            Float.NaN,
            Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY
          }) {
        float[] gate = new float[20];
        Arrays.fill(gate, .5f);
        gate[17] = special;
        float[] up = new float[20];
        Arrays.fill(up, .25f);
        float[] out = new float[20];
        Arrays.fill(out, 42);
        float[] gateBefore = gate.clone(), upBefore = up.clone(), outBefore = out.clone();
        boolean accepted = kernel.trySwiGlu(out, 2, gate, 2, up, 2, 16);
        if (System.getProperty("os.arch").equals("aarch64")) {
          assertThat(accepted).isTrue();
          float[] expected = outBefore.clone();
          VectorUtil.swiGlu(expected, 2, gateBefore, 2, upBefore, 2, 16);
          equalBits(expected, out);
          continue;
        }
        assertThat(accepted).isFalse();
        equalBits(gateBefore, gate);
        equalBits(upBefore, up);
        equalBits(outBefore, out);
        VectorUtil.swiGlu(out, 2, gate, 2, up, 2, 16);
        float[] expected = outBefore.clone();
        VectorUtil.swiGlu(expected, 2, gateBefore, 2, upBefore, 2, 16);
        equalBits(expected, out);
      }
      for (boolean overlapGate : new boolean[] {true, false}) {
        float[] shared = new float[40];
        Arrays.fill(shared, .5f);
        float[] other = new float[40];
        Arrays.fill(other, .25f);
        float[] before = shared.clone();
        assertThat(
                kernel.trySwiGlu(
                    shared,
                    4,
                    overlapGate ? shared : other,
                    1,
                    overlapGate ? other : shared,
                    1,
                    32))
            .isFalse();
        equalBits(before, shared);
      }
      float[] unchanged = {0f, -0f, 1f, -1f};
      float[] expectedTail = unchanged.clone();
      boolean tailAccepted = kernel.trySwiGlu(unchanged, 0, unchanged, 0, unchanged, 0, 3);
      if (System.getProperty("os.arch").equals("aarch64")) {
        assertThat(tailAccepted).isTrue();
        VectorUtil.swiGlu(expectedTail, 0, expectedTail, 0, expectedTail, 0, 3);
      } else {
        assertThat(tailAccepted).isFalse();
      }
      equalBits(expectedTail, unchanged);
    }
  }

  @Test
  void checksBoundsAndClosedLifetimeBeforeDowncall() {
    var kernel = RustGgufBatchedMatrixKernel.open(libraryPath());
    float[] values = new float[8];
    assertThatThrownBy(() -> kernel.trySwiGlu(values, -1, values, 0, values, 0, 4))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> kernel.trySwiGlu(values, 0, values, 5, values, 0, 4))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> kernel.trySwiGlu(values, 0, values, 0, values, 0, Integer.MAX_VALUE))
        .isInstanceOf(IndexOutOfBoundsException.class);
    kernel.close();
    assertThatThrownBy(() -> kernel.trySwiGlu(values, 0, values, 0, values, 0, 4))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void concurrentFirstUsePublishesAValidBindingAndKeepsCallerBuffersSeparate() throws Exception {
    try (var library = NativeKernelLibrary.open(libraryPath());
        var pool = Executors.newFixedThreadPool(4)) {
      CountDownLatch ready = new CountDownLatch(4);
      CountDownLatch start = new CountDownLatch(1);
      var futures = new ArrayList<Future<?>>();
      try {
        for (int worker = 0; worker < 4; worker++) {
          int seed = worker;
          futures.add(
              pool.submit(
                  () -> {
                    float[] gate = new float[64], up = new float[64], actual = new float[64];
                    SplittableRandom random = new SplittableRandom(37219 + seed);
                    for (int i = 0; i < gate.length; i++) {
                      gate[i] = (float) random.nextDouble(-80, 80);
                      up[i] = (float) random.nextDouble(-2, 2);
                    }
                    float[] expected = new float[64];
                    VectorUtil.swiGlu(expected, 0, gate, 0, up, 0, 64);
                    ready.countDown();
                    assertThat(start.await(10, TimeUnit.SECONDS)).isTrue();
                    assertThat(library.trySwiGlu(actual, 0, gate, 0, up, 0, 64)).isTrue();
                    equalBits(expected, actual);
                    return null;
                  }));
        }
        assertThat(ready.await(10, TimeUnit.SECONDS)).isTrue();
      } finally {
        start.countDown();
      }
      for (Future<?> future : futures) future.get(30, TimeUnit.SECONDS);
    }
  }

  private static void equalBits(float[] expected, float[] actual) {
    for (int i = 0; i < expected.length; i++) {
      if (Float.isNaN(expected[i])) assertThat(actual[i]).isNaN();
      else
        assertThat(Float.floatToRawIntBits(actual[i]))
            .as("index %s", i)
            .isEqualTo(Float.floatToRawIntBits(expected[i]));
    }
  }

  private static Path libraryPath() {
    return Path.of(System.getProperty("models.native.kernels.library"));
  }
}
