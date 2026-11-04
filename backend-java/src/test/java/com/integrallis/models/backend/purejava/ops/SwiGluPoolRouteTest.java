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
package com.integrallis.models.backend.purejava.ops;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Which pool SwiGLU fanned out on, read from the route it recorded.
 *
 * <p>{@code models.purejava.swiGluSharedPool} selects between two real pools, and a switch that
 * quietly does nothing is indistinguishable from a switch whose two settings perform alike. That
 * failure mode turns "no difference measured" into a claim about the code when it is really a claim
 * about the harness, so the route is asserted rather than assumed.
 *
 * <p>Asserted from {@link TensorOps#lastSwiGluRoute()} and not from thread names. Thread names are
 * JVM-wide: any other test in the same JVM can have started either pool already, so "no thread
 * named X exists" passes or fails on which tests ran first. This test was written that way at first
 * and did exactly that -- passed locally, failed in CI.
 */
class SwiGluPoolRouteTest {

  /** Comfortably past the 1024-element fan-out threshold, and not a multiple of it. */
  private static final int SIZE = 9_216;

  @Test
  void swiGluFansOutOnThePoolTheFlagSelects() {
    boolean shared = Boolean.getBoolean("models.purejava.swiGluSharedPool");

    float[] gate = new float[SIZE];
    float[] up = new float[SIZE];
    float[] out = new float[SIZE];
    for (int index = 0; index < SIZE; index++) {
      gate[index] = (index % 17) * 0.125f - 1.0f;
      up[index] = (index % 23) * 0.0625f;
    }

    TensorOps.swiGlu(out, 0, gate, 0, up, 0, SIZE);

    assertThat(TensorOps.lastSwiGluRoute())
        .as("flag shared=%s must select the matching pool", shared)
        .isEqualTo(
            shared ? TensorOps.SwiGluRoute.SHARED_POOL : TensorOps.SwiGluRoute.DEDICATED_POOL);

    // The arithmetic is identical either way, which is the point: the flag moves the work, never
    // the result.
    for (int index = 0; index < SIZE; index++) {
      float x = gate[index];
      float expected = (x / (1.0f + (float) Math.exp(-x))) * up[index];
      assertThat(out[index]).as("element %s", index).isEqualTo(expected);
    }
  }

  @Test
  void aSizeBelowTheThresholdStaysOnTheCallingThread() {
    // The third route, and the one a reader would not think to check: below the threshold neither
    // pool is involved at all, so a test that only distinguished the two pools would call a serial
    // run a pool run.
    float[] gate = new float[64];
    float[] up = new float[64];
    float[] out = new float[64];

    TensorOps.swiGlu(out, 0, gate, 0, up, 0, 64);

    assertThat(TensorOps.lastSwiGluRoute()).isEqualTo(TensorOps.SwiGluRoute.SERIAL);
  }
}
