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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The clamped SwiGLU, against {@code ggml_compute_forward_swiglu_clamp_f32} at ggml-org/llama.cpp
 * 3f03b1772, {@code ggml/src/ggml-cpu/ops.cpp}:
 *
 * <pre>
 *   gate = std::min(src0_p[k], limit);
 *   up   = std::clamp(src1_p[k], -limit, limit);
 *   dst  = gate / (1 + expf(-gate)) * up;
 * </pre>
 *
 * <p>The two halves are clamped differently and the name says nothing about it: the gate has no
 * lower bound, the up projection has both. A symmetric implementation passes every test whose
 * inputs happen to be positive, so the sign cases below are the point of this class.
 */
@Tag("unit")
class SwiGluClampTest {

  private static final float LIMIT = 7.0f;
  private static final float TOLERANCE = 1.0e-6f;

  /** The reference, transcribed term for term, as an independent expectation. */
  private static float reference(float gate, float up, float limit) {
    float g = Math.min(gate, limit);
    float u = Math.max(-limit, Math.min(up, limit));
    return (float) (g / (1.0 + Math.exp(-g)) * u);
  }

  private static float clamped(float gate, float up) {
    float[] out = new float[1];
    TensorOps.swiGluClamped(out, 0, new float[] {gate}, 0, new float[] {up}, 0, 1, LIMIT);
    return out[0];
  }

  @Test
  void leavesTheGateUnboundedBelow() {
    // -40 is far past -limit. Clamping it to -7 would change the answer by orders of magnitude,
    // because the sigmoid factor is what collapses: 1/(1+e^40) against 1/(1+e^7).
    float unbounded = clamped(-40.0f, 1.0f);
    float ifSymmetric = reference(-LIMIT, 1.0f, LIMIT);

    assertThat(unbounded).isEqualTo(reference(-40.0f, 1.0f, LIMIT), within());
    assertThat(unbounded).isNotEqualTo(ifSymmetric);
    assertThat(Math.abs(unbounded)).isLessThan(Math.abs(ifSymmetric) / 100.0f);
  }

  @Test
  void boundsTheUpProjectionOnBothSides() {
    assertThat(clamped(1.0f, 50.0f)).isEqualTo(clamped(1.0f, LIMIT), within());
    assertThat(clamped(1.0f, -50.0f)).isEqualTo(clamped(1.0f, -LIMIT), within());
  }

  @Test
  void capsTheGateFromAbove() {
    assertThat(clamped(50.0f, 1.0f)).isEqualTo(clamped(LIMIT, 1.0f), within());
  }

  @Test
  void agreesWithTheUnclampedKernelInsideTheLimit() {
    // Within the box the clamp is inert, so the clamped route must not perturb anything the
    // vectorized kernel already produces -- otherwise enabling it for maple would move every
    // in-range activation too.
    float[] gate = {0.5f, -0.25f, 2.0f, -3.5f, 6.9f, -6.9f, 0.0f};
    float[] up = {1.5f, 2.5f, -1.0f, 0.75f, -6.5f, 6.5f, 3.0f};

    float[] unclamped = new float[gate.length];
    TensorOps.swiGlu(unclamped, gate.clone(), up.clone(), gate.length);
    float[] viaClamp = new float[gate.length];
    TensorOps.swiGluClamped(viaClamp, 0, gate, 0, up, 0, gate.length, LIMIT);

    assertThat(viaClamp).containsExactly(unclamped, within());
  }

  @Test
  void honoursOffsetsIntoLargerBuffers() {
    float[] out = new float[5];
    float[] gate = {9.0f, 9.0f, 1.0f, 2.0f, 9.0f};
    float[] up = {9.0f, 9.0f, 9.0f, 3.0f, 50.0f};

    TensorOps.swiGluClamped(out, 1, gate, 2, up, 3, 2, LIMIT);

    assertThat(out[0]).isZero();
    assertThat(out[1]).isEqualTo(reference(1.0f, 3.0f, LIMIT), within());
    assertThat(out[2]).isEqualTo(reference(2.0f, 50.0f, LIMIT), within());
    assertThat(out[3]).isZero();
    assertThat(out[4]).isZero();
  }

  private static org.assertj.core.data.Offset<Float> within() {
    return org.assertj.core.data.Offset.offset(TOLERANCE);
  }
}
