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
package com.integrallis.models.backend.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The decode-attention ablation switch must be <b>observable</b>, not merely obeyed.
 *
 * <p>Why the switch exists: the projections are bit-exact by construction, while attention carries
 * a stated 2.0e-5 relative-L2 contract because of {@code expf} (UPSTREAM.md CU-005). When G1
 * diverges at a token where the report says both stages routed, there are two live suspects and the
 * report alone cannot separate them. This switch is what lets the same binary re-run with one stage
 * withdrawn.
 *
 * <p>Why it is tested: a toggle that silently does nothing produces a measurement that reads as
 * "this stage does not matter" when the truth is "this stage never ran" — and an ablation whose
 * only evidence is an absence cannot be told apart from a stage that was never reached. So the
 * assertion here is not just that attention stops routing; it is that every diverted operation is
 * <b>counted</b>, under a reason that names the switch that caused it.
 *
 * <p>The device-gated cases below skip where there is no NVIDIA driver, which is every CI runner
 * and every laptop here. {@link #theSwitchNamesItself()} runs everywhere, because a renamed
 * property with a stale reason string would make every ablation record in the evidence trail
 * unattributable.
 */
class CudaAttentionAblationTest {

  private static final String SWITCH = CudaGgufBatchedMatrixKernel.ATTENTION_DISABLED_PROPERTY;
  private static final String REASON = CudaStage.DECODE_ATTENTION.name() + "/ablated-by-" + SWITCH;

  /** Runs the body with the ablation switch set, restoring whatever was there before. */
  private static void withAblation(Runnable body) {
    String previous = System.getProperty(SWITCH);
    System.setProperty(SWITCH, "true");
    try {
      body.run();
    } finally {
      if (previous == null) {
        System.clearProperty(SWITCH);
      } else {
        System.setProperty(SWITCH, previous);
      }
    }
  }

  private static CudaGgufBatchedMatrixKernel openOnDeviceOrSkip() {
    CudaGgufBatchedMatrixKernel.Status status = CudaGgufBatchedMatrixKernel.open();
    assumeTrue(status.accelerated(), "no CUDA device on this host: " + status.reason());
    return status.kernel().orElseThrow();
  }

  @Test
  @DisplayName("the switch names itself, so an ablation in the evidence trail is attributable")
  void theSwitchNamesItself() {
    // Not a tautology: it pins the reason string to the property name. If either is renamed
    // without the other, every "why did this fall back" line in a run report silently stops
    // pointing at the switch that caused it.
    assertTrue(
        SWITCH.startsWith("models.cuda."), "the switch must sit in the models.cuda namespace");
    assertTrue(REASON.contains(SWITCH), "the refusal reason must name the switch: " + REASON);
    assertFalse(
        SWITCH.equals(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY),
        "the per-stage ablation must not be the whole-backend kill switch");
  }

  @Test
  @DisplayName("with the switch set, attention is refused and each refusal is counted")
  void attentionIsRefusedAndCounted() {
    CudaGgufBatchedMatrixKernel kernel = openOnDeviceOrSkip();
    try {
      withAblation(
          () -> {
            long before = kernel.counters().refusals().getOrDefault(REASON, 0L);

            assertFalse(
                kernel.supportsGroupedAttention(),
                "the switch must withdraw attention from the eligible set");
            assertFalse(kernel.supportsGroupedAttention(), "and keep withdrawing it");

            Map<String, Long> refusals = kernel.counters().refusals();
            assertEquals(
                before + 2,
                refusals.getOrDefault(REASON, 0L),
                "each diverted attention operation must be counted, not silently dropped: "
                    + refusals);
          });
    } finally {
      kernel.close();
    }
  }

  @Test
  @DisplayName("with the switch clear, attention is eligible again and nothing is counted")
  void theSwitchIsReversibleAndCostsNothingWhenClear() {
    CudaGgufBatchedMatrixKernel kernel = openOnDeviceOrSkip();
    try {
      // Guards the other direction: a switch that cannot be turned back off would make the
      // ablation arm and the control arm the same run.
      assertTrue(System.getProperty(SWITCH) == null || !Boolean.getBoolean(SWITCH));
      long before = kernel.counters().refusals().getOrDefault(REASON, 0L);

      assertTrue(kernel.supportsGroupedAttention(), "attention must route when nothing refuses it");

      assertEquals(
          before,
          kernel.counters().refusals().getOrDefault(REASON, 0L),
          "an unset switch must not record an ablation");
    } finally {
      kernel.close();
    }
  }

  @Test
  @DisplayName("an ablated attention call is refused outright rather than quietly computed")
  void anAblatedAttentionCallThrowsAndIsCounted() {
    CudaGgufBatchedMatrixKernel kernel = openOnDeviceOrSkip();
    try {
      withAblation(
          () -> {
            long before = kernel.counters().refusals().getOrDefault(REASON, 0L);
            // The caller checks supportsGroupedAttention first; this is the belt-and-braces path
            // for a caller that does not. It must refuse loudly — an ablation that half-runs is
            // worse than no ablation, because the result looks like a measurement.
            assertThrows(
                UnsupportedOperationException.class,
                () -> {
                  float[] unused = new float[1];
                  kernel.groupedAttention(
                      unused, 0, unused, 0, unused, 0, 0, unused, 0, unused, 0, 0, unused, 0,
                      unused, 1, 1, 1, 1, 1, 1, 1.0f);
                });
            assertEquals(
                before + 1,
                kernel.counters().refusals().getOrDefault(REASON, 0L),
                "the refused call must still be counted");
          });
    } finally {
      kernel.close();
    }
  }
}
