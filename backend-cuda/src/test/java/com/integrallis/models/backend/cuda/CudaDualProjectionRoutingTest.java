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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The FFN gate and up projections must be admitted, and a declined projection must be counted.
 *
 * <p>{@code LlamaForwardPass.dualMatmulDispatch} asks {@code isDualEligible} before it will fuse
 * the gate and up projections. This kernel did not override it, so it inherited the SPI default of
 * {@code false} and both projections went to {@code TensorOps.ggufDualMatmul} on every layer of
 * every token. On Granite 4.1 3B each is 8192x2560 -- together 53.3% of a layer's projection
 * arithmetic -- and the G4 decode measurement of 1.788x was taken with all of it on the CPU.
 *
 * <p>Nothing recorded that. An unimplemented dual path is not a refusal, and the routing counters
 * had no entry for a projection that took the Java branch, so an empty refusals map read as
 * "everything ran on the device". Both halves are tested here: the admission, and the counter that
 * would have shown the omission.
 */
class CudaDualProjectionRoutingTest {

  // Granite 4.1 3B, from its GGUF header: embedding_length 2560, feed_forward_length 8192.
  private static final int DIM = 2560;
  private static final int HIDDEN = 8192;

  @Test
  @DisplayName("the SPI default would have left the gate and up projections on the CPU")
  void theSpiDefaultDeclinesDualProjections() {
    GgufBatchedMatrixKernel spiDefault =
        new GgufBatchedMatrixKernel() {
          @Override
          public boolean supports(GgufTensorType type) {
            return true;
          }

          @Override
          public void multiply(
              float[] output,
              float[] input,
              java.lang.foreign.MemorySegment weights,
              GgufTensorType type,
              int batchSize,
              int rows,
              int cols) {
            throw new UnsupportedOperationException("not used");
          }
        };
    // This is the behaviour that cost 53.3% of the arithmetic: supports() says yes to the format,
    // and the dual path still says no.
    assertTrue(spiDefault.supports(GgufTensorType.Q4_K));
    assertFalse(spiDefault.supportsDual(GgufTensorType.Q4_K, GgufTensorType.Q4_K));
    assertFalse(
        spiDefault.isDualEligible(
            GgufTensorType.Q4_K, HIDDEN, GgufTensorType.Q4_K, HIDDEN, 1, DIM));
  }

  @Test
  @DisplayName("a declined projection is counted, with its reason and shape")
  void aDeclinedProjectionIsCounted() {
    CudaRoutingCounters counters = new CudaRoutingCounters();
    assertEquals(0L, counters.totalDeclinedProjections());

    counters.declined(
        GgufTensorType.Q4_K, CudaStage.DECODE_PROJECTION, "ineligible-shape", HIDDEN, DIM);
    counters.declined(
        GgufTensorType.Q4_K, CudaStage.DECODE_PROJECTION, "ineligible-shape", HIDDEN, DIM);
    counters.declined(
        GgufTensorType.Q5_K, CudaStage.DECODE_PROJECTION, "unsupported-type", DIM, DIM);

    assertEquals(3L, counters.totalDeclinedProjections());
    Map<String, Long> declined = counters.declinedProjections();
    assertEquals(
        2L,
        declined.get("Q4_K/DECODE_PROJECTION/ineligible-shape/" + HIDDEN + "x" + DIM),
        "the shape is in the key so the offending projection is identifiable, not just its count");
    assertEquals(1L, declined.get("Q5_K/DECODE_PROJECTION/unsupported-type/" + DIM + "x" + DIM));
  }

  @Test
  @DisplayName("declines are separate from refusals, which mean an explicit ablation")
  void declinesAreNotRefusals() {
    CudaRoutingCounters counters = new CudaRoutingCounters();
    counters.declined(
        GgufTensorType.Q4_K, CudaStage.DECODE_PROJECTION, "ineligible-shape", HIDDEN, DIM);

    // The conflation is the bug being prevented: a reader of an empty refusals map concluded that
    // nothing fell back, while a decline had happened on every layer of every token.
    assertTrue(counters.refusals().isEmpty(), "a decline must not appear as a refusal");
    assertEquals(1L, counters.totalDeclinedProjections());
    assertFalse(
        counters.declinedProjections().isEmpty(),
        "and it must appear somewhere, or the fallback is invisible again");
  }

  @Test
  @DisplayName("Granite's gate and up shapes are eligible on their own merits")
  void graniteGateAndUpShapesAreEligible() {
    // Both are 8192 rows by 2560 columns. 2560 divides the 256-value super-block, and 10 blocks
    // per row is far inside MAX_BLOCKS_PER_ROW, so nothing about the shape justified declining
    // them -- which is why the omission was in the dual path and not in the shape rule.
    assertTrue(CudaGgufBatchedMatrixKernel.isShapeEligible(1, HIDDEN, DIM));
    // ffn_down is the transpose and was already routing, which is the control for that claim.
    assertTrue(CudaGgufBatchedMatrixKernel.isShapeEligible(1, DIM, HIDDEN));
  }
}
