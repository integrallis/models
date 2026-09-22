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

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Gate G6: if the accelerator cannot preserve physical KV prefix sharing, it must refuse rather
 * than silently lose it. Tested, not asserted.
 *
 * <p>Our published claim is that two sequences forked from one prefix point at the <em>same</em>
 * host KV rows. The host cache expresses that by handing attention two spans — the shared prefix
 * and the private suffix. A device that mirrored the cache per sequence would serve both spans
 * correctly and quietly hold two copies of the prefix, which is a silent loss of the property we
 * publish, not a visible failure.
 *
 * <p>So the attention kernel accepts a single contiguous span and refuses anything else. These
 * tests pin the refusal in both directions: it must fire for a shared-prefix shape, and it must
 * <em>not</em> fire for an ordinary unshared one, or the accelerator would be inert for every
 * sequence and G2 would catch it as such.
 */
class CudaKvSharingRefusalTest {

  private static final int NUM_HEADS = 16;
  private static final int NUM_KV_HEADS = 8;
  private static final int KEY_LENGTH = 256;

  @Test
  @DisplayName("an ordinary single-span window is accepted")
  void anOrdinarySingleSpanWindowIsAccepted() {
    assertTrue(
        CudaGgufBatchedMatrixKernel.isAttentionEligible(
            512, 0, NUM_HEADS, NUM_KV_HEADS, KEY_LENGTH),
        "an unshared sequence must be accelerated, or the accelerator is inert everywhere");
  }

  @Test
  @DisplayName("a shared KV prefix presents a second span and is refused")
  void aSharedKvPrefixIsRefused() {
    // 128 shared prefix positions plus 64 private ones: the shape a fork produces.
    assertFalse(
        CudaGgufBatchedMatrixKernel.isAttentionEligible(
            128, 64, NUM_HEADS, NUM_KV_HEADS, KEY_LENGTH),
        "a shared prefix must refuse the accelerator rather than lose physical sharing");
  }

  @Test
  @DisplayName("the refusal fires on the second span alone, not on its size")
  void theRefusalFiresOnTheSecondSpanAlone() {
    // Even one shared position is a shared prefix. A threshold would make the property depend on
    // prefix length, which is not what the claim says.
    assertFalse(
        CudaGgufBatchedMatrixKernel.isAttentionEligible(1, 1, NUM_HEADS, NUM_KV_HEADS, KEY_LENGTH));
    assertFalse(
        CudaGgufBatchedMatrixKernel.isAttentionEligible(
            4096, 1, NUM_HEADS, NUM_KV_HEADS, KEY_LENGTH));
  }

  @Test
  @DisplayName("grouped-query shapes that the kernel cannot map are refused")
  void ungroupableShapesAreRefused() {
    // num_heads must be a whole multiple of num_kv_heads: the kernel maps head -> kv head by
    // integer division and would silently read the wrong cache rows otherwise.
    assertFalse(CudaGgufBatchedMatrixKernel.isAttentionEligible(64, 0, 16, 5, KEY_LENGTH));
    assertFalse(CudaGgufBatchedMatrixKernel.isAttentionEligible(64, 0, 0, 8, KEY_LENGTH));
    assertFalse(CudaGgufBatchedMatrixKernel.isAttentionEligible(64, 0, 16, 0, KEY_LENGTH));
    assertFalse(CudaGgufBatchedMatrixKernel.isAttentionEligible(0, 0, 16, 8, KEY_LENGTH));
  }

  @Test
  @DisplayName("both Gemma 4 attention shapes are accelerable")
  void bothGemma4AttentionShapesAreAccelerable() {
    // Gemma 4 26B-A4B has 25 sliding-window layers (8 kv heads x 256) and 5 full-attention
    // layers (2 kv heads x 512). A kernel that served only one of them would accelerate five
    // sixths of the stack and report a number nobody could interpret.
    assertTrue(CudaGgufBatchedMatrixKernel.isAttentionEligible(1_024, 0, 16, 8, 256));
    assertTrue(CudaGgufBatchedMatrixKernel.isAttentionEligible(4_096, 0, 16, 2, 512));
  }
}
