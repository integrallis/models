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
package com.integrallis.models.backend.purejava.bert;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.util.EnumSet;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Which tensors #360 widens to F32 at load, and which keep the form they were stored in.
 *
 * <p>The trade is 4x weight memory for per-token latency, so every case it covers without paying
 * back is pure cost. Two did (#452): the embedding tables, which are row lookups rather than
 * multiplications, and projections on a path that has a kernel able to take the quantized form,
 * where widening decides the route invisibly at load instead of leaving it to {@code
 * BertForwardPass.project}. Both are asserted here against the whole tensor-type enum rather than a
 * sample, because the defect was a condition that named types and forgot everything else about the
 * context.
 */
@Tag("unit")
class BertWeightsPreDequantizationTest {

  private static final Set<GgufTensorType> K_QUANTS =
      EnumSet.of(GgufTensorType.Q4_K, GgufTensorType.Q5_K, GgufTensorType.Q6_K);

  /** A kernel that takes every type, standing for the native provider the bert families install. */
  private static GgufBatchedMatrixKernel acceptsEverything() {
    return new GgufBatchedMatrixKernel() {
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
          int columns) {
        throw new UnsupportedOperationException("not executed by this test");
      }
    };
  }

  @Test
  void isOffUnlessOptedIn() {
    // The default. Widening changes a qualified reranker's scores and slows cold load, so it is a
    // choice a caller makes, never one made for them. Every other case below passes enabled = true.
    assertThat(Boolean.getBoolean(BertWeights.PRE_DEQUANTIZE_K_QUANT_PROPERTY)).isFalse();
    for (GgufTensorType type : GgufTensorType.values()) {
      assertThat(BertWeights.preDequantizes(false, true, type, GgufBatchedMatrixKernel.none()))
          .describedAs("disabled, projection, no kernel, %s", type)
          .isFalse();
    }
  }

  @Test
  void widensKQuantProjectionsOnlyWhenNoKernelTakesTheQuantizedForm() {
    for (GgufTensorType type : GgufTensorType.values()) {
      assertThat(BertWeights.preDequantizes(true, true, type, GgufBatchedMatrixKernel.none()))
          .describedAs("projection, no kernel, %s", type)
          .isEqualTo(K_QUANTS.contains(type));
    }
  }

  @Test
  void neverWidensWhenAKernelTakesTheQuantizedForm() {
    GgufBatchedMatrixKernel kernel = acceptsEverything();
    for (GgufTensorType type : GgufTensorType.values()) {
      assertThat(BertWeights.preDequantizes(true, true, type, kernel))
          .describedAs("projection, kernel accepts %s, so project() must choose the route", type)
          .isFalse();
    }
  }

  @Test
  void neverWidensATable() {
    // token_embd and position_embd reach matrix() with projection = false. They are the largest
    // tensors in the model and are read a row at a time, so widening them is cost without benefit.
    for (GgufTensorType type : GgufTensorType.values()) {
      assertThat(BertWeights.preDequantizes(true, false, type, GgufBatchedMatrixKernel.none()))
          .describedAs("table, no kernel, %s", type)
          .isFalse();
      assertThat(BertWeights.preDequantizes(true, false, type, acceptsEverything()))
          .describedAs("table, kernel accepts %s", type)
          .isFalse();
    }
  }

  @Test
  void leavesEveryNonKQuantFormAlone() {
    for (GgufTensorType type : GgufTensorType.values()) {
      if (K_QUANTS.contains(type)) {
        continue;
      }
      assertThat(BertWeights.preDequantizes(true, true, type, GgufBatchedMatrixKernel.none()))
          .describedAs("projection, no kernel, non-K-quant %s", type)
          .isFalse();
    }
  }
}
