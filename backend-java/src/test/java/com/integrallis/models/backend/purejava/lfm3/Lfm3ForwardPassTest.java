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
package com.integrallis.models.backend.purejava.lfm3;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.backend.purejava.gguf.*;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Lfm3ForwardPassTest {
  static Lfm3ForwardPass pass(GgufBatchedMatrixKernel kernel) {
    var f = Lfm3ToyModel.file(30, 40);
    return Lfm3ForwardPass.fromGgufFile(f, Lfm3Config.fromMetadata(f.metadata()), 64, kernel);
  }

  @Test
  void chunkedPrefillMatchesSerialAcrossConvolutionAndRingBoundaries() {
    int[] tokens = java.util.stream.IntStream.range(0, 40).map(i -> i % 8).toArray();
    var serial = pass(null);
    var batch = pass(null);
    float[] expected = null;
    for (int i = 0; i < tokens.length; i++) expected = serial.forward(tokens[i], i).clone();
    assertThat(batch.prefill(tokens, 0)).containsExactly(expected);
    assertThat(batch.nextPosition()).isEqualTo(40);
    assertThat(batch.forward(3, 40)).containsExactly(serial.forward(3, 40));
  }

  @Test
  void nativeProjectionBoundaryReceivesBatchedWorkAndPreservesLogits() {
    AtomicInteger batched = new AtomicInteger();
    var kernel =
        new GgufBatchedMatrixKernel() {
          public boolean supports(GgufTensorType type) {
            return type == GgufTensorType.F32;
          }

          public void multiply(
              float[] out,
              float[] in,
              MemorySegment w,
              GgufTensorType type,
              int n,
              int rows,
              int cols) {
            if (n > 1) batched.incrementAndGet();
            TensorOps.ggufExactBatchedMatmul(out, in, w, type, n, rows, cols, new float[cols]);
          }
        };
    assertThat(pass(kernel).prefill(new int[] {0, 1, 2, 3, 4, 5, 6, 7}, 0))
        .containsExactly(pass(null).prefill(new int[] {0, 1, 2, 3, 4, 5, 6, 7}, 0));
    assertThat(batched.get()).isGreaterThan(0);
  }

  @Test
  void resetClearsConvolutionAndAttentionHistory() {
    var p = pass(null);
    float[] expected = p.prefill(new int[] {1, 2, 3, 4}, 0).clone();
    p.prefill(new int[] {5, 6, 7, 1, 2, 3, 4}, 4);
    p.reset();
    assertThat(p.nextPosition()).isZero();
    assertThat(p.prefill(new int[] {1, 2, 3, 4}, 0)).containsExactly(expected);
  }

  @Test
  void rejectsInvalidInputsBeforeMutatingState() {
    var p = pass(null);
    assertThatThrownBy(() -> p.prefill(new int[] {1, 99}, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(p.nextPosition()).isZero();
    assertThat(p.forward(1, 0)).containsExactly(pass(null).forward(1, 0));
  }

  @Test
  void configRetainsThreeDistinctOperators() {
    var c = Lfm3Config.fromMetadata(Lfm3ToyModel.file(30, 40).metadata());
    assertThat(c.usesAttention(0)).isFalse();
    assertThat(c.usesSlidingWindow(1)).isTrue();
    assertThat(c.usesSlidingWindow(2)).isFalse();
    assertThat(c.shortConvCache()).isEqualTo(7);
  }
}
