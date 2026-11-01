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
package com.integrallis.models.backend.onnx;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ReductionsTest {

  private static final Tensor M23 =
      Tensor.ofFloats(new long[] {2, 3}, new float[] {1, 2, 3, 4, 5, 6});

  @Test
  void sumsAlongRequestedAxes() {
    Tensor cols = Reductions.reduceSum(M23, Tensor.vector(0), true, false);
    assertThat(cols.shape()).containsExactly(1, 3);
    assertThat(cols.floats()).containsExactly(5, 7, 9);
    Tensor rows = Reductions.reduceSum(M23, Tensor.vector(-1), false, false);
    assertThat(rows.shape()).containsExactly(2);
    assertThat(rows.floats()).containsExactly(6, 15);
    Tensor all = Reductions.reduceSum(M23, null, false, false);
    assertThat(all.rank()).isZero();
    assertThat(all.floats()).containsExactly(21);
    assertThat(Reductions.reduceSum(M23, Tensor.ofLongs(new long[] {0}, new long[0]), true, true))
        .isSameAs(M23);
    assertThat(Reductions.reduceSum(Tensor.vector(1, 2, 3), Tensor.vector(0), false, false).longs())
        .containsExactly(6);
    assertThat(
            Reductions.reduceSum(
                    Tensor.ofDoubles(new long[] {2}, new double[] {0.1, 0.2}), null, true, false)
                .doubles())
        .containsExactly(0.30000000000000004);
  }

  @Test
  void accumulatesCumulativeSumsForwardReverseAndExclusive() {
    Tensor v = Tensor.ofFloats(new long[] {4}, new float[] {1, 2, 3, 4});
    assertThat(Reductions.cumSum(v, Tensor.scalar(0L), false, false).floats())
        .containsExactly(1, 3, 6, 10);
    assertThat(Reductions.cumSum(v, Tensor.scalar(0L), true, false).floats())
        .containsExactly(0, 1, 3, 6);
    assertThat(Reductions.cumSum(v, Tensor.scalar(0L), false, true).floats())
        .containsExactly(10, 9, 7, 4);
    assertThat(Reductions.cumSum(Tensor.vector(1, 2), Tensor.scalar(0L), false, false).longs())
        .containsExactly(1, 3);
    assertThat(
            Reductions.cumSum(
                    Tensor.ofDoubles(new long[] {2}, new double[] {0.5, 0.25}),
                    Tensor.scalar(-1L),
                    false,
                    false)
                .doubles())
        .containsExactly(0.5, 0.75);
    Tensor byColumn = Reductions.cumSum(M23, Tensor.scalar(0L), false, false);
    assertThat(byColumn.floats()).containsExactly(1, 2, 3, 5, 7, 9);
  }

  @Test
  void normalizesWithSoftmax() {
    Tensor s = Reductions.softmax(Tensor.ofFloats(new long[] {1, 2}, new float[] {0, 0}), -1);
    assertThat(s.floats()).containsExactly(0.5f, 0.5f);
    Tensor big = Reductions.softmax(Tensor.ofFloats(new long[] {2}, new float[] {1000, 0}), 0);
    assertThat(big.floats()[0]).isCloseTo(1f, within(1e-6f));
    Tensor d = Reductions.softmax(Tensor.ofDoubles(new long[] {2}, new double[] {1, 1}), 0);
    assertThat(d.doubles()).containsExactly(0.5, 0.5);
    Tensor axis0 = Reductions.softmax(M23, 0);
    assertThat(axis0.floats()[0] + axis0.floats()[3]).isCloseTo(1f, within(1e-6f));
    assertThat(Reductions.softmax(Tensor.vector(0, 0), 0).floats()).containsExactly(0.5f, 0.5f);
  }
}
