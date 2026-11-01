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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ElementwiseTest {

  private static Tensor f(float... v) {
    return Tensor.ofFloats(new long[] {v.length}, v);
  }

  private static Tensor d(double... v) {
    return Tensor.ofDoubles(new long[] {v.length}, v);
  }

  @Test
  void promotesArithmeticByStorageAndBroadcasts() {
    Tensor sum = Elementwise.arithmetic(Elementwise.Arithmetic.ADD, f(1, 2), Tensor.scalar(0.5f));
    assertThat(sum.floats()).containsExactly(1.5f, 2.5f);
    Tensor mixed = Elementwise.arithmetic(Elementwise.Arithmetic.MUL, f(2, 3), d(0.5, 0.25));
    assertThat(mixed.isDouble()).isTrue();
    assertThat(mixed.doubles()).containsExactly(1.0, 0.75);
    Tensor ints =
        Elementwise.arithmetic(
            Elementwise.Arithmetic.DIV, Tensor.vector(7, -7), Tensor.vector(2, 2));
    assertThat(ints.longs()).containsExactly(3, -3);
    Tensor int32 =
        Elementwise.arithmetic(
            Elementwise.Arithmetic.SUB,
            Tensor.ofInts(new long[] {1}, new long[] {5}),
            Tensor.ofInts(new long[] {1}, new long[] {7}));
    assertThat(int32.type()).isEqualTo(OnnxDataType.INT32);
    assertThat(int32.longs()).containsExactly(-2);
    Tensor outer =
        Elementwise.arithmetic(
            Elementwise.Arithmetic.ADD,
            Tensor.ofFloats(new long[] {2, 1}, new float[] {10, 20}),
            Tensor.ofFloats(new long[] {1, 3}, new float[] {1, 2, 3}));
    assertThat(outer.shape()).containsExactly(2, 3);
    assertThat(outer.floats()).containsExactly(11, 12, 13, 21, 22, 23);
    assertThat(Elementwise.arithmetic(Elementwise.Arithmetic.SUB, f(5), f(2)).floats())
        .containsExactly(3f);
    assertThat(Elementwise.arithmetic(Elementwise.Arithmetic.DIV, f(1), f(4)).floats())
        .containsExactly(0.25f);
    assertThat(Elementwise.arithmetic(Elementwise.Arithmetic.POW, f(2), f(3)).floats())
        .containsExactly(8f);
    assertThat(
            Elementwise.arithmetic(Elementwise.Arithmetic.POW, Tensor.vector(2), Tensor.vector(10))
                .longs())
        .containsExactly(1024);
    assertThat(Elementwise.arithmetic(Elementwise.Arithmetic.MIN, f(2, 9), f(3, 1)).floats())
        .containsExactly(2f, 1f);
    assertThat(
            Elementwise.arithmetic(Elementwise.Arithmetic.MIN, Tensor.vector(2), Tensor.vector(1))
                .longs())
        .containsExactly(1);
    assertThat(
            Elementwise.arithmetic(
                    Elementwise.Arithmetic.MOD_FLOOR, Tensor.vector(-7), Tensor.vector(3))
                .longs())
        .containsExactly(2);
    assertThat(
            Elementwise.arithmetic(
                    Elementwise.Arithmetic.MOD_TRUNC, Tensor.vector(-7), Tensor.vector(3))
                .longs())
        .containsExactly(-1);
    assertThat(Elementwise.arithmetic(Elementwise.Arithmetic.MOD_FLOOR, f(-7), f(3)).floats())
        .containsExactly(2f);
    assertThat(Elementwise.arithmetic(Elementwise.Arithmetic.MOD_TRUNC, d(-7), d(3)).doubles())
        .containsExactly(-1.0);
    assertThat(Elementwise.arithmetic(Elementwise.Arithmetic.ADD, f(1), Tensor.vector(2)).floats())
        .containsExactly(3f);
    assertThat(
            Elementwise.arithmetic(Elementwise.Arithmetic.MUL, Tensor.vector(3), f(2, 2)).floats())
        .containsExactly(6f, 6f);
  }

  @Test
  void appliesUnaryFunctionsInTheInputPrecision() {
    assertThat(Elementwise.sigmoid(f(0)).floats()).containsExactly(0.5f);
    assertThat(Elementwise.tanh(d(0)).doubles()).containsExactly(0.0);
    assertThat(Elementwise.exp(f(0)).floats()).containsExactly(1f);
    assertThat(Elementwise.sqrt(f(9)).floats()).containsExactly(3f);
    assertThat(Elementwise.sin(d(0)).doubles()).containsExactly(0.0);
    assertThat(Elementwise.cos(f(0)).floats()).containsExactly(1f);
    assertThat(Elementwise.erf(f(0)).floats()).containsExactly(0f);
    assertThat(Elementwise.erf(0.5)).isCloseTo(0.5204998778, within(1e-9));
    assertThat(Elementwise.erf(4.5)).isCloseTo(0.9999999998, within(1e-9));
    assertThat(Elementwise.erf(1.0000001)).isCloseTo(0.8427008, within(1e-7));
    assertThat(Elementwise.erf(0.9999999)).isCloseTo(0.8427007, within(1e-7));
    assertThat(Elementwise.erf(2.0)).isCloseTo(0.9953222650, within(1e-9));
    assertThat(Elementwise.erf(-7.0)).isEqualTo(-1.0);
    assertThat(Elementwise.erf(1.0)).isCloseTo(0.8427007929, within(1e-9));
    assertThat(Elementwise.erf(-1.0)).isCloseTo(-0.8427007929, within(1e-9));
    assertThat(Elementwise.erf(3.0)).isCloseTo(0.9999779095, within(1e-9));
    assertThat(Elementwise.leakyRelu(f(-2, 2), 0.2f).floats()).containsExactly(-0.4f, 2f);
    assertThat(Elementwise.floor(f(1.7f, -1.2f)).floats()).containsExactly(1f, -2f);
    Tensor integral = Tensor.vector(3);
    assertThat(Elementwise.floor(integral)).isSameAs(integral);
    assertThat(Elementwise.round(f(2.5f, 3.5f, -2.5f)).floats()).containsExactly(2f, 4f, -2f);
    assertThat(Elementwise.round(Tensor.vector(1)).longs()).containsExactly(1);
    assertThat(Elementwise.neg(f(1)).floats()).containsExactly(-1f);
    assertThat(Elementwise.neg(Tensor.vector(1, -2)).longs()).containsExactly(-1, 2);
    assertThat(Elementwise.neg(Tensor.ofInts(new long[] {1}, new long[] {4})).type())
        .isEqualTo(OnnxDataType.INT32);
    assertThat(Elementwise.isNaN(f(Float.NaN, 1)).bytes()).containsExactly(1, 0);
    assertThat(Elementwise.unary(Tensor.vector(4), Math::sqrt).floats()).containsExactly(2f);
  }

  @Test
  void clipsCastsComparesAndSelects() {
    assertThat(Elementwise.clip(f(-5, 0.5f, 5), Tensor.scalar(-1f), Tensor.scalar(1f)).floats())
        .containsExactly(-1f, 0.5f, 1f);
    assertThat(Elementwise.clip(f(-5, 5), null, Tensor.scalar(1f)).floats())
        .containsExactly(-5f, 1f);
    assertThat(Elementwise.clip(Tensor.vector(-5, 5), Tensor.scalar(0L), null).longs())
        .containsExactly(0, 5);
    assertThat(
            Elementwise.clip(Tensor.ofInts(new long[] {1}, new long[] {9}), null, Tensor.scalar(1L))
                .type())
        .isEqualTo(OnnxDataType.INT32);

    assertThat(Elementwise.cast(Tensor.vector(3), 1).floats()).containsExactly(3f);
    assertThat(Elementwise.cast(f(1.9f), 7).longs()).containsExactly(1);
    assertThat(Elementwise.cast(f(1.5f), 11).doubles()).containsExactly(1.5);
    assertThat(Elementwise.cast(f(0, 2), 9).bytes()).containsExactly(0, 1);
    assertThat(Elementwise.cast(Tensor.vector(1L << 33), 6).longs()).containsExactly(0);
    assertThat(Elementwise.cast(f(2), 10).floats()).containsExactly(2f);
    assertThatThrownBy(() -> Elementwise.cast(f(1), 8))
        .isInstanceOf(UnsupportedOperationException.class);

    assertThat(
            Elementwise.compare(
                    Elementwise.Comparison.EQUAL, Tensor.vector(1, 2), Tensor.vector(1, 3))
                .bytes())
        .containsExactly(1, 0);
    assertThat(Elementwise.compare(Elementwise.Comparison.LESS, f(1, 2), Tensor.scalar(2f)).bytes())
        .containsExactly(1, 0);
    assertThat(
            Elementwise.compare(
                    Elementwise.Comparison.GREATER, Tensor.vector(1, 3), Tensor.vector(2, 2))
                .bytes())
        .containsExactly(0, 1);
    assertThat(Elementwise.compare(Elementwise.Comparison.GREATER_OR_EQUAL, f(2), f(2)).bytes())
        .containsExactly(1);
    assertThat(
            Elementwise.compare(
                    Elementwise.Comparison.GREATER_OR_EQUAL, Tensor.vector(1), Tensor.vector(2))
                .bytes())
        .containsExactly(0);
    assertThat(Elementwise.compare(Elementwise.Comparison.EQUAL, f(1), f(1)).bytes())
        .containsExactly(1);
    assertThat(
            Elementwise.compare(Elementwise.Comparison.LESS, Tensor.vector(1), Tensor.vector(2))
                .bytes())
        .containsExactly(1);
    assertThat(Elementwise.compare(Elementwise.Comparison.GREATER, f(1), f(2)).bytes())
        .containsExactly(0);
    Tensor t = Tensor.ofBooleans(new long[] {2}, new boolean[] {true, false});
    assertThat(
            Elementwise.compare(
                    Elementwise.Comparison.AND,
                    t,
                    Tensor.ofBooleans(new long[] {1}, new boolean[] {true}))
                .bytes())
        .containsExactly(1, 0);

    Tensor where = Elementwise.where(t, f(1, 2), Tensor.scalar(9f));
    assertThat(where.floats()).containsExactly(1f, 9f);
    assertThat(Elementwise.where(t, Tensor.vector(1, 2), f(0.5f, 0.5f)).floats())
        .containsExactly(1f, 0.5f);
    assertThat(Elementwise.where(t, d(1, 2), f(3, 4)).doubles()).containsExactly(1.0, 4.0);
    assertThat(
            Elementwise.where(
                    t, Tensor.vector(1, 2), Tensor.ofInts(new long[] {2}, new long[] {3, 4}))
                .longs())
        .containsExactly(1, 4);
    assertThat(
            Elementwise.where(t, t, Tensor.ofBooleans(new long[] {2}, new boolean[] {true, true}))
                .bytes())
        .containsExactly(1, 1);
  }
}
