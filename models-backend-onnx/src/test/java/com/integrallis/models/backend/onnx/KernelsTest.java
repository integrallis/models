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

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class KernelsTest {

  private static final Offset<Float> EPS = within(1e-5f);

  private static Tensor f(long[] shape, float... v) {
    return Tensor.ofFloats(shape, v);
  }

  @Test
  void multipliesMatricesWithBatchBroadcastAndVectorPromotion() {
    Tensor a = f(new long[] {2, 3}, 1, 2, 3, 4, 5, 6);
    Tensor b = f(new long[] {3, 2}, 1, 0, 0, 1, 1, 1);
    assertThat(Linear.matMul(a, b).floats()).containsExactly(4, 5, 10, 11);
    Tensor batched = Linear.matMul(f(new long[] {2, 1, 2}, 1, 2, 3, 4), f(new long[] {2, 1}, 1, 1));
    assertThat(batched.shape()).containsExactly(2, 1, 1);
    assertThat(batched.floats()).containsExactly(3, 7);
    Tensor broadcast =
        Linear.matMul(f(new long[] {2, 1, 2}, 1, 0, 0, 1), f(new long[] {1, 2, 2}, 1, 2, 3, 4));
    assertThat(broadcast.shape()).containsExactly(2, 1, 2);
    assertThat(broadcast.floats()).containsExactly(1, 2, 3, 4);
    assertThat(Linear.matMul(f(new long[] {2}, 1, 2), f(new long[] {2}, 3, 4)).rank()).isZero();
    assertThat(Linear.matMul(f(new long[] {2}, 1, 2), f(new long[] {2}, 3, 4)).floats())
        .containsExactly(11f);
    assertThat(Linear.matMul(f(new long[] {3}, 1, 1, 1), b).shape()).containsExactly(2);
    assertThat(Linear.matMul(f(new long[] {3}, 1, 1, 1), b).floats()).containsExactly(2, 2);
    assertThat(Linear.matMul(a, f(new long[] {3}, 1, 1, 1)).floats()).containsExactly(6, 15);
    Tensor dbl =
        Linear.matMul(
            Tensor.ofDoubles(new long[] {1, 2}, new double[] {0.5, 0.25}),
            f(new long[] {2, 1}, 2, 4));
    assertThat(dbl.doubles()).containsExactly(2.0);
    assertThatThrownBy(() -> Linear.matMul(a, a)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void computesGemmInEveryTransposeLayout() {
    Tensor a = f(new long[] {2, 3}, 1, 2, 3, 4, 5, 6);
    Tensor bT = f(new long[] {2, 3}, 1, 0, 0, 0, 1, 0);
    Tensor c = f(new long[] {2}, 10, 20);
    Tensor y = Linear.gemm(a, bT, c, 1f, 1f, false, true);
    assertThat(y.floats()).containsExactly(11, 22, 14, 25);
    Tensor plain =
        Linear.gemm(a, f(new long[] {3, 2}, 1, 0, 0, 1, 0, 0), null, 2f, 1f, false, false);
    assertThat(plain.floats()).containsExactly(2, 4, 8, 10);
    Tensor transA =
        Linear.gemm(
            f(new long[] {3, 2}, 1, 4, 2, 5, 3, 6),
            f(new long[] {3, 1}, 1, 1, 1),
            Tensor.scalar(1f),
            1f,
            0.5f,
            true,
            false);
    assertThat(transA.floats()).containsExactly(6.5f, 15.5f);
    assertThatThrownBy(() -> Linear.gemm(a, a, null, 1f, 1f, false, false))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void convolvesOneDimensionalSignalsWithPaddingStrideDilationAndGroups() {
    Tensor x = f(new long[] {1, 1, 5}, 1, 2, 3, 4, 5);
    Tensor w = f(new long[] {1, 1, 3}, 1, 0, -1);
    Tensor same =
        Convolution.conv(x, w, null, new long[] {1}, 1, new long[] {1, 1}, new long[] {1});
    assertThat(same.shape()).containsExactly(1, 1, 5);
    assertThat(same.floats()).containsExactly(-2, -2, -2, -2, 4);
    Tensor strided =
        Convolution.conv(x, w, f(new long[] {1}, 100), new long[0], 1, new long[0], new long[] {2});
    assertThat(strided.floats()).containsExactly(98, 98);
    Tensor dilated =
        Convolution.conv(
            x,
            f(new long[] {1, 1, 2}, 1, 1),
            null,
            new long[] {2},
            1,
            new long[] {0, 0},
            new long[] {1});
    assertThat(dilated.floats()).containsExactly(4, 6, 8);
    Tensor grouped =
        Convolution.conv(
            f(new long[] {1, 2, 2}, 1, 2, 3, 4),
            f(new long[] {2, 1, 1}, 10, 100),
            null,
            new long[0],
            2,
            new long[0],
            new long[0]);
    assertThat(grouped.floats()).containsExactly(10, 20, 300, 400);
    Tensor twoOut =
        Convolution.conv(
            f(new long[] {1, 2, 2}, 1, 2, 3, 4),
            f(new long[] {2, 2, 1}, 1, 1, 1, -1),
            null,
            new long[0],
            1,
            new long[0],
            new long[0]);
    assertThat(twoOut.floats()).containsExactly(4, 6, -2, -2);
    assertThatThrownBy(
            () ->
                Convolution.conv(
                    f(new long[] {1, 1, 2, 2}, 1, 2, 3, 4),
                    f(new long[] {1, 1, 1, 1}, 1),
                    null,
                    new long[0],
                    1,
                    new long[0],
                    new long[0]))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(
            () ->
                Convolution.conv(
                    x,
                    f(new long[] {1, 2, 1}, 1, 1),
                    null,
                    new long[0],
                    1,
                    new long[0],
                    new long[0]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                Convolution.conv(
                    x,
                    f(new long[] {1, 1, 7}, 1, 1, 1, 1, 1, 1, 1),
                    null,
                    new long[0],
                    1,
                    new long[0],
                    new long[0]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void transposesConvolutionsWithStrideOutputPaddingAndGroups() {
    Tensor x = f(new long[] {1, 1, 3}, 1, 2, 3);
    Tensor w = f(new long[] {1, 1, 2}, 1, 1);
    Tensor up =
        Convolution.convTranspose(
            x, w, null, new long[0], 1, new long[0], new long[0], new long[] {2});
    assertThat(up.shape()).containsExactly(1, 1, 6);
    assertThat(up.floats()).containsExactly(1, 1, 2, 2, 3, 3);
    Tensor padded =
        Convolution.convTranspose(
            x,
            f(new long[] {1, 1, 3}, 1, 1, 1),
            f(new long[] {1}, 1),
            new long[0],
            1,
            new long[] {1},
            new long[] {1, 1},
            new long[] {2});
    assertThat(padded.shape()).containsExactly(1, 1, 6);
    assertThat(padded.floats()).containsExactly(2, 4, 3, 6, 4, 4);
    Tensor grouped =
        Convolution.convTranspose(
            f(new long[] {1, 2, 1}, 1, 2),
            f(new long[] {2, 1, 1}, 10, 100),
            null,
            new long[0],
            2,
            new long[0],
            new long[0],
            new long[0]);
    assertThat(grouped.floats()).containsExactly(10, 200);
    Tensor fanOut =
        Convolution.convTranspose(
            x,
            f(new long[] {1, 2, 1}, 1, -1),
            null,
            new long[0],
            1,
            new long[0],
            new long[0],
            new long[0]);
    assertThat(fanOut.floats()).containsExactly(1, 2, 3, -1, -2, -3);
    assertThatThrownBy(
            () ->
                Convolution.convTranspose(
                    x,
                    f(new long[] {2, 1, 1}, 1, 1),
                    null,
                    new long[0],
                    1,
                    new long[0],
                    new long[0],
                    new long[0]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                Convolution.convTranspose(
                    x, w, null, new long[0], 1, new long[0], new long[] {5, 5}, new long[0]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void normalizesLayersAndInstances() {
    Tensor x = f(new long[] {2, 2}, 1, 3, -1, 1);
    Tensor[] ln =
        Normalization.layerNorm(x, f(new long[] {2}, 1, 2), f(new long[] {2}, 0, 1), -1, 0f);
    assertThat(ln[0].floats()).containsExactly(-1, 3, -1, 3);
    assertThat(ln[1].shape()).containsExactly(2, 1);
    assertThat(ln[1].floats()).containsExactly(2, 0);
    assertThat(ln[2].floats()).containsExactly(1, 1);
    Tensor[] noBias =
        Normalization.layerNorm(x, Tensor.scalar(1f).reshape(new long[] {1}), null, 1, 1e-12f);
    assertThat(noBias[0].floats()[0]).isCloseTo(-1f, EPS);

    Tensor planes = f(new long[] {1, 2, 2}, 1, 3, 10, 10);
    Tensor in =
        Normalization.instanceNorm(planes, f(new long[] {2}, 1, 1), f(new long[] {2}, 0, 5), 0f);
    assertThat(in.floats()[0]).isCloseTo(-1f, EPS);
    assertThat(in.floats()[1]).isCloseTo(1f, EPS);
    assertThat(Float.isNaN(in.floats()[2])).isTrue();
    Tensor stable = Normalization.instanceNorm(planes, f(new long[] {2}, 1, 1), null, 1e-5f);
    assertThat(stable.floats()[2]).isCloseTo(0f, EPS);
  }

  @Test
  void runsAnLstmCellForwardReverseAndBidirectionally() {
    // hidden 1, input 1: W gates [i, o, f, c] = [0, 0, 0, 1] so the cell copies tanh(x) into c
    // with i = o = f = sigmoid(0) = 0.5.
    Tensor x = f(new long[] {2, 1, 1}, 1, 2);
    Tensor w = f(new long[] {1, 4, 1}, 0, 0, 0, 1);
    Tensor r = f(new long[] {1, 4, 1}, 0, 0, 0, 0);
    Tensor[] out = Recurrent.lstm(x, w, r, null, null, null, null, "forward", 1);
    double c1 = 0.5 * Math.tanh(1);
    double h1 = 0.5 * Math.tanh(c1);
    double c2 = 0.5 * c1 + 0.5 * Math.tanh(2);
    double h2 = 0.5 * Math.tanh(c2);
    assertThat(out[0].shape()).containsExactly(2, 1, 1, 1);
    assertThat(out[0].floats()[0]).isCloseTo((float) h1, EPS);
    assertThat(out[0].floats()[1]).isCloseTo((float) h2, EPS);
    assertThat(out[1].floats()[0]).isCloseTo((float) h2, EPS);
    assertThat(out[2].floats()[0]).isCloseTo((float) c2, EPS);

    Tensor[] reverse = Recurrent.lstm(x, w, r, null, null, null, null, "reverse", 1);
    double rc1 = 0.5 * Math.tanh(2);
    double rh1 = 0.5 * Math.tanh(rc1);
    assertThat(reverse[0].floats()[1]).isCloseTo((float) rh1, EPS);

    Tensor w2 = f(new long[] {2, 4, 1}, 0, 0, 0, 1, 0, 0, 0, 1);
    Tensor r2 = f(new long[] {2, 4, 1}, 0, 0, 0, 0, 0, 0, 0, 0);
    Tensor b2 = f(new long[] {2, 8}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    Tensor[] both = Recurrent.lstm(x, w2, r2, b2, Tensor.vector(2), null, null, "bidirectional", 1);
    assertThat(both[0].shape()).containsExactly(2, 2, 1, 1);
    assertThat(both[0].floats()[0]).isCloseTo((float) h1, EPS);
    assertThat(both[0].floats()[1]).isCloseTo(reverse[0].floats()[0], EPS);
    assertThat(both[0].floats()[3]).isCloseTo((float) rh1, EPS);

    Tensor[] seeded =
        Recurrent.lstm(
            x,
            w,
            r,
            null,
            null,
            f(new long[] {1, 1, 1}, 0.3f),
            f(new long[] {1, 1, 1}, 0.7f),
            "forward",
            1);
    assertThat(seeded[2].floats()[0])
        .isCloseTo((float) (0.5 * (0.5 * 0.7 + c1) + 0.5 * Math.tanh(2)), EPS);
    assertThatThrownBy(() -> Recurrent.lstm(x, w, r, null, null, null, null, "bidirectional", 1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void resizesByNearestNeighbourAndPads() {
    Tensor x = f(new long[] {1, 1, 3}, 1, 2, 3);
    Tensor doubled =
        Resampling.resizeNearest(
            x, f(new long[] {3}, 1, 1, 2), null, "nearest", "asymmetric", "floor");
    assertThat(doubled.floats()).containsExactly(1, 1, 2, 2, 3, 3);
    Tensor sized =
        Resampling.resizeNearest(x, null, Tensor.vector(1, 1, 5), "nearest", "asymmetric", "floor");
    assertThat(sized.floats()).containsExactly(1, 1, 2, 2, 3);
    assertThatThrownBy(
            () ->
                Resampling.resizeNearest(
                    x, f(new long[] {3}, 1, 1, 2), null, "linear", "asymmetric", "floor"))
        .isInstanceOf(UnsupportedOperationException.class);

    Tensor v = f(new long[] {1, 4}, 1, 2, 3, 4);
    assertThat(Resampling.pad(v, Tensor.vector(0, 2, 0, 1), null, "reflect").floats())
        .containsExactly(3, 2, 1, 2, 3, 4, 3);
    assertThat(Resampling.pad(v, Tensor.vector(0, 1, 0, 2), null, "edge").floats())
        .containsExactly(1, 1, 2, 3, 4, 4, 4);
    assertThat(Resampling.pad(v, Tensor.vector(0, 1, 0, 1), Tensor.scalar(9f), "constant").floats())
        .containsExactly(9, 1, 2, 3, 4, 9);
    assertThat(Resampling.pad(v, Tensor.vector(1, 0, 0, 0), null, "constant").floats())
        .containsExactly(0, 0, 0, 0, 1, 2, 3, 4);
    assertThat(Resampling.pad(f(new long[] {1}, 7), Tensor.vector(2, 2), null, "reflect").floats())
        .containsExactly(7, 7, 7, 7, 7);
    assertThat(Resampling.pad(v, Tensor.vector(0, 5, 0, 0), null, "reflect").floats())
        .containsExactly(2, 3, 4, 3, 2, 1, 2, 3, 4);
    assertThatThrownBy(() -> Resampling.pad(v, Tensor.vector(0, 1, 0, 1), null, "wrap"))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  void dequantizesPerTensorAndPerAxis() {
    Tensor q = Tensor.ofBytes(OnnxDataType.INT8, new long[] {2, 2}, new byte[] {-128, 0, 1, 127});
    assertThat(
            Quantization.dequantizeLinear(
                    q,
                    Tensor.scalar(0.5f),
                    Tensor.ofBytes(OnnxDataType.INT8, new long[0], new byte[] {1}),
                    1)
                .floats())
        .containsExactly(-64.5f, -0.5f, 0f, 63f);
    assertThat(Quantization.dequantizeLinear(q, f(new long[] {2}, 1, 10), null, 0).floats())
        .containsExactly(-128, 0, 10, 1270);
    assertThat(Quantization.dequantizeLinear(q, f(new long[] {2}, 1, 10), null, 1).floats())
        .containsExactly(-128, 0, 1, 1270);
    assertThatThrownBy(() -> Quantization.dequantizeLinear(q, f(new long[] {3}, 1, 1, 1), null, 0))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
