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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class StructuralTest {

  private static final Tensor M23 =
      Tensor.ofFloats(new long[] {2, 3}, new float[] {1, 2, 3, 4, 5, 6});

  @Test
  void reportsShapesAndReshapes() {
    assertThat(Structural.shape(M23, 0, Long.MAX_VALUE).longs()).containsExactly(2, 3);
    assertThat(Structural.shape(M23, 1, Long.MAX_VALUE).longs()).containsExactly(3);
    assertThat(Structural.shape(M23, 0, -1).longs()).containsExactly(2);
    assertThat(Structural.shape(M23, 5, Long.MAX_VALUE).size()).isZero();
    assertThat(Structural.reshape(M23, Tensor.vector(3, -1)).shape()).containsExactly(3, 2);
    assertThat(Structural.reshape(M23, Tensor.vector(0, -1, 1)).shape()).containsExactly(2, 3, 1);
    assertThat(
            Structural.reshape(Tensor.ofFloats(new long[] {0}, new float[0]), Tensor.vector(-1, 2))
                .shape())
        .containsExactly(0, 2);
    assertThatThrownBy(() -> Structural.reshape(M23, Tensor.vector(-1, -1)))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(Structural.flatten(Tensor.ofFloats(new long[] {2, 3, 4}, new float[24]), 2).shape())
        .containsExactly(6, 4);
    assertThat(Structural.flatten(M23, -2).shape()).containsExactly(1, 6);
    assertThat(Structural.flatten(Tensor.ofFloats(new long[] {0, 3}, new float[0]), 1).shape())
        .containsExactly(0, 0);
    assertThat(Structural.unsqueeze(M23, Tensor.vector(0, -1)).shape()).containsExactly(1, 2, 3, 1);
    assertThat(
            Structural.squeeze(Tensor.ofFloats(new long[] {1, 2, 1}, new float[2]), null).shape())
        .containsExactly(2);
    assertThat(
            Structural.squeeze(
                    Tensor.ofFloats(new long[] {1, 2, 1}, new float[2]), Tensor.vector(-1))
                .shape())
        .containsExactly(1, 2);
  }

  @Test
  void transposesConcatenatesAndSlices() {
    Tensor t = Structural.transpose(M23, new long[0]);
    assertThat(t.shape()).containsExactly(3, 2);
    assertThat(t.floats()).containsExactly(1, 4, 2, 5, 3, 6);
    Tensor cube = Tensor.ofFloats(new long[] {1, 2, 3}, new float[] {1, 2, 3, 4, 5, 6});
    assertThat(Structural.transpose(cube, new long[] {0, 2, 1}).floats())
        .containsExactly(1, 4, 2, 5, 3, 6);
    assertThat(Structural.transpose(cube, new long[] {2, 0, 1}).shape()).containsExactly(3, 1, 2);

    Tensor rows = Structural.concat(new Tensor[] {M23, M23}, 0);
    assertThat(rows.shape()).containsExactly(4, 3);
    Tensor cols =
        Structural.concat(
            new Tensor[] {M23, Tensor.ofFloats(new long[] {2, 1}, new float[] {7, 8})}, -1);
    assertThat(cols.shape()).containsExactly(2, 4);
    assertThat(cols.floats()).containsExactly(1, 2, 3, 7, 4, 5, 6, 8);
    Tensor promoted =
        Structural.concat(
            new Tensor[] {Tensor.vector(1), Tensor.ofInts(new long[] {1}, new long[] {2})}, 0);
    assertThat(promoted.longs()).containsExactly(1, 2);
    assertThat(
            Structural.concat(
                    new Tensor[] {Tensor.scalar(1.0f).reshape(new long[] {1}), Tensor.vector(2)}, 0)
                .floats())
        .containsExactly(1f, 2f);
    assertThat(
            Structural.concat(
                    new Tensor[] {
                      Tensor.ofDoubles(new long[] {1}, new double[] {1}), Tensor.vector(2)
                    },
                    0)
                .doubles())
        .containsExactly(1.0, 2.0);
    Tensor bools = Tensor.ofBooleans(new long[] {1}, new boolean[] {true});
    assertThat(Structural.concat(new Tensor[] {bools, Tensor.vector(0)}, 0).bytes())
        .containsExactly(1, 0);

    Tensor v = Tensor.vector(0, 1, 2, 3, 4, 5);
    assertThat(Structural.slice(v, Tensor.vector(1), Tensor.vector(4), null, null).longs())
        .containsExactly(1, 2, 3);
    assertThat(
            Structural.slice(v, Tensor.vector(-2), Tensor.vector(Long.MAX_VALUE), null, null)
                .longs())
        .containsExactly(4, 5);
    assertThat(
            Structural.slice(
                    v, Tensor.vector(0), Tensor.vector(6), Tensor.vector(0), Tensor.vector(2))
                .longs())
        .containsExactly(0, 2, 4);
    assertThat(
            Structural.slice(v, Tensor.vector(5), Tensor.vector(-7), null, Tensor.vector(-2))
                .longs())
        .containsExactly(5, 3, 1);
    assertThat(Structural.slice(v, Tensor.vector(4), Tensor.vector(2), null, null).size()).isZero();
    Tensor sub = Structural.slice(M23, Tensor.vector(1), Tensor.vector(3), Tensor.vector(1), null);
    assertThat(sub.shape()).containsExactly(2, 2);
    assertThat(sub.floats()).containsExactly(2, 3, 5, 6);
    assertThatThrownBy(
            () -> Structural.slice(v, Tensor.vector(0), Tensor.vector(1), null, Tensor.vector(0)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void gathersExpandsAndFills() {
    Tensor g0 = Structural.gather(M23, Tensor.vector(1, 0), 0);
    assertThat(g0.floats()).containsExactly(4, 5, 6, 1, 2, 3);
    Tensor g1 = Structural.gather(M23, Tensor.scalar(-1L), 1);
    assertThat(g1.shape()).containsExactly(2);
    assertThat(g1.floats()).containsExactly(3, 6);
    Tensor g2 = Structural.gather(M23, Tensor.ofLongs(new long[] {1, 2}, new long[] {0, 2}), 1);
    assertThat(g2.shape()).containsExactly(2, 1, 2);
    assertThatThrownBy(() -> Structural.gather(M23, Tensor.vector(3), 1))
        .isInstanceOf(IndexOutOfBoundsException.class);

    Tensor ge =
        Structural.gatherElements(
            M23, Tensor.ofLongs(new long[] {2, 2}, new long[] {0, 2, -1, 1}), 1);
    assertThat(ge.floats()).containsExactly(1, 3, 6, 5);
    Tensor ge0 =
        Structural.gatherElements(M23, Tensor.ofLongs(new long[] {1, 3}, new long[] {1, 0, 1}), 0);
    assertThat(ge0.floats()).containsExactly(4, 2, 6);

    Tensor expanded = Structural.expand(Tensor.vector(1, 2), Tensor.vector(2, 2));
    Tensor tiled =
        Structural.tile(Tensor.ofLongs(new long[] {1, 2}, new long[] {1, 2}), Tensor.vector(2, 2));
    assertThat(tiled.shape()).containsExactly(2, 4);
    assertThat(tiled.longs()).containsExactly(1, 2, 1, 2, 1, 2, 1, 2);
    assertThat(Structural.tile(M23, Tensor.vector(1, 1))).isSameAs(M23);
    assertThat(expanded.longs()).containsExactly(1, 2, 1, 2);
    assertThat(Structural.expand(M23, Tensor.vector(2, 3))).isSameAs(M23);
    assertThat(Structural.expand(Tensor.scalar(5f), Tensor.vector(3)).floats())
        .containsExactly(5, 5, 5);

    assertThat(Structural.constantOfShape(Tensor.vector(2), null).floats()).containsExactly(0, 0);
    assertThat(Structural.constantOfShape(Tensor.vector(1, 2), Tensor.vector(7)).longs())
        .containsExactly(7, 7);
    assertThat(Structural.constantOfShape(Tensor.ofLongs(new long[] {0}, new long[0]), null).rank())
        .isZero();

    assertThat(Structural.range(Tensor.scalar(0L), Tensor.scalar(5L), Tensor.scalar(2L)).longs())
        .containsExactly(0, 2, 4);
    assertThat(Structural.range(Tensor.scalar(5L), Tensor.scalar(0L), Tensor.scalar(-2L)).longs())
        .containsExactly(5, 3, 1);
    assertThat(Structural.range(Tensor.scalar(3L), Tensor.scalar(0L), Tensor.scalar(1L)).size())
        .isZero();
    assertThat(Structural.range(Tensor.scalar(0f), Tensor.scalar(1f), Tensor.scalar(0.5f)).floats())
        .containsExactly(0f, 0.5f);
    assertThat(
            Structural.range(
                    Tensor.ofDoubles(new long[0], new double[] {0}),
                    Tensor.scalar(1f),
                    Tensor.scalar(0.25f))
                .doubles())
        .containsExactly(0, 0.25, 0.5, 0.75);
  }
}
