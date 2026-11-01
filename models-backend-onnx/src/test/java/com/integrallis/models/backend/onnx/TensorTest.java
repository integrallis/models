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
class TensorTest {

  @Test
  void checksElementCountAgainstShape() {
    assertThatThrownBy(() -> Tensor.ofFloats(new long[] {2, 2}, new float[3]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Tensor.ofBytes(OnnxDataType.FLOAT, new long[] {1}, new byte[1]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Tensor.ofLongs(new long[] {-1}, new long[0]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void exposesStorageByKindAndWidensOnRead() {
    Tensor floats = Tensor.ofFloats(new long[] {2}, new float[] {1.5f, -2f});
    Tensor longs = Tensor.ofInts(new long[] {2}, new long[] {3, -4});
    Tensor bytes = Tensor.ofBytes(OnnxDataType.UINT8, new long[] {1}, new byte[] {(byte) 200});

    assertThat(floats.isFloat()).isTrue();
    assertThat(longs.isIntegral()).isTrue();
    assertThat(bytes.isBytes()).isTrue();
    assertThat(longs.type()).isEqualTo(OnnxDataType.INT32);
    assertThat(floats.longAt(0)).isEqualTo(1);
    assertThat(longs.floatAt(1)).isEqualTo(-4f);
    assertThat(bytes.floatAt(0)).isEqualTo(200f);
    assertThat(bytes.longAt(0)).isEqualTo(200);
    assertThat(Tensor.ofBytes(OnnxDataType.INT8, new long[] {1}, new byte[] {-3}).floatAt(0))
        .isEqualTo(-3f);
    assertThatThrownBy(floats::longs).isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(longs::floats).isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(floats::bytes).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void convertsBetweenFloatAndInt64Storage() {
    Tensor longs = Tensor.vector(1, 2, 3);
    Tensor asFloat = longs.toFloat();
    assertThat(asFloat.type()).isEqualTo(OnnxDataType.FLOAT);
    assertThat(asFloat.floats()).containsExactly(1f, 2f, 3f);
    assertThat(asFloat.toFloat()).isSameAs(asFloat);
    Tensor back = asFloat.toInt64();
    assertThat(back.longs()).containsExactly(1, 2, 3);
    assertThat(longs.toInt64()).isSameAs(longs);
    assertThat(Tensor.ofInts(new long[] {1}, new long[] {7}).toInt64().type())
        .isEqualTo(OnnxDataType.INT64);
  }

  @Test
  void keepsDoublePrecisionAndConvertsAcrossEveryStore() {
    Tensor doubles = Tensor.ofDoubles(new long[] {2}, new double[] {1.0 / 3.0, -2.0});
    assertThat(doubles.isDouble()).isTrue();
    assertThat(doubles.isFloatingPoint()).isTrue();
    assertThat(doubles.doubleAt(0)).isEqualTo(1.0 / 3.0);
    assertThat(doubles.floatAt(0)).isEqualTo((float) (1.0 / 3.0));
    assertThat(doubles.longAt(1)).isEqualTo(-2);
    assertThat(doubles.booleanAt(1)).isTrue();
    assertThat(doubles.toDouble()).isSameAs(doubles);
    assertThat(doubles.toFloat().type()).isEqualTo(OnnxDataType.FLOAT);
    assertThat(doubles.toInt64().longs()).containsExactly(0, -2);
    assertThat(Tensor.vector(3).toDouble().doubles()).containsExactly(3.0);
    assertThat(Tensor.vector(3).doubleAt(0)).isEqualTo(3.0);
    assertThat(Tensor.ofFloats(new long[] {1}, new float[] {0.5f}).doubleAt(0)).isEqualTo(0.5);
    assertThat(
            Tensor.ofBytes(OnnxDataType.UINT8, new long[] {1}, new byte[] {(byte) 255}).doubleAt(0))
        .isEqualTo(255.0);
    assertThatThrownBy(doubles::floats).isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(Tensor.scalar(1f)::doubles).isInstanceOf(IllegalStateException.class);

    Tensor booleans = Tensor.ofBooleans(new long[] {2}, new boolean[] {true, false});
    assertThat(booleans.isBoolean()).isTrue();
    assertThat(booleans.booleanAt(0)).isTrue();
    assertThat(booleans.booleanAt(1)).isFalse();
    assertThat(booleans.toBoolean()).isSameAs(booleans);
    assertThat(Tensor.vector(0, 5).toBoolean().bytes()).containsExactly(0, 1);
    assertThat(Tensor.vector(0, 5).booleanAt(1)).isTrue();
    assertThat(Tensor.scalar(0f).booleanAt(0)).isFalse();

    Tensor target = doubles.emptyLike(new long[] {1});
    target.copyElement(doubles, 1, 0);
    assertThat(target.doubles()).containsExactly(-2.0);
    assertThat(Tensor.scalar(1f).emptyLike(new long[] {2}).floats()).hasSize(2);
    assertThat(Tensor.vector(1).emptyLike(new long[] {2}).longs()).hasSize(2);
    assertThat(booleans.emptyLike(new long[] {3}).bytes()).hasSize(3);
    Tensor floats = Tensor.ofFloats(new long[] {2}, new float[] {1f, 2f});
    Tensor floatTarget = floats.emptyLike(new long[] {1});
    floatTarget.copyElement(floats, 1, 0);
    assertThat(floatTarget.floats()).containsExactly(2f);
    Tensor longTarget = Tensor.vector(7, 8).emptyLike(new long[] {1});
    longTarget.copyElement(Tensor.vector(7, 8), 0, 0);
    assertThat(longTarget.longs()).containsExactly(7);
    Tensor byteTarget = booleans.emptyLike(new long[] {1});
    byteTarget.copyElement(booleans, 0, 0);
    assertThat(byteTarget.bytes()).containsExactly(1);
    assertThat(floats.sameKind(floatTarget)).isTrue();
    assertThat(floats.sameKind(doubles)).isFalse();
    assertThat(Tensor.vector(1).sameKind(Tensor.ofInts(new long[] {1}, new long[] {1}))).isFalse();
  }

  @Test
  void reportsShapeStridesAndAxes() {
    Tensor tensor = Tensor.ofFloats(new long[] {2, 3, 4}, new float[24]);
    assertThat(tensor.rank()).isEqualTo(3);
    assertThat(tensor.size()).isEqualTo(24);
    assertThat(tensor.shape()).containsExactly(2, 3, 4);
    assertThat(tensor.strides()).containsExactly(12, 4, 1);
    assertThat(tensor.dim(-1)).isEqualTo(4);
    assertThat(tensor.normalizeAxis(-3)).isZero();
    assertThatThrownBy(() -> tensor.dim(3)).isInstanceOf(IllegalArgumentException.class);
    assertThat(tensor.reshape(new long[] {6, 4}).shape()).containsExactly(6, 4);
    assertThat(Tensor.scalar(2f).rank()).isZero();
    assertThat(Tensor.scalar(2L).longs()).containsExactly(2L);
    assertThat(tensor.toString()).isEqualTo("FLOAT[2, 3, 4]");
  }
}
