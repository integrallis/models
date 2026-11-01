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

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.LongBinaryOperator;

/**
 * Elementwise operators with numpy-style broadcasting.
 *
 * <p>Arithmetic follows the storage of its operands: double if either operand is double, float if
 * either is float, otherwise 64-bit integer. Comparisons and {@code IsNaN} produce {@code BOOL}.
 */
final class Elementwise {

  private Elementwise() {}

  /**
   * Applies a unary function in the tensor's own precision; integer inputs are promoted to float.
   */
  static Tensor unary(Tensor x, DoubleUnaryOperator f) {
    if (x.isDouble()) {
      double[] in = x.doubles();
      double[] out = new double[in.length];
      for (int i = 0; i < in.length; i++) {
        out[i] = f.applyAsDouble(in[i]);
      }
      return Tensor.ofDoubles(x.shape(), out);
    }
    float[] in = x.toFloat().floats();
    float[] out = new float[in.length];
    for (int i = 0; i < in.length; i++) {
      out[i] = (float) f.applyAsDouble(in[i]);
    }
    return Tensor.ofFloats(x.shape(), out);
  }

  static Tensor sqrt(Tensor x) {
    return unary(x, Math::sqrt);
  }

  static Tensor exp(Tensor x) {
    return unary(x, Math::exp);
  }

  static Tensor tanh(Tensor x) {
    return unary(x, Math::tanh);
  }

  static Tensor sigmoid(Tensor x) {
    return unary(x, v -> 1.0 / (1.0 + Math.exp(-v)));
  }

  static Tensor sin(Tensor x) {
    return unary(x, Math::sin);
  }

  static Tensor cos(Tensor x) {
    return unary(x, Math::cos);
  }

  static Tensor erf(Tensor x) {
    return unary(x, Elementwise::erf);
  }

  static Tensor leakyRelu(Tensor x, float alpha) {
    return unary(x, v -> v >= 0 ? v : alpha * v);
  }

  static Tensor floor(Tensor x) {
    return x.isFloatingPoint() ? unary(x, Math::floor) : x;
  }

  /** Round half to even, which is what ONNX specifies. */
  static Tensor round(Tensor x) {
    return x.isFloatingPoint() ? unary(x, Math::rint) : x;
  }

  static Tensor neg(Tensor x) {
    if (x.isIntegral()) {
      long[] in = x.longs();
      long[] out = new long[in.length];
      for (int i = 0; i < in.length; i++) {
        out[i] = -in[i];
      }
      return x.type() == OnnxDataType.INT32
          ? Tensor.ofInts(x.shape(), out)
          : Tensor.ofLongs(x.shape(), out);
    }
    return unary(x, v -> -v);
  }

  static Tensor isNaN(Tensor x) {
    boolean[] out = new boolean[x.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = Double.isNaN(x.doubleAt(i));
    }
    return Tensor.ofBooleans(x.shape(), out);
  }

  /** {@code Clip} with optional min and max tensors (scalars). */
  static Tensor clip(Tensor x, Tensor min, Tensor max) {
    double lo = min == null ? Double.NEGATIVE_INFINITY : min.doubleAt(0);
    double hi = max == null ? Double.POSITIVE_INFINITY : max.doubleAt(0);
    if (x.isIntegral()) {
      long[] in = x.longs();
      long[] out = new long[in.length];
      for (int i = 0; i < in.length; i++) {
        out[i] = (long) Math.min(Math.max(in[i], lo), hi);
      }
      return x.type() == OnnxDataType.INT32
          ? Tensor.ofInts(x.shape(), out)
          : Tensor.ofLongs(x.shape(), out);
    }
    return unary(x, v -> Math.min(Math.max(v, lo), hi));
  }

  /** {@code Cast} to an ONNX element type code. */
  static Tensor cast(Tensor x, long toCode) {
    OnnxDataType to = OnnxDataType.fromCode(toCode);
    switch (to) {
      case FLOAT, FLOAT16, BFLOAT16 -> {
        return x.toFloat();
      }
      case DOUBLE -> {
        return x.toDouble();
      }
      case INT64 -> {
        return x.toInt64();
      }
      case INT32 -> {
        long[] values = new long[x.size()];
        for (int i = 0; i < values.length; i++) {
          values[i] = (int) x.longAt(i);
        }
        return Tensor.ofInts(x.shape(), values);
      }
      case BOOL -> {
        return x.toBoolean();
      }
      default -> throw new UnsupportedOperationException("Cast to " + to + " is not supported");
    }
  }

  // --- binary arithmetic ---

  enum Arithmetic {
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    MIN,
    MOD_FLOOR,
    MOD_TRUNC
  }

  static Tensor arithmetic(Arithmetic op, Tensor a, Tensor b) {
    long[] shape = Broadcast.shape(a.shape(), b.shape());
    int[] ia = Broadcast.same(a.shape(), shape) ? null : Broadcast.indexMap(a.shape(), shape);
    int[] ib = Broadcast.same(b.shape(), shape) ? null : Broadcast.indexMap(b.shape(), shape);
    int count = (int) Tensor.elementCount(shape);
    if (a.isDouble() || b.isDouble()) {
      DoubleBinaryOperator f = doubleOperator(op);
      double[] out = new double[count];
      for (int i = 0; i < count; i++) {
        out[i] =
            f.applyAsDouble(a.doubleAt(ia == null ? i : ia[i]), b.doubleAt(ib == null ? i : ib[i]));
      }
      return Tensor.ofDoubles(shape, out);
    }
    if (a.isFloat() || b.isFloat()) {
      float[] out = new float[count];
      if (ia == null && ib == null && a.isFloat() && b.isFloat()) {
        float[] fa = a.floats();
        float[] fb = b.floats();
        switch (op) {
          case ADD -> {
            for (int i = 0; i < count; i++) {
              out[i] = fa[i] + fb[i];
            }
          }
          case SUB -> {
            for (int i = 0; i < count; i++) {
              out[i] = fa[i] - fb[i];
            }
          }
          case MUL -> {
            for (int i = 0; i < count; i++) {
              out[i] = fa[i] * fb[i];
            }
          }
          case DIV -> {
            for (int i = 0; i < count; i++) {
              out[i] = fa[i] / fb[i];
            }
          }
          default -> {
            DoubleBinaryOperator f = doubleOperator(op);
            for (int i = 0; i < count; i++) {
              out[i] = (float) f.applyAsDouble(fa[i], fb[i]);
            }
          }
        }
        return Tensor.ofFloats(shape, out);
      }
      DoubleBinaryOperator f = doubleOperator(op);
      for (int i = 0; i < count; i++) {
        out[i] =
            (float)
                f.applyAsDouble(
                    a.floatAt(ia == null ? i : ia[i]), b.floatAt(ib == null ? i : ib[i]));
      }
      return Tensor.ofFloats(shape, out);
    }
    LongBinaryOperator f = longOperator(op);
    long[] out = new long[count];
    for (int i = 0; i < count; i++) {
      out[i] = f.applyAsLong(a.longAt(ia == null ? i : ia[i]), b.longAt(ib == null ? i : ib[i]));
    }
    boolean int32 = a.type() == OnnxDataType.INT32 && b.type() == OnnxDataType.INT32;
    return int32 ? Tensor.ofInts(shape, out) : Tensor.ofLongs(shape, out);
  }

  private static DoubleBinaryOperator doubleOperator(Arithmetic op) {
    return switch (op) {
      case ADD -> Double::sum;
      case SUB -> (x, y) -> x - y;
      case MUL -> (x, y) -> x * y;
      case DIV -> (x, y) -> x / y;
      case POW -> Math::pow;
      case MIN -> Math::min;
      case MOD_FLOOR -> (x, y) -> x - Math.floor(x / y) * y;
      case MOD_TRUNC -> (x, y) -> x % y;
    };
  }

  private static LongBinaryOperator longOperator(Arithmetic op) {
    return switch (op) {
      case ADD -> Long::sum;
      case SUB -> (x, y) -> x - y;
      case MUL -> (x, y) -> x * y;
      case DIV -> (x, y) -> x / y;
      case POW -> (x, y) -> (long) Math.pow(x, y);
      case MIN -> Math::min;
      case MOD_FLOOR -> Math::floorMod;
      case MOD_TRUNC -> (x, y) -> x % y;
    };
  }

  // --- comparisons and logic ---

  enum Comparison {
    EQUAL,
    LESS,
    GREATER,
    GREATER_OR_EQUAL,
    AND
  }

  static Tensor compare(Comparison op, Tensor a, Tensor b) {
    long[] shape = Broadcast.shape(a.shape(), b.shape());
    int[] ia = Broadcast.same(a.shape(), shape) ? null : Broadcast.indexMap(a.shape(), shape);
    int[] ib = Broadcast.same(b.shape(), shape) ? null : Broadcast.indexMap(b.shape(), shape);
    int count = (int) Tensor.elementCount(shape);
    boolean[] out = new boolean[count];
    boolean integral = !a.isFloatingPoint() && !b.isFloatingPoint();
    for (int i = 0; i < count; i++) {
      int ja = ia == null ? i : ia[i];
      int jb = ib == null ? i : ib[i];
      out[i] =
          switch (op) {
            case EQUAL ->
                integral ? a.longAt(ja) == b.longAt(jb) : a.doubleAt(ja) == b.doubleAt(jb);
            case LESS -> integral ? a.longAt(ja) < b.longAt(jb) : a.doubleAt(ja) < b.doubleAt(jb);
            case GREATER ->
                integral ? a.longAt(ja) > b.longAt(jb) : a.doubleAt(ja) > b.doubleAt(jb);
            case GREATER_OR_EQUAL ->
                integral ? a.longAt(ja) >= b.longAt(jb) : a.doubleAt(ja) >= b.doubleAt(jb);
            case AND -> a.booleanAt(ja) && b.booleanAt(jb);
          };
    }
    return Tensor.ofBooleans(shape, out);
  }

  /** {@code Where(condition, x, y)} with three-way broadcasting; the result takes x's storage. */
  static Tensor where(Tensor condition, Tensor x, Tensor y) {
    long[] shape = Broadcast.shape(Broadcast.shape(condition.shape(), x.shape()), y.shape());
    int[] ic =
        Broadcast.same(condition.shape(), shape)
            ? null
            : Broadcast.indexMap(condition.shape(), shape);
    int[] ix = Broadcast.same(x.shape(), shape) ? null : Broadcast.indexMap(x.shape(), shape);
    int[] iy = Broadcast.same(y.shape(), shape) ? null : Broadcast.indexMap(y.shape(), shape);
    Tensor left = x;
    Tensor right = y;
    if (!x.sameKind(y)) {
      // Operand types match in a valid graph; when storage differs, promote to the wider one.
      if (x.isDouble() || y.isDouble()) {
        left = x.toDouble();
        right = y.toDouble();
      } else if (x.isFloat() || y.isFloat()) {
        left = x.toFloat();
        right = y.toFloat();
      } else if (!x.isBoolean()) {
        left = x.toInt64();
        right = y.toInt64();
      }
    }
    Tensor out = left.emptyLike(shape);
    int count = (int) Tensor.elementCount(shape);
    for (int i = 0; i < count; i++) {
      boolean take = condition.booleanAt(ic == null ? i : ic[i]);
      if (take) {
        out.copyElement(left, ix == null ? i : ix[i], i);
      } else {
        out.copyElement(right, iy == null ? i : iy[i], i);
      }
    }
    return out;
  }

  /**
   * Error function in double precision: the Maclaurin series below {@code |x| = 4}, where the
   * alternating terms are summed until they vanish against the partial sum, and the asymptotic
   * expansion of {@code erfc} beyond it, where {@code erfc} is already below {@code 1.6e-8}. Exact
   * at zero and within a few units in the last place of {@code float} everywhere, which is what
   * parity with the reference runtime needs.
   */
  static double erf(double x) {
    double z = Math.abs(x);
    if (z == 0.0) {
      return x;
    }
    double result;
    if (z < 4.0) {
      double term = z;
      double sum = z;
      double z2 = z * z;
      for (int n = 1; n < 200; n++) {
        term *= -z2 / n;
        double contribution = term / (2 * n + 1);
        sum += contribution;
        if (Math.abs(contribution) < 1e-17 * Math.abs(sum)) {
          break;
        }
      }
      result = sum * 2.0 / Math.sqrt(Math.PI);
    } else {
      double z2 = z * z;
      double series = 1 - 1 / (2 * z2) + 3 / (4 * z2 * z2) - 15 / (8 * z2 * z2 * z2);
      double erfc = Math.exp(-z2) / (z * Math.sqrt(Math.PI)) * series;
      result = 1.0 - erfc;
    }
    return x < 0 ? -result : result;
  }
}
