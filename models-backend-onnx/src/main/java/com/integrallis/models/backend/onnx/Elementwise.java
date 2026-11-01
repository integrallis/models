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

  /**
   * Sine and cosine stay on {@link java.lang.Math}: the graph feeds them accumulated phases in the
   * hundreds of thousands of radians, where only a full range reduction is correct, and on this JDK
   * the Vector API's lane-wise sine was measured slower than the scalar call, not faster.
   */
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
    boolean basic =
        op == Arithmetic.ADD
            || op == Arithmetic.SUB
            || op == Arithmetic.MUL
            || op == Arithmetic.DIV;
    if (basic && a.isFloat() && b.isFloat()) {
      return stridedFloat(op, a, b, shape);
    }
    if (basic
        && a.isFloatingPoint()
        && b.isFloatingPoint()
        && (a.isDouble() || b.isDouble())
        && shape.length > 0) {
      return stridedDouble(op, a.toDouble(), b.toDouble(), shape);
    }
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

  /**
   * Float arithmetic over a broadcast without materializing index maps: the output is walked in
   * row-major order one innermost run at a time, and each run is a contiguous SIMD loop whose
   * operands are either contiguous or a single broadcast value.
   */
  private static Tensor stridedFloat(Arithmetic op, Tensor a, Tensor b, long[] shape) {
    int rank = shape.length;
    float[] fa = a.floats();
    float[] fb = b.floats();
    int count = (int) Tensor.elementCount(shape);
    float[] out = new float[count];
    if (rank == 0) {
      out[0] = apply(op, fa[0], fb[0]);
      return Tensor.ofFloats(shape, out);
    }
    long[] sa = Broadcast.strides(a.shape(), shape);
    long[] sb = Broadcast.strides(b.shape(), shape);
    int inner = (int) shape[rank - 1];
    int innerA = (int) sa[rank - 1];
    int innerB = (int) sb[rank - 1];
    int outer = inner == 0 ? 0 : count / inner;
    long[] counter = new long[rank - 1];
    int oa = 0;
    int ob = 0;
    for (int run = 0; run < outer; run++) {
      int oi = run * inner;
      if (innerA == 1 && innerB == 1) {
        switch (op) {
          case ADD -> Simd.add(out, oi, fa, oa, fb, ob, inner);
          case SUB -> Simd.sub(out, oi, fa, oa, fb, ob, inner);
          case MUL -> Simd.mul(out, oi, fa, oa, fb, ob, inner);
          default -> Simd.div(out, oi, fa, oa, fb, ob, inner);
        }
      } else if (innerA == 1 && innerB == 0) {
        float s = fb[ob];
        switch (op) {
          case ADD -> Simd.addScalar(out, oi, fa, oa, s, inner);
          case SUB -> Simd.addScalar(out, oi, fa, oa, -s, inner);
          case MUL -> Simd.mulScalar(out, oi, fa, oa, s, inner);
          default -> Simd.divScalar(out, oi, fa, oa, s, inner);
        }
      } else if (innerA == 0 && innerB == 1) {
        float s = fa[oa];
        for (int i = 0; i < inner; i++) {
          out[oi + i] = apply(op, s, fb[ob + i]);
        }
      } else {
        float v = apply(op, fa[oa], fb[ob]);
        java.util.Arrays.fill(out, oi, oi + inner, v);
      }
      // Advance the outer odometer.
      for (int axis = rank - 2; axis >= 0; axis--) {
        counter[axis]++;
        oa += (int) sa[axis];
        ob += (int) sb[axis];
        if (counter[axis] < shape[axis]) {
          break;
        }
        oa -= (int) (sa[axis] * counter[axis]);
        ob -= (int) (sb[axis] * counter[axis]);
        counter[axis] = 0;
      }
    }
    return Tensor.ofFloats(shape, out);
  }

  /** Double arithmetic over a broadcast without index maps; the phase arithmetic runs here. */
  private static Tensor stridedDouble(Arithmetic op, Tensor a, Tensor b, long[] shape) {
    int rank = shape.length;
    double[] da = a.doubles();
    double[] db = b.doubles();
    int count = (int) Tensor.elementCount(shape);
    double[] out = new double[count];
    long[] sa = Broadcast.strides(a.shape(), shape);
    long[] sb = Broadcast.strides(b.shape(), shape);
    int inner = (int) shape[rank - 1];
    int stepA = (int) sa[rank - 1];
    int stepB = (int) sb[rank - 1];
    int outer = inner == 0 ? 0 : count / inner;
    long[] counter = new long[rank - 1];
    int oa = 0;
    int ob = 0;
    for (int run = 0; run < outer; run++) {
      int oi = run * inner;
      int ia = oa;
      int ib = ob;
      switch (op) {
        case ADD -> {
          for (int i = 0; i < inner; i++, ia += stepA, ib += stepB) {
            out[oi + i] = da[ia] + db[ib];
          }
        }
        case SUB -> {
          for (int i = 0; i < inner; i++, ia += stepA, ib += stepB) {
            out[oi + i] = da[ia] - db[ib];
          }
        }
        case MUL -> {
          for (int i = 0; i < inner; i++, ia += stepA, ib += stepB) {
            out[oi + i] = da[ia] * db[ib];
          }
        }
        default -> {
          for (int i = 0; i < inner; i++, ia += stepA, ib += stepB) {
            out[oi + i] = da[ia] / db[ib];
          }
        }
      }
      for (int axis = rank - 2; axis >= 0; axis--) {
        counter[axis]++;
        oa += (int) sa[axis];
        ob += (int) sb[axis];
        if (counter[axis] < shape[axis]) {
          break;
        }
        oa -= (int) (sa[axis] * counter[axis]);
        ob -= (int) (sb[axis] * counter[axis]);
        counter[axis] = 0;
      }
    }
    return Tensor.ofDoubles(shape, out);
  }

  private static float apply(Arithmetic op, float x, float y) {
    return switch (op) {
      case ADD -> x + y;
      case SUB -> x - y;
      case MUL -> x * y;
      default -> x / y;
    };
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
    if (a.isFloat() && b.isFloat() && op != Comparison.AND && shape.length > 0) {
      return stridedCompare(op, a, b, shape);
    }
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

  /** Float comparison over a broadcast, one contiguous innermost run at a time. */
  private static Tensor stridedCompare(Comparison op, Tensor a, Tensor b, long[] shape) {
    int rank = shape.length;
    float[] fa = a.floats();
    float[] fb = b.floats();
    int count = (int) Tensor.elementCount(shape);
    byte[] out = new byte[count];
    long[] sa = Broadcast.strides(a.shape(), shape);
    long[] sb = Broadcast.strides(b.shape(), shape);
    int inner = (int) shape[rank - 1];
    int stepA = (int) sa[rank - 1];
    int stepB = (int) sb[rank - 1];
    int outer = inner == 0 ? 0 : count / inner;
    long[] counter = new long[rank - 1];
    int oa = 0;
    int ob = 0;
    for (int run = 0; run < outer; run++) {
      int oi = run * inner;
      int ia = oa;
      int ib = ob;
      switch (op) {
        case LESS -> {
          for (int i = 0; i < inner; i++, ia += stepA, ib += stepB) {
            out[oi + i] = (byte) (fa[ia] < fb[ib] ? 1 : 0);
          }
        }
        case GREATER -> {
          for (int i = 0; i < inner; i++, ia += stepA, ib += stepB) {
            out[oi + i] = (byte) (fa[ia] > fb[ib] ? 1 : 0);
          }
        }
        case GREATER_OR_EQUAL -> {
          for (int i = 0; i < inner; i++, ia += stepA, ib += stepB) {
            out[oi + i] = (byte) (fa[ia] >= fb[ib] ? 1 : 0);
          }
        }
        default -> {
          for (int i = 0; i < inner; i++, ia += stepA, ib += stepB) {
            out[oi + i] = (byte) (fa[ia] == fb[ib] ? 1 : 0);
          }
        }
      }
      for (int axis = rank - 2; axis >= 0; axis--) {
        counter[axis]++;
        oa += (int) sa[axis];
        ob += (int) sb[axis];
        if (counter[axis] < shape[axis]) {
          break;
        }
        oa -= (int) (sa[axis] * counter[axis]);
        ob -= (int) (sb[axis] * counter[axis]);
        counter[axis] = 0;
      }
    }
    return Tensor.ofBytes(OnnxDataType.BOOL, shape, out);
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
      // Maclaurin series: alternating terms summed until they vanish against the partial sum.
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
      // Asymptotic erfc beyond 4, where erfc is already below 1.6e-8.
      double z2 = z * z;
      double series = 1 - 1 / (2 * z2) + 3 / (4 * z2 * z2) - 15 / (8 * z2 * z2 * z2);
      result = 1.0 - Math.exp(-z2) / (z * Math.sqrt(Math.PI)) * series;
    }
    return x < 0 ? -result : result;
  }
}
