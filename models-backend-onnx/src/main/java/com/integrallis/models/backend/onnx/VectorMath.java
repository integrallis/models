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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Single-precision exp, tanh, logistic and erf over whole arrays, on the Vector API.
 *
 * <p>These are onnxruntime's own approximations (MLAS {@code MlasComputeExpVector}, {@code
 * MlasTanhKernel}, {@code MlasLogisticKernel}, {@code MlasErfKernel}), translated with their
 * constants digit for digit, so the executor computes what the reference runtime computes rather
 * than a double-precision {@link Math} result rounded to float. They replace one virtual {@code
 * DoubleUnaryOperator} call and one double-precision libm call per element, which measured 18x to
 * 39x slower than onnxruntime on the Paradee graph.
 *
 * <p>Every multiply-add is fused, in the vector body and in the scalar tail alike, so an element's
 * result does not depend on whether it fell in a full vector or in the tail.
 */
final class VectorMath {
  private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
  private static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;

  private VectorMath() {}

  /** {@code a * b + c}, fused, with a scalar addend. */
  private static FloatVector fma(FloatVector a, FloatVector b, float c) {
    return a.fma(b, FloatVector.broadcast(F, c));
  }

  /** {@code a * k + c}, fused, with a scalar multiplier. */
  private static FloatVector fmaScaled(FloatVector a, float k, FloatVector c) {
    return a.fma(FloatVector.broadcast(F, k), c);
  }

  // ---- exp: MlasExpConstants (compute.cpp) ----
  private static final float EXP_LOWER = -103.9720840454f;
  private static final float EXP_UPPER = 88.7762626647950f;
  private static final float EXP_BIAS = 12582912.f; // MLAS_ROUNDING_BIAS_MAGIC
  private static final float EXP_LOG2_RECIPROCAL = 1.44269504088896341f;
  private static final float EXP_LOG2_HIGH = -6.93145752e-1f;
  private static final float EXP_LOG2_LOW = -1.42860677e-6f;
  private static final float EXP_P0 = 0x1.694000p-10f;
  private static final float EXP_P1 = 0x1.125edcp-7f;
  private static final float EXP_P2 = 0x1.555b5ap-5f;
  private static final float EXP_P3 = 0x1.555450p-3f;
  private static final float EXP_P4 = 0x1.fffff6p-2f;
  private static final float EXP_P56 = 0x1.000000p+0f;
  private static final int EXP_MIN_EXPONENT = 0xC1000000;
  private static final int EXP_MAX_EXPONENT = 0x3F800000;

  /** {@code out[i] = exp(in[i])}. */
  static void exp(float[] in, float[] out, int n) {
    exp(in, 0, out, 0, n);
  }

  /**
   * As {@link #exp(float[], float[], int)} over {@code in[from..from+n)} into {@code
   * out[to..to+n)}.
   */
  static void exp(float[] in, int from, float[] out, int to, int n) {
    int i = 0;
    for (int bound = F.loopBound(n); i < bound; i += F.length()) {
      expVector(FloatVector.fromArray(F, in, from + i)).intoArray(out, to + i);
    }
    for (; i < n; i++) {
      out[to + i] = expScalar(in[from + i]);
    }
  }

  private static FloatVector expVector(FloatVector v) {
    v = v.max(EXP_LOWER).min(EXP_UPPER);
    FloatVector biased = v.fma(EXP_LOG2_RECIPROCAL, EXP_BIAS);
    FloatVector m = biased.sub(EXP_BIAS);
    v = fmaScaled(m, EXP_LOG2_HIGH, v);
    v = fmaScaled(m, EXP_LOG2_LOW, v);
    IntVector overflow =
        ((IntVector) biased.reinterpretShape(I, 0)).lanewise(VectorOperators.LSHL, 23);
    IntVector normal = overflow.min(EXP_MAX_EXPONENT).max(EXP_MIN_EXPONENT);
    overflow = overflow.sub(normal).add(EXP_MAX_EXPONENT);
    normal = normal.add(EXP_MAX_EXPONENT);
    FloatVector p = FloatVector.broadcast(F, EXP_P0);
    p = fma(p, v, EXP_P1);
    p = fma(p, v, EXP_P2);
    p = fma(p, v, EXP_P3);
    p = fma(p, v, EXP_P4);
    p = fma(p, v, EXP_P56);
    FloatVector overflowScale = (FloatVector) overflow.reinterpretShape(F, 0);
    v = v.mul(overflowScale);
    p = p.fma(v, overflowScale);
    return p.mul((FloatVector) normal.reinterpretShape(F, 0));
  }

  static float expScalar(float x) {
    float v = Math.min(Math.max(x, EXP_LOWER), EXP_UPPER);
    float biased = Math.fma(v, EXP_LOG2_RECIPROCAL, EXP_BIAS);
    float m = biased - EXP_BIAS;
    v = Math.fma(m, EXP_LOG2_HIGH, v);
    v = Math.fma(m, EXP_LOG2_LOW, v);
    int overflow = Float.floatToRawIntBits(biased) << 23;
    int normal = Math.max(Math.min(overflow, EXP_MAX_EXPONENT), EXP_MIN_EXPONENT);
    overflow = overflow - normal + EXP_MAX_EXPONENT;
    normal = normal + EXP_MAX_EXPONENT;
    float p = EXP_P0;
    p = Math.fma(p, v, EXP_P1);
    p = Math.fma(p, v, EXP_P2);
    p = Math.fma(p, v, EXP_P3);
    p = Math.fma(p, v, EXP_P4);
    p = Math.fma(p, v, EXP_P56);
    float overflowScale = Float.intBitsToFloat(overflow);
    v = v * overflowScale;
    p = Math.fma(p, v, overflowScale);
    return p * Float.intBitsToFloat(normal);
  }

  // ---- tanh: MlasTanhConstants (tanh.cpp) ----
  private static final float TANH_LOWER = -9.0f;
  private static final float TANH_UPPER = 9.0f;
  private static final float TANH_A13 = -2.76076847742355e-16f;
  private static final float TANH_A11 = 2.00018790482477e-13f;
  private static final float TANH_A9 = -8.60467152213735e-11f;
  private static final float TANH_A7 = 5.12229709037114e-08f;
  private static final float TANH_A5 = 1.48572235717979e-05f;
  private static final float TANH_A3 = 6.37261928875436e-04f;
  private static final float TANH_A1 = 4.89352455891786e-03f;
  private static final float TANH_B6 = 1.19825839466702e-06f;
  private static final float TANH_B4 = 1.18534705686654e-04f;
  private static final float TANH_B2 = 2.26843463243900e-03f;
  private static final float TANH_B0 = 4.89352518554385e-03f;

  /** {@code out[i] = tanh(in[i])}. */
  static void tanh(float[] in, float[] out, int n) {
    tanh(in, 0, out, 0, n);
  }

  /**
   * As {@link #tanh(float[], float[], int)} over {@code in[from..from+n)} into {@code
   * out[to..to+n)}.
   */
  static void tanh(float[] in, int from, float[] out, int to, int n) {
    int i = 0;
    for (int bound = F.loopBound(n); i < bound; i += F.length()) {
      FloatVector v = FloatVector.fromArray(F, in, from + i).max(TANH_LOWER).min(TANH_UPPER);
      FloatVector s = v.mul(v);
      FloatVector p = s.fma(TANH_A13, TANH_A11);
      p = fma(p, s, TANH_A9);
      p = fma(p, s, TANH_A7);
      p = fma(p, s, TANH_A5);
      p = fma(p, s, TANH_A3);
      p = fma(p, s, TANH_A1);
      p = p.mul(v);
      FloatVector q = s.fma(TANH_B6, TANH_B4);
      q = fma(q, s, TANH_B2);
      q = fma(q, s, TANH_B0);
      p.div(q).intoArray(out, to + i);
    }
    for (; i < n; i++) {
      out[to + i] = tanhScalar(in[from + i]);
    }
  }

  static float tanhScalar(float x) {
    // MLAS's scalar tail clamps with comparisons so a NaN input stays NaN.
    float v = x < TANH_LOWER ? TANH_LOWER : x;
    v = v > TANH_UPPER ? TANH_UPPER : v;
    float s = v * v;
    float p = Math.fma(s, TANH_A13, TANH_A11);
    p = Math.fma(p, s, TANH_A9);
    p = Math.fma(p, s, TANH_A7);
    p = Math.fma(p, s, TANH_A5);
    p = Math.fma(p, s, TANH_A3);
    p = Math.fma(p, s, TANH_A1);
    p = p * v;
    float q = Math.fma(s, TANH_B6, TANH_B4);
    q = Math.fma(q, s, TANH_B2);
    q = Math.fma(q, s, TANH_B0);
    return p / q;
  }

  // ---- logistic: MlasLogisticConstants (logistic.cpp) ----
  private static final float LOGISTIC_LOWER = -18.0f;
  private static final float LOGISTIC_UPPER = 18.0f;
  private static final float LOGISTIC_A9 = 4.37031012579801e-11f;
  private static final float LOGISTIC_A7 = 1.15627324459942e-07f;
  private static final float LOGISTIC_A5 = 6.08574864600143e-05f;
  private static final float LOGISTIC_A3 = 8.51377133304701e-03f;
  private static final float LOGISTIC_A1 = 2.48287947061529e-01f;
  private static final float LOGISTIC_B10 = 6.10247389755681e-13f;
  private static final float LOGISTIC_B8 = 5.76102136993427e-09f;
  private static final float LOGISTIC_B6 = 6.29106785017040e-06f;
  private static final float LOGISTIC_B4 = 1.70198817374094e-03f;
  private static final float LOGISTIC_B2 = 1.16817656904453e-01f;
  private static final float LOGISTIC_B0 = 9.93151921023180e-01f;

  /** {@code out[i] = 1 / (1 + exp(-in[i]))}. */
  static void logistic(float[] in, float[] out, int n) {
    logistic(in, 0, out, 0, n);
  }

  /**
   * As {@link #logistic(float[], float[], int)} over {@code in[from..from+n)} into {@code
   * out[to..to+n)}.
   */
  static void logistic(float[] in, int from, float[] out, int to, int n) {
    int i = 0;
    for (int bound = F.loopBound(n); i < bound; i += F.length()) {
      FloatVector v =
          FloatVector.fromArray(F, in, from + i).max(LOGISTIC_LOWER).min(LOGISTIC_UPPER);
      FloatVector s = v.mul(v);
      FloatVector p = s.fma(LOGISTIC_A9, LOGISTIC_A7);
      p = fma(p, s, LOGISTIC_A5);
      p = fma(p, s, LOGISTIC_A3);
      p = fma(p, s, LOGISTIC_A1);
      p = p.mul(v);
      FloatVector q = s.fma(LOGISTIC_B10, LOGISTIC_B8);
      q = fma(q, s, LOGISTIC_B6);
      q = fma(q, s, LOGISTIC_B4);
      q = fma(q, s, LOGISTIC_B2);
      q = fma(q, s, LOGISTIC_B0);
      p.div(q).add(0.5f).max(0.0f).min(1.0f).intoArray(out, to + i);
    }
    for (; i < n; i++) {
      out[to + i] = logisticScalar(in[from + i]);
    }
  }

  static float logisticScalar(float x) {
    float v = x < LOGISTIC_LOWER ? LOGISTIC_LOWER : x;
    v = v > LOGISTIC_UPPER ? LOGISTIC_UPPER : v;
    float s = v * v;
    float p = Math.fma(s, LOGISTIC_A9, LOGISTIC_A7);
    p = Math.fma(p, s, LOGISTIC_A5);
    p = Math.fma(p, s, LOGISTIC_A3);
    p = Math.fma(p, s, LOGISTIC_A1);
    p = p * v;
    float q = Math.fma(s, LOGISTIC_B10, LOGISTIC_B8);
    q = Math.fma(q, s, LOGISTIC_B6);
    q = Math.fma(q, s, LOGISTIC_B4);
    q = Math.fma(q, s, LOGISTIC_B2);
    q = Math.fma(q, s, LOGISTIC_B0);
    return Math.min(Math.max(p / q + 0.5f, 0.0f), 1.0f);
  }

  // ---- erf: MlasErfConstants (erf.cpp) ----
  private static final float ERF_UPPER_ABS = 3.925f;
  private static final float ERF_SPLIT = 0.921875f;
  private static final float ERF_SMALL_P0 = -5.99104969e-4f;
  private static final float ERF_SMALL_P1 = 4.99339588e-3f;
  private static final float ERF_SMALL_P2 = -2.67667342e-2f;
  private static final float ERF_SMALL_P3 = 1.12818025e-1f;
  private static final float ERF_SMALL_P4 = -3.76124859e-1f;
  private static final float ERF_SMALL_P5_MINUS_ONE = 1.28379151e-1f;
  private static final float ERF_BIG_P0 = 1.72948930e-5f;
  private static final float ERF_BIG_P1 = -3.83208680e-4f;
  private static final float ERF_BIG_P2 = 3.88393435e-3f;
  private static final float ERF_BIG_P3 = -2.42545605e-2f;
  private static final float ERF_BIG_P4 = 1.06777847e-1f;
  private static final float ERF_BIG_P5 = 6.34846687e-1f;
  private static final float ERF_BIG_P6_MINUS_ONE = 1.28717512e-1f;
  private static final float ERF_EXP_LOWER = -88.3762626647949f;
  private static final float ERF_EXP_LOG2_RECIPROCAL = 1.44269504088896341f;
  private static final float ERF_EXP_LOG2_HI = -6.93145752e-1f;
  private static final float ERF_EXP_LOG2_LO = -1.42860677e-6f;
  private static final float ERF_EXP_P0 = 1.38319808e-3f;
  private static final float ERF_EXP_P1 = 8.37550033e-3f;
  private static final float ERF_EXP_P2 = 4.16689515e-2f;
  private static final float ERF_EXP_P3 = 1.66664466e-1f;
  private static final float ERF_EXP_P4 = 4.99999851e-1f;
  private static final float ERF_EXP_P5 = 1.00000000e+0f;
  private static final float ERF_EXP_P6 = 1.00000000e+0f;
  private static final float ERF_EXP_C = 1.25829120e+7f;

  /** {@code out[i] = erf(in[i])}. */
  static void erf(float[] in, float[] out, int n) {
    erf(in, 0, out, 0, n);
  }

  /**
   * As {@link #erf(float[], float[], int)} over {@code in[from..from+n)} into {@code
   * out[to..to+n)}.
   */
  static void erf(float[] in, int from, float[] out, int to, int n) {
    int i = 0;
    for (int bound = F.loopBound(n); i < bound; i += F.length()) {
      FloatVector value = FloatVector.fromArray(F, in, from + i);
      FloatVector abs = value.abs().min(ERF_UPPER_ABS);
      FloatVector square = abs.mul(abs);
      FloatVector small = FloatVector.broadcast(F, ERF_SMALL_P0);
      small = fma(small, square, ERF_SMALL_P1);
      small = fma(small, square, ERF_SMALL_P2);
      small = fma(small, square, ERF_SMALL_P3);
      small = fma(small, square, ERF_SMALL_P4);
      small = fma(small, square, ERF_SMALL_P5_MINUS_ONE);
      small = small.fma(abs, abs);
      VectorMask<Float> big = abs.compare(VectorOperators.GT, ERF_SPLIT);
      FloatVector r = FloatVector.broadcast(F, ERF_BIG_P0);
      r = fma(r, abs, ERF_BIG_P1);
      r = fma(r, abs, ERF_BIG_P2);
      r = fma(r, abs, ERF_BIG_P3);
      r = fma(r, abs, ERF_BIG_P4);
      r = fma(r, abs, ERF_BIG_P5);
      r = fma(r, abs, ERF_BIG_P6_MINUS_ONE);
      r = r.fma(abs, abs);
      r = r.neg().max(ERF_EXP_LOWER);
      FloatVector k = r.fma(ERF_EXP_LOG2_RECIPROCAL, ERF_EXP_C).sub(ERF_EXP_C);
      FloatVector fx = fmaScaled(k, ERF_EXP_LOG2_HI, r);
      fx = fmaScaled(k, ERF_EXP_LOG2_LO, fx);
      FloatVector y = FloatVector.broadcast(F, ERF_EXP_P0);
      y = fma(y, fx, ERF_EXP_P1);
      y = fma(y, fx, ERF_EXP_P2);
      y = fma(y, fx, ERF_EXP_P3);
      y = fma(y, fx, ERF_EXP_P4);
      y = fma(y, fx, ERF_EXP_P5);
      y = fma(y, fx, ERF_EXP_P6);
      IntVector exponent =
          ((IntVector) k.convert(VectorOperators.F2I, 0))
              .add(127)
              .lanewise(VectorOperators.LSHL, 23);
      y = y.mul((FloatVector) exponent.reinterpretShape(F, 0));
      FloatVector bigResult = FloatVector.broadcast(F, 1.0f).sub(y);
      FloatVector magnitude = small.blend(bigResult, big);
      // Restore the sign by copying the input's sign bit, as MLAS does, so erf(-0) = -0.
      IntVector sign = ((IntVector) value.reinterpretShape(I, 0)).and(0x80000000);
      IntVector bits = ((IntVector) magnitude.reinterpretShape(I, 0)).or(sign);
      ((FloatVector) bits.reinterpretShape(F, 0)).intoArray(out, to + i);
    }
    for (; i < n; i++) {
      out[to + i] = erfScalar(in[from + i]);
    }
  }

  static float erfScalar(float value) {
    float abs = Math.min(Math.abs(value), ERF_UPPER_ABS);
    if (abs > ERF_SPLIT) {
      float r = ERF_BIG_P0;
      r = Math.fma(r, abs, ERF_BIG_P1);
      r = Math.fma(r, abs, ERF_BIG_P2);
      r = Math.fma(r, abs, ERF_BIG_P3);
      r = Math.fma(r, abs, ERF_BIG_P4);
      r = Math.fma(r, abs, ERF_BIG_P5);
      r = Math.fma(r, abs, ERF_BIG_P6_MINUS_ONE);
      r = Math.fma(r, abs, abs);
      r = Math.max(-r, ERF_EXP_LOWER);
      float k = Math.fma(r, ERF_EXP_LOG2_RECIPROCAL, ERF_EXP_C) - ERF_EXP_C;
      float fx = Math.fma(k, ERF_EXP_LOG2_HI, r);
      fx = Math.fma(k, ERF_EXP_LOG2_LO, fx);
      float y = ERF_EXP_P0;
      y = Math.fma(y, fx, ERF_EXP_P1);
      y = Math.fma(y, fx, ERF_EXP_P2);
      y = Math.fma(y, fx, ERF_EXP_P3);
      y = Math.fma(y, fx, ERF_EXP_P4);
      y = Math.fma(y, fx, ERF_EXP_P5);
      y = Math.fma(y, fx, ERF_EXP_P6);
      y = y * Float.intBitsToFloat(((int) k + 127) << 23);
      return Math.copySign(1.0f - y, value);
    }
    float square = abs * abs;
    float r = ERF_SMALL_P0;
    r = Math.fma(r, square, ERF_SMALL_P1);
    r = Math.fma(r, square, ERF_SMALL_P2);
    r = Math.fma(r, square, ERF_SMALL_P3);
    r = Math.fma(r, square, ERF_SMALL_P4);
    r = Math.fma(r, square, ERF_SMALL_P5_MINUS_ONE);
    r = Math.fma(r, abs, abs);
    return Math.copySign(r, value);
  }
}
