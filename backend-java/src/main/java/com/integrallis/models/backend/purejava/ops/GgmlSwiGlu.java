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
package com.integrallis.models.backend.purejava.ops;

import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** The vision decoder's fixed eight-lane exponential and two-stage SwiGLU rounding. */
final class GgmlSwiGlu {
  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;

  private GgmlSwiGlu() {}

  static void apply(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int size) {
    Objects.checkFromIndexSize(outOffset, size, out.length);
    Objects.checkFromIndexSize(gateOffset, size, gate.length);
    Objects.checkFromIndexSize(upOffset, size, up.length);
    rejectForwardAlias(out, outOffset, gate, gateOffset, size);
    rejectForwardAlias(out, outOffset, up, upOffset, size);
    var zero = FloatVector.zero(SPECIES);
    var one = FloatVector.broadcast(SPECIES, 1f);
    var rounding = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x4b400000));
    var log2e = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x3fb8aa3b));
    var ln2High = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x3f317200));
    var ln2Low = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x35bfbe8e));
    var c0 = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x3c072010));
    var c1 = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x3d2b9f17));
    var c2 = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x3e2aaf33));
    var c3 = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x3efffedb));
    var c4 = FloatVector.broadcast(SPECIES, Float.intBitsToFloat(0x3f7ffff6));
    int limit = SPECIES.loopBound(size);
    int i = 0;
    for (; i < limit; i += SPECIES.length()) {
      var g = FloatVector.fromArray(SPECIES, gate, gateOffset + i);
      var x = zero.sub(g);
      var z = x.fma(log2e, rounding);
      var n = z.sub(rounding);
      var b = n.neg().fma(ln2Low, n.neg().fma(ln2High, x));
      var exponent = z.reinterpretAsInts().lanewise(VectorOperators.LSHL, 23);
      var k = exponent.add(0x3f800000).reinterpretAsFloats();
      var square = b.mul(b);
      var polynomial = c0.fma(b, c1).fma(square, c2.fma(b, c3)).fma(square, c4.mul(b));
      var exp = polynomial.fma(k, k);
      var special = n.abs().compare(VectorOperators.GT, 126f);
      if (special.anyTrue()) {
        var sign = n.compare(VectorOperators.LE, 0f).toVector().reinterpretAsInts().and(0x82000000);
        var scale1 = sign.add(0x7f000000).reinterpretAsFloats();
        var scale2 = exponent.sub(sign).reinterpretAsFloats();
        var adjusted = polynomial.fma(scale2, scale2).mul(scale1);
        exp =
            exp.blend(adjusted, special)
                .blend(scale1.mul(scale1), n.abs().compare(VectorOperators.GT, 192f));
      }
      g.div(one.add(exp))
          .mul(FloatVector.fromArray(SPECIES, up, upOffset + i))
          .intoArray(out, outOffset + i);
    }
    // ggml's ragged tail calls scalar expf, rather than its vector polynomial.
    for (; i < size; i++) {
      float g = gate[gateOffset + i];
      float silu = g / (1f + (float) Math.exp(-g));
      out[outOffset + i] = silu * up[upOffset + i];
    }
  }

  private static void rejectForwardAlias(
      float[] out, int outOffset, float[] input, int inputOffset, int size) {
    if (out == input && outOffset > inputOffset && outOffset - inputOffset < size)
      throw new IllegalArgumentException("SwiGLU output must not overlap unread input");
  }
}
