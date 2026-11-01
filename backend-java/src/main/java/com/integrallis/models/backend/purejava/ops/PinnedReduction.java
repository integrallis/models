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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Float reductions whose result does not depend on the host.
 *
 * <p>A float reduction's last bits are decided by how many partial sums it keeps and in what order
 * it folds them. Every vectorised reduction in this stack sizes itself from the host — {@code
 * SPECIES_PREFERRED} capped by {@code vectors.maxBits} — so the same model produced <b>different
 * numbers on different machines</b>. In a retrieval score that is invisible; in a transformer it
 * decides which token is selected, so it changes the output text.
 *
 * <p>{@link #dot} is a transcription of {@code PanamaVectorUtilSupport.dotProduct} <b>pinned to
 * eight logical lanes</b>: four accumulators stepping thirty-two elements, a full-vector tail into
 * the first accumulator, the pairwise fold {@code (acc1+acc2) + (acc3+acc4)}, the lane fold {@code
 * ((l4+l0) + (l6+l2)) + ((l5+l1) + (l7+l3))}, then the ragged scalar tail. The {@code length > 16}
 * threshold below which that function stays scalar is reproduced too.
 *
 * <p><b>Why a copy rather than a fix in {@code vectors}.</b> {@code VectorUtil.dotProduct} is the
 * hottest function in the stack and the core similarity kernel for vector search. Pinning it there
 * would slow retrieval on wide hosts to buy a reproducibility guarantee retrieval does not need,
 * and would move recall scores across the whole product. This pins only the model path.
 *
 * <p><b>Why eight lanes.</b> 256 bits is the default ceiling ({@code
 * PanamaConstants.DEFAULT_MAX_BITS}) and therefore what almost every host already computes, so this
 * changes nothing in the common case and converges 128-bit and 512-bit hosts onto that same answer.
 *
 * <p>Fused multiply-add is always genuinely fused. {@code Math.fma} and {@code FloatVector.fma} are
 * exact on every host regardless of hardware support — the fast-FMA flags are a speed choice, never
 * a numeric one — so always fusing is portable rather than a compromise.
 *
 * <p><b>Do not "unify" this with {@code GroupedQueryAttentionKernel}'s pinned fold.</b> That one is
 * a rotate-and-add tree because it pins a different original function. Each pinned reduction must
 * match the specific CPU code its consumers are compared against; making them share one shape would
 * break whichever one was changed.
 */
public final class PinnedReduction {

  /** Eight lanes, on every host. See the class comment. */
  static final VectorSpecies<Float> SPECIES = VectorSpecies.of(float.class, VectorShape.S_256_BIT);

  private static final int LANES = 8;

  // Preserve eight logical lanes without asking a 128-bit host to emulate Float256 objects.
  private static final boolean NARROW_VECTORS = FloatVector.SPECIES_PREFERRED.vectorBitSize() < 256;
  private static final VectorSpecies<Float> HALF_SPECIES = FloatVector.SPECIES_128;

  private PinnedReduction() {}

  /**
   * Sum of squares, host-independent.
   *
   * <p>Exactly {@code dot(x, offset, x, offset, size)}, which is what {@code TensorOps.rmsNorm}
   * asks of {@code VectorUtil.dotProduct}.
   */
  public static float sumOfSquares(float[] x, int offset, int size) {
    return dot(x, offset, x, offset, size);
  }

  /** Dot product of two sub-vectors, host-independent. */
  public static float dot(float[] a, int aOffset, float[] b, int bOffset, int length) {
    int i = 0;
    float result = 0.0f;
    // Reproduced from the original: below this width it never vectorises, and the scalar order is
    // therefore the whole answer.
    if (length > 2 * LANES) {
      int limit = SPECIES.loopBound(length);
      result +=
          NARROW_VECTORS
              ? vectorBody128(a, aOffset, b, bOffset, limit)
              : vectorBody(a, aOffset, b, bOffset, limit);
      i += limit;
    }
    for (; i < length; i++) {
      result = Math.fma(a[aOffset + i], b[bOffset + i], result);
    }
    return result;
  }

  private static float vectorBody(float[] a, int aOffset, float[] b, int bOffset, int limit) {
    FloatVector acc1 = FloatVector.zero(SPECIES);
    FloatVector acc2 = FloatVector.zero(SPECIES);
    FloatVector acc3 = FloatVector.zero(SPECIES);
    FloatVector acc4 = FloatVector.zero(SPECIES);
    int unrolledLimit = limit - 3 * LANES;
    int i = 0;
    for (; i < unrolledLimit; i += 4 * LANES) {
      acc1 = load(a, aOffset + i).fma(load(b, bOffset + i), acc1);
      acc2 = load(a, aOffset + i + LANES).fma(load(b, bOffset + i + LANES), acc2);
      acc3 = load(a, aOffset + i + 2 * LANES).fma(load(b, bOffset + i + 2 * LANES), acc3);
      acc4 = load(a, aOffset + i + 3 * LANES).fma(load(b, bOffset + i + 3 * LANES), acc4);
    }
    // Remaining whole vectors fold into the first accumulator only, as in the original.
    for (; i < limit; i += LANES) {
      acc1 = load(a, aOffset + i).fma(load(b, bOffset + i), acc1);
    }
    // Keep the vector and its fixed lane fold in this primitive-array boundary. Passing the
    // vector to another helper can materialize it when the caller is compiled in a different
    // context; the arithmetic below is the unchanged eight-lane contract.
    FloatVector vector = acc1.add(acc2).add(acc3.add(acc4));
    float even = (vector.lane(4) + vector.lane(0)) + (vector.lane(6) + vector.lane(2));
    float odd = (vector.lane(5) + vector.lane(1)) + (vector.lane(7) + vector.lane(3));
    return even + odd;
  }

  /** Two physical vectors per logical accumulator; every FMA and addition keeps its old order. */
  private static float vectorBody128(float[] a, int aOffset, float[] b, int bOffset, int limit) {
    FloatVector a1l = FloatVector.zero(HALF_SPECIES);
    FloatVector a1h = FloatVector.zero(HALF_SPECIES);
    FloatVector a2l = FloatVector.zero(HALF_SPECIES);
    FloatVector a2h = FloatVector.zero(HALF_SPECIES);
    FloatVector a3l = FloatVector.zero(HALF_SPECIES);
    FloatVector a3h = FloatVector.zero(HALF_SPECIES);
    FloatVector a4l = FloatVector.zero(HALF_SPECIES);
    FloatVector a4h = FloatVector.zero(HALF_SPECIES);
    int unrolledLimit = limit - 3 * LANES;
    int i = 0;
    for (; i < unrolledLimit; i += 4 * LANES) {
      a1l = load128(a, aOffset + i).fma(load128(b, bOffset + i), a1l);
      a1h = load128(a, aOffset + i + 4).fma(load128(b, bOffset + i + 4), a1h);
      a2l = load128(a, aOffset + i + 8).fma(load128(b, bOffset + i + 8), a2l);
      a2h = load128(a, aOffset + i + 12).fma(load128(b, bOffset + i + 12), a2h);
      a3l = load128(a, aOffset + i + 16).fma(load128(b, bOffset + i + 16), a3l);
      a3h = load128(a, aOffset + i + 20).fma(load128(b, bOffset + i + 20), a3h);
      a4l = load128(a, aOffset + i + 24).fma(load128(b, bOffset + i + 24), a4l);
      a4h = load128(a, aOffset + i + 28).fma(load128(b, bOffset + i + 28), a4h);
    }
    for (; i < limit; i += LANES) {
      a1l = load128(a, aOffset + i).fma(load128(b, bOffset + i), a1l);
      a1h = load128(a, aOffset + i + 4).fma(load128(b, bOffset + i + 4), a1h);
    }
    FloatVector low = a1l.add(a2l).add(a3l.add(a4l));
    FloatVector high = a1h.add(a2h).add(a3h.add(a4h));
    float even = (high.lane(0) + low.lane(0)) + (high.lane(2) + low.lane(2));
    float odd = (high.lane(1) + low.lane(1)) + (high.lane(3) + low.lane(3));
    return even + odd;
  }

  private static FloatVector load128(float[] array, int offset) {
    return FloatVector.fromArray(HALF_SPECIES, array, offset);
  }

  private static FloatVector load(float[] array, int offset) {
    return FloatVector.fromArray(SPECIES, array, offset);
  }
}
