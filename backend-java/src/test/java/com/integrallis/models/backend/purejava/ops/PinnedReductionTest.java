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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.vectors.core.PanamaConstants;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * {@link PinnedReduction} must be host-independent and, at 256 bits, unchanged from the original.
 */
class PinnedReductionTest {

  private static float[] fixture(int size, long seed) {
    Random random = new Random(seed);
    float[] values = new float[size];
    for (int index = 0; index < size; index++) {
      // Spread magnitudes so the summation order actually matters; a uniform fixture can hide it.
      values[index] = (random.nextFloat() - 0.5f) * (float) Math.pow(10, random.nextInt(5) - 2);
    }
    return values;
  }

  @ParameterizedTest(name = "{0} elements")
  @ValueSource(ints = {1, 7, 8, 16, 17, 31, 32, 33, 64, 127, 128, 2048, 4096})
  @DisplayName("the pinned dot is bit-identical to VectorUtil.dotProduct on a 256-bit host")
  void pinnedMatchesTheOriginalAt256Bits(int size) {
    // This is the claim that makes the change free: on the default ceiling the pinned shape IS the
    // shape already computed, so no pinned model oracle moves. It can only be asserted where the
    // host actually is 256 bits and fuses, which is the common case and this development machine.
    assumeTrue(
        PanamaConstants.MAX_BITS == 256, "only meaningful where the original computes 8 lanes");
    assumeTrue(PanamaConstants.HAS_FAST_VECTOR_FMA, "the original fuses only with fast vector FMA");
    assumeTrue(PanamaConstants.HAS_FAST_SCALAR_FMA, "and fuses its scalar tail likewise");

    float[] a = fixture(size, 11);
    float[] b = fixture(size, 29);
    assertEquals(
        Float.floatToRawIntBits(VectorUtil.dotProduct(a, 0, b, 0, size)),
        Float.floatToRawIntBits(PinnedReduction.dot(a, 0, b, 0, size)),
        "pinned dot differs from the original at " + size + " elements");
  }

  @Test
  @DisplayName("sumOfSquares is exactly the self-dot rmsNorm asks for")
  void sumOfSquaresIsTheSelfDot() {
    float[] x = fixture(2048, 5);
    assertEquals(
        Float.floatToRawIntBits(PinnedReduction.dot(x, 0, x, 0, 2048)),
        Float.floatToRawIntBits(PinnedReduction.sumOfSquares(x, 0, 2048)));
  }

  @Test
  @DisplayName("offsets are honoured, so a row in the middle of a batch reduces like a row alone")
  void offsetsAreHonoured() {
    float[] whole = fixture(4096, 7);
    int offset = 1024;
    int size = 2048;
    float[] isolated = new float[size];
    System.arraycopy(whole, offset, isolated, 0, size);
    assertEquals(
        Float.floatToRawIntBits(PinnedReduction.sumOfSquares(isolated, 0, size)),
        Float.floatToRawIntBits(PinnedReduction.sumOfSquares(whole, offset, size)));
  }

  @Test
  @DisplayName("the eight-lane fold groups its lanes the way the original's width switch did")
  void theLaneFoldIsTheOriginalsEightLaneCase() {
    // An independent statement of the grouping, so a future edit to the private fold is caught
    // rather than silently redefining "pinned". Sixteen elements is two whole vectors and no tail,
    // which isolates the fold from the unrolled path.
    float[] a = fixture(16, 3);
    float[] b = fixture(16, 4);
    float[] p1 = new float[8];
    float[] p2 = new float[8];
    for (int lane = 0; lane < 8; lane++) {
      // acc1 takes lanes 0-7 then the second whole vector folds into acc1 as well.
      p1[lane] = Math.fma(a[lane], b[lane], 0.0f);
      p2[lane] = Math.fma(a[8 + lane], b[8 + lane], p1[lane]);
    }
    float even = (p2[4] + p2[0]) + (p2[6] + p2[2]);
    float odd = (p2[5] + p2[1]) + (p2[7] + p2[3]);
    assertEquals(
        Float.floatToRawIntBits(even + odd),
        Float.floatToRawIntBits(PinnedReduction.dot(a, 0, b, 0, 16)),
        "the pinned fold no longer groups lanes as the original's 8-lane case");
  }
}
