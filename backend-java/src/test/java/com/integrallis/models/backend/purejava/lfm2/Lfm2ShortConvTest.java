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
package com.integrallis.models.backend.purejava.lfm2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * LFM2's short convolution.
 *
 * <p>The reference computes a whole sequence at once by concatenating the state in front of the
 * tokens and running a strided window over it. This implementation steps one token at a time, so
 * the property that matters is that the two agree: {@link
 * #steppingOneTokenAtATimeMatchesAWholeSequence} builds the reference's window explicitly and
 * compares.
 */
@Tag("unit")
class Lfm2ShortConvTest {

  private static final int DIM = 3;
  private static final int L_CACHE = 3;

  @Test
  void oneTokenFromAKnownStateIsTheExpectedThreeTapFilter() {
    // Hand-computed, so the tap order is pinned rather than merely self-consistent. Channel 0:
    // b=2, c=5, x=3 -> bx=6; state [1, 4]; kernel taps [10, 100, 1000] oldest-first.
    // conv = 10*1 + 100*4 + 1000*6 = 6410; out = c * conv = 5 * 6410 = 32050.
    float[] projected = {
      2.0f, 0.0f, 0.0f, // b
      5.0f, 1.0f, 1.0f, // c
      3.0f, 0.0f, 0.0f // x
    };
    float[] kernel = {10.0f, 100.0f, 1000.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float[] state = {1.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float[] gated = new float[DIM];

    Lfm2ShortConv.applyToken(projected, DIM, kernel, L_CACHE, state, gated);

    assertThat(gated[0]).isEqualTo(32_050.0f);
    // The window advanced: the oldest value fell out and this token's gate entered.
    assertThat(state[0]).isEqualTo(4.0f);
    assertThat(state[1]).isEqualTo(6.0f);
  }

  @Test
  void steppingOneTokenAtATimeMatchesAWholeSequence() {
    // The reference's own shape: prepend the state to the token gates, then slide a width-lCache
    // window. Stepping must produce the same outputs, or decoding would diverge from prefill.
    int tokens = 6;
    java.util.Random rng = new java.util.Random(20260929L);
    float[] kernel = new float[L_CACHE * DIM];
    for (int i = 0; i < kernel.length; i++) {
      kernel[i] = (rng.nextFloat() - 0.5f) * 2.0f;
    }
    float[] initialState = new float[(L_CACHE - 1) * DIM];
    for (int i = 0; i < initialState.length; i++) {
      initialState[i] = (rng.nextFloat() - 0.5f);
    }
    float[][] projected = new float[tokens][3 * DIM];
    for (int t = 0; t < tokens; t++) {
      for (int i = 0; i < 3 * DIM; i++) {
        projected[t][i] = (rng.nextFloat() - 0.5f) * 2.0f;
      }
    }

    // Stepped.
    float[] state = initialState.clone();
    float[][] stepped = new float[tokens][DIM];
    for (int t = 0; t < tokens; t++) {
      Lfm2ShortConv.applyToken(projected[t], DIM, kernel, L_CACHE, state, stepped[t]);
    }

    // Reference: one long window per channel, oldest first, sliding by one token.
    for (int channel = 0; channel < DIM; channel++) {
      float[] series = new float[(L_CACHE - 1) + tokens];
      for (int tap = 0; tap < L_CACHE - 1; tap++) {
        series[tap] = initialState[channel * (L_CACHE - 1) + tap];
      }
      for (int t = 0; t < tokens; t++) {
        series[(L_CACHE - 1) + t] = projected[t][channel] * projected[t][2 * DIM + channel];
      }
      for (int t = 0; t < tokens; t++) {
        float conv = 0.0f;
        for (int tap = 0; tap < L_CACHE; tap++) {
          conv += kernel[channel * L_CACHE + tap] * series[t + tap];
        }
        float expected = projected[t][DIM + channel] * conv;
        assertThat(stepped[t][channel])
            .describedAs("token %s channel %s", t, channel)
            .isEqualTo(expected, within(1.0e-5f));
      }
    }
  }

  @Test
  void theStateAfterAStepHoldsTheLastTwoGates() {
    // The state is the only thing carried between tokens, so what it holds is the whole contract.
    float[] kernel = new float[L_CACHE * DIM];
    float[] state = new float[(L_CACHE - 1) * DIM];
    float[] gated = new float[DIM];

    float[] first = new float[3 * DIM];
    first[0] = 3.0f; // b
    first[2 * DIM] = 2.0f; // x -> bx = 6
    Lfm2ShortConv.applyToken(first, DIM, kernel, L_CACHE, state, gated);
    assertThat(state[0]).isEqualTo(0.0f);
    assertThat(state[1]).isEqualTo(6.0f);

    float[] second = new float[3 * DIM];
    second[0] = 5.0f;
    second[2 * DIM] = 2.0f; // bx = 10
    Lfm2ShortConv.applyToken(second, DIM, kernel, L_CACHE, state, gated);
    assertThat(state[0]).isEqualTo(6.0f);
    assertThat(state[1]).isEqualTo(10.0f);
  }

  @Test
  void theGatesAreNotInterchangeable() {
    // b gates BEFORE the convolution and c AFTER it, so swapping them changes the answer. A wrong
    // chunk order type-checks and still produces fluent text, which is why this is pinned.
    float[] kernel = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float[] gated = new float[DIM];

    float[] bFirst = new float[3 * DIM];
    bFirst[0] = 2.0f; // b
    bFirst[DIM] = 5.0f; // c
    bFirst[2 * DIM] = 3.0f; // x
    Lfm2ShortConv.applyToken(bFirst, DIM, kernel, L_CACHE, new float[(L_CACHE - 1) * DIM], gated);
    float withOrder = gated[0]; // c * (b * x) = 5 * 6 = 30

    float[] swapped = new float[3 * DIM];
    swapped[0] = 5.0f; // b and c exchanged
    swapped[DIM] = 2.0f;
    swapped[2 * DIM] = 3.0f;
    Lfm2ShortConv.applyToken(swapped, DIM, kernel, L_CACHE, new float[(L_CACHE - 1) * DIM], gated);

    assertThat(withOrder).isEqualTo(30.0f);
    assertThat(gated[0]).isEqualTo(30.0f);
    // Equal here only because multiplication commutes when the convolution is the identity tap.
    // With a real kernel the two differ, which is the case that matters.
    float[] realKernel = {7.0f, 11.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float[] stateA = {1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float[] stateB = stateA.clone();
    float[] outA = new float[DIM];
    float[] outB = new float[DIM];
    Lfm2ShortConv.applyToken(bFirst, DIM, realKernel, L_CACHE, stateA, outA);
    Lfm2ShortConv.applyToken(swapped, DIM, realKernel, L_CACHE, stateB, outB);
    assertThat(outA[0]).isNotEqualTo(outB[0]);
  }

  @Test
  void aConvolutionWidthBelowTwoIsRefused() {
    assertThatThrownBy(
            () ->
                Lfm2ShortConv.applyToken(
                    new float[3 * DIM], DIM, new float[DIM], 1, new float[DIM], new float[DIM]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("lCache must be at least 2");
  }
}
