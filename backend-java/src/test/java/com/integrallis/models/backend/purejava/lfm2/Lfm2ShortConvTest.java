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
  void theCentredWindowIsolatesEachTapWithSymmetricPadding() {
    // Three positions, one channel's worth of structure per position. b = 1 and c = 1 throughout
    // so the gating is the identity and the taps are the only thing under test; x carries the
    // position so a misaligned tap is visible as the wrong token.
    int sequenceLength = 3;
    float[] projected = new float[sequenceLength * 3 * DIM];
    for (int position = 0; position < sequenceLength; position++) {
      int base = position * 3 * DIM;
      for (int channel = 0; channel < DIM; channel++) {
        projected[base + channel] = 1.0f;
        projected[base + DIM + channel] = 1.0f;
        projected[base + 2 * DIM + channel] = 10.0f * (position + 1);
      }
    }
    float[] gated = new float[sequenceLength * DIM];

    // Tap 0 is the furthest-back column of the padded window, so it selects the previous token --
    // and at position 0 it selects the left zero padding rather than wrapping or repeating.
    Lfm2ShortConv.applyCentered(projected, sequenceLength, DIM, taps(1, 0, 0), L_CACHE, gated);
    assertThat(gated[0 * DIM]).isEqualTo(0.0f);
    assertThat(gated[1 * DIM]).isEqualTo(10.0f);
    assertThat(gated[2 * DIM]).isEqualTo(20.0f);

    // The centre tap is this token.
    Lfm2ShortConv.applyCentered(projected, sequenceLength, DIM, taps(0, 1, 0), L_CACHE, gated);
    assertThat(gated[0 * DIM]).isEqualTo(10.0f);
    assertThat(gated[1 * DIM]).isEqualTo(20.0f);
    assertThat(gated[2 * DIM]).isEqualTo(30.0f);

    // The last tap is the NEXT token: the one thing the causal operator cannot express, and the
    // reason a bidirectional file needs this method at all. The right zero padding ends it.
    Lfm2ShortConv.applyCentered(projected, sequenceLength, DIM, taps(0, 0, 1), L_CACHE, gated);
    assertThat(gated[0 * DIM]).isEqualTo(20.0f);
    assertThat(gated[1 * DIM]).isEqualTo(30.0f);
    assertThat(gated[2 * DIM]).isEqualTo(0.0f);
  }

  @Test
  void theCentredWindowDisagreesWithTheCausalStepAndThatIsThePoint() {
    // Same inputs through both operators. A test that only checked "centred produces numbers"
    // would pass against the causal operator too, which is exactly the failure that shipped: the
    // decoder ran these files and returned worst-probe cosine 0.03 instead of an error.
    int sequenceLength = 4;
    float[] projected = new float[sequenceLength * 3 * DIM];
    for (int index = 0; index < projected.length; index++) {
      projected[index] = (index % 7) * 0.5f - 1.0f;
    }
    float[] kernel = taps(0.25f, 0.5f, -0.75f);

    float[] centred = new float[sequenceLength * DIM];
    Lfm2ShortConv.applyCentered(projected, sequenceLength, DIM, kernel, L_CACHE, centred);

    float[] causal = new float[sequenceLength * DIM];
    float[] state = new float[(L_CACHE - 1) * DIM];
    float[] oneToken = new float[3 * DIM];
    float[] step = new float[DIM];
    for (int position = 0; position < sequenceLength; position++) {
      System.arraycopy(projected, position * 3 * DIM, oneToken, 0, 3 * DIM);
      Lfm2ShortConv.applyToken(oneToken, DIM, kernel, L_CACHE, state, step);
      System.arraycopy(step, 0, causal, position * DIM, DIM);
    }

    assertThat(centred)
        .describedAs("a centred window and a backward-only window are different operators")
        .isNotEqualTo(causal);

    // The property, not an index: changing the LAST token must move the second-to-last output
    // under the centred window and must not move it under the causal one. Asserting a particular
    // index differs instead let a fixture that happened to produce 0.0 and -0.0 there decide the
    // test.
    float[] perturbed = projected.clone();
    int lastX = (sequenceLength - 1) * 3 * DIM + 2 * DIM;
    perturbed[lastX] += 3.0f;

    float[] centredAfter = new float[sequenceLength * DIM];
    Lfm2ShortConv.applyCentered(perturbed, sequenceLength, DIM, kernel, L_CACHE, centredAfter);
    assertThat(centredAfter[(sequenceLength - 2) * DIM])
        .describedAs("the centred window at n-2 sees token n-1")
        .isNotEqualTo(centred[(sequenceLength - 2) * DIM]);

    float[] causalAfter = new float[sequenceLength * DIM];
    float[] perturbedState = new float[(L_CACHE - 1) * DIM];
    for (int position = 0; position < sequenceLength; position++) {
      System.arraycopy(perturbed, position * 3 * DIM, oneToken, 0, 3 * DIM);
      Lfm2ShortConv.applyToken(oneToken, DIM, kernel, L_CACHE, perturbedState, step);
      System.arraycopy(step, 0, causalAfter, position * DIM, DIM);
    }
    assertThat(causalAfter[(sequenceLength - 2) * DIM])
        .describedAs("the causal window at n-2 cannot see token n-1")
        .isEqualTo(causal[(sequenceLength - 2) * DIM]);
  }

  @Test
  void anEvenConvolutionWidthHasNoCentreAndIsRefused() {
    assertThatThrownBy(
            () ->
                Lfm2ShortConv.applyCentered(
                    new float[4 * 3 * DIM], 4, DIM, new float[4 * DIM], 4, new float[4 * DIM]))
        .describedAs(
            "the reference pads by (l_cache - 1) / 2 on each side, which only spans the window"
                + " for an odd width")
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("odd l_cache");
  }

  private static float[] taps(float first, float second, float third) {
    float[] kernel = new float[L_CACHE * DIM];
    for (int channel = 0; channel < DIM; channel++) {
      kernel[channel * L_CACHE] = first;
      kernel[channel * L_CACHE + 1] = second;
      kernel[channel * L_CACHE + 2] = third;
    }
    return kernel;
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
