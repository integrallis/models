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
package com.integrallis.models.backend.purejava.gemma3n;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * AltUp's arithmetic, checked against hand-computed values.
 *
 * <p>Hand-computed on purpose. The alternative -- comparing two of my own implementations -- would
 * agree on a transposed coefficient block, which is the single most likely mistake here and the one
 * that cannot be seen in the output. Every expected number below is written out from the
 * reference's definition rather than produced by running anything.
 */
@Tag("unit")
class Gemma3nAltUpTest {

  private static final float TOLERANCE = 1.0e-6f;

  /** Two streams of three, so a transpose is visible and the arithmetic stays checkable by hand. */
  private static final int ALTUP = 2;

  private static final int DIM = 3;

  @Test
  void theMagnitudeIsTheL2Norm() {
    float[] values = {3.0f, 4.0f, 0.0f, 1.0f};

    assertThat(Gemma3nAltUp.magnitude(values, 0, 3)).isEqualTo(5.0f, within(TOLERANCE));
    // Offset and length are honoured, so one stream can be measured inside the flat state.
    assertThat(Gemma3nAltUp.magnitude(values, 3, 1)).isEqualTo(1.0f, within(TOLERANCE));
  }

  @Test
  void rescalingMakesTheMagnitudeTheTarget() {
    float[] values = {3.0f, 4.0f, 0.0f};

    Gemma3nAltUp.rescaleToMagnitude(values, 0, 3, 10.0f);

    assertThat(Gemma3nAltUp.magnitude(values, 0, 3)).isEqualTo(10.0f, within(TOLERANCE));
    // Direction preserved: it is a scale, not a normalisation to a fixed vector.
    assertThat(values[0] / values[1]).isEqualTo(0.75f, within(TOLERANCE));
  }

  @Test
  void rescalingAZeroStreamLeavesItAloneRatherThanDividingByZero() {
    float[] values = {0.0f, 0.0f, 0.0f};

    Gemma3nAltUp.rescaleToMagnitude(values, 0, 3, 5.0f);

    assertThat(values).containsExactly(0.0f, 0.0f, 0.0f);
  }

  /**
   * The predict step, with coefficients chosen so a transpose gives a different answer.
   *
   * <p>Streams {@code x0 = [1,2,3]}, {@code x1 = [10,20,30]}. Coefficients laid out as the
   * reference lays them out: stream {@code a} reads {@code [a*altup, (a+1)*altup)}, so stream 0
   * uses {@code (0.5, 0.25)} and stream 1 uses {@code (0.1, 0.0)}.
   *
   * <p>pred0 = x0 + 0.5*x0 + 0.25*x1 = [1,2,3] + [0.5,1,1.5] + [2.5,5,7.5] = [4, 8, 12]
   *
   * <p>pred1 = x1 + 0.1*x0 + 0.0*x1 = [10,20,30] + [0.1,0.2,0.3] = [10.1, 20.2, 30.3]
   */
  @Test
  void predictMixesEveryStreamWithItsOwnCoefficientRow() {
    float[] streams = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    float[] coefficients = {0.5f, 0.25f, 0.1f, 0.0f};
    float[] predictions = new float[ALTUP * DIM];

    Gemma3nAltUp.predict(predictions, streams, coefficients, ALTUP, DIM);

    assertThat(predictions[0]).isEqualTo(4.0f, within(TOLERANCE));
    assertThat(predictions[1]).isEqualTo(8.0f, within(TOLERANCE));
    assertThat(predictions[2]).isEqualTo(12.0f, within(TOLERANCE));
    assertThat(predictions[3]).isEqualTo(10.1f, within(TOLERANCE));
    assertThat(predictions[4]).isEqualTo(20.2f, within(TOLERANCE));
    assertThat(predictions[5]).isEqualTo(30.3f, within(TOLERANCE));
  }

  /**
   * The same coefficients read as a transpose give a different answer, so the test above is
   * actually pinning the layout and not merely the shape.
   *
   * <p>Transposed, stream 0 would use {@code (0.5, 0.1)} giving {@code x0 + 0.5*x0 + 0.1*x1 = [2.5,
   * 5, 7.5]}, which is not 4.
   */
  @Test
  void theCoefficientLayoutIsLoadBearing() {
    float[] streams = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    float[] asWritten = {0.5f, 0.25f, 0.1f, 0.0f};
    float[] transposed = {0.5f, 0.1f, 0.25f, 0.0f};
    float[] one = new float[ALTUP * DIM];
    float[] other = new float[ALTUP * DIM];

    Gemma3nAltUp.predict(one, streams, asWritten, ALTUP, DIM);
    Gemma3nAltUp.predict(other, streams, transposed, ALTUP, DIM);

    assertThat(one).isNotEqualTo(other);
  }

  /**
   * The correct step, including the {@code +1} on the coefficients.
   *
   * <p>predictions {@code p0 = [1,1,1]}, {@code p1 = [2,2,2]}; active stream 0; the layer produced
   * {@code [4,4,4]}, so the innovation is {@code [3,3,3]}. Coefficients {@code (0.0, 1.0)} become
   * {@code (1.0, 2.0)}.
   *
   * <p>stream0 = p0 + 3*1.0 = [4,4,4] -- with the +1 missing it would be [1,1,1]
   *
   * <p>stream1 = p1 + 3*2.0 = [8,8,8]
   */
  @Test
  void correctAddsOneToTheCoefficientsBeforeScalingTheInnovation() {
    float[] predictions = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    float[] activated = {4.0f, 4.0f, 4.0f};
    float[] coefficients = {0.0f, 1.0f};
    float[] streams = new float[ALTUP * DIM];

    Gemma3nAltUp.correct(streams, predictions, activated, coefficients, ALTUP, DIM, 0);

    assertThat(streams[0]).isEqualTo(4.0f, within(TOLERANCE));
    assertThat(streams[3]).isEqualTo(8.0f, within(TOLERANCE));
  }

  /** A zero coefficient must carry the innovation at unit weight, not drop it. */
  @Test
  void aZeroCorrectionCoefficientStillAppliesTheInnovation() {
    float[] predictions = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float[] activated = {5.0f, 5.0f, 5.0f};
    float[] streams = new float[ALTUP * DIM];

    Gemma3nAltUp.correct(streams, predictions, activated, new float[ALTUP], ALTUP, DIM, 0);

    // innovation 4, weight 0 + 1 = 1, so the active stream lands exactly on what the layer
    // produced.
    assertThat(streams[0]).isEqualTo(5.0f, within(TOLERANCE));
    assertThat(streams[3]).isEqualTo(5.0f, within(TOLERANCE));
  }

  /** The innovation is measured against the ACTIVE prediction, whichever stream that is. */
  @Test
  void theInnovationIsMeasuredAgainstTheActiveStream() {
    float[] predictions = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    float[] activated = {4.0f, 4.0f, 4.0f};
    float[] coefficients = {0.0f, 0.0f};
    float[] fromStreamZero = new float[ALTUP * DIM];
    float[] fromStreamOne = new float[ALTUP * DIM];

    Gemma3nAltUp.correct(fromStreamZero, predictions, activated, coefficients, ALTUP, DIM, 0);
    Gemma3nAltUp.correct(fromStreamOne, predictions, activated, coefficients, ALTUP, DIM, 1);

    // Against p0 the innovation is 3; against p1 it is 2.
    assertThat(fromStreamZero[0]).isEqualTo(4.0f, within(TOLERANCE));
    assertThat(fromStreamOne[0]).isEqualTo(3.0f, within(TOLERANCE));
  }

  @Test
  void theCorrectedStateMayBeWrittenOverThePredictions() {
    float[] shared = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    float[] separate = new float[ALTUP * DIM];
    float[] predictions = shared.clone();
    float[] activated = {4.0f, 4.0f, 4.0f};
    float[] coefficients = {0.5f, -0.25f};

    Gemma3nAltUp.correct(separate, predictions, activated, coefficients, ALTUP, DIM, 0);
    Gemma3nAltUp.correct(shared, shared, activated, coefficients, ALTUP, DIM, 0);

    // In place must give the same answer: the innovation is re-read per stream, so writing stream 0
    // first would corrupt it for stream 1 if the active prediction were read from the destination.
    assertThat(shared).containsExactly(separate);
  }

  /** The per-layer contribution goes to the inactive streams only. */
  @Test
  void thePerLayerContributionSkipsTheActiveStream() {
    float[] streams = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float[] contribution = {10.0f, 20.0f, 30.0f};

    Gemma3nAltUp.addToInactiveStreams(streams, contribution, ALTUP, DIM, 0);

    assertThat(streams[0]).isEqualTo(1.0f, within(TOLERANCE));
    assertThat(streams[1]).isEqualTo(1.0f, within(TOLERANCE));
    assertThat(streams[3]).isEqualTo(11.0f, within(TOLERANCE));
    assertThat(streams[5]).isEqualTo(31.0f, within(TOLERANCE));
  }

  /**
   * Gaussian sparsity, hand-computed.
   *
   * <p>{@code [1,2,3,4]}: mean 2.5, sample variance {@code (2.25+0.25+0.25+2.25)/3 = 5/3}, so the
   * deviation is {@code sqrt(5/3) = 1.2909944}. With a multiplier of 1 the cutoff is 3.7909944, so
   * only the 4 survives, at {@code 4 - 3.7909944 = 0.2090056}.
   */
  @Test
  void gaussianSparsityKeepsWhatIsAboveTheCutoffAndShiftsItDown() {
    float[] values = {1.0f, 2.0f, 3.0f, 4.0f};

    Gemma3nAltUp.gaussianTopk(values, 4, 1.0f);

    assertThat(values[0]).isZero();
    assertThat(values[1]).isZero();
    assertThat(values[2]).isZero();
    assertThat(values[3]).isEqualTo(0.2090056f, within(1.0e-5f));
  }

  /** The population denominator would give a different cutoff, so the {@code n-1} is pinned. */
  @Test
  void theSampleStandardDeviationDenominatorIsLoadBearing() {
    float[] sample = {1.0f, 2.0f, 3.0f, 4.0f};
    Gemma3nAltUp.gaussianTopk(sample, 4, 1.0f);

    // Population variance would be 5/4, deviation 1.1180, cutoff 3.6180, survivor 0.3819...
    assertThat(sample[3]).isNotEqualTo(0.3819661f);
  }

  /** A large multiplier zeroes everything, which is what a fully sparse layer means. */
  @Test
  void aLargeMultiplierZeroesEverything() {
    float[] values = {1.0f, 2.0f, 3.0f, 4.0f};

    Gemma3nAltUp.gaussianTopk(values, 4, 100.0f);

    assertThat(values).containsExactly(0.0f, 0.0f, 0.0f, 0.0f);
  }

  /**
   * The merge is a mean over all {@code altup} streams, the active one unprojected.
   *
   * <p>active {@code [3,6,9]}, one projected stream {@code [1,2,3]}: the mean over 2 streams is
   * {@code [2, 4, 6]}.
   */
  @Test
  void theMergeAveragesTheActiveStreamWithTheProjectedOnes() {
    float[] merged = new float[DIM];
    float[] active = {3.0f, 6.0f, 9.0f};
    float[] projected = {1.0f, 2.0f, 3.0f};

    Gemma3nAltUp.merge(merged, active, projected, ALTUP, DIM);

    assertThat(merged[0]).isEqualTo(2.0f, within(TOLERANCE));
    assertThat(merged[1]).isEqualTo(4.0f, within(TOLERANCE));
    assertThat(merged[2]).isEqualTo(6.0f, within(TOLERANCE));
  }

  /** The divisor is the stream count, not the projected-stream count. */
  @Test
  void theMergeDividesByTheFullStreamCount() {
    float[] merged = new float[1];
    float[] active = {4.0f};
    float[] projected = {4.0f, 4.0f, 4.0f};

    Gemma3nAltUp.merge(merged, active, projected, 4, 1);

    // 16 / 4 = 4, not 16 / 3.
    assertThat(merged[0]).isEqualTo(4.0f, within(TOLERANCE));
  }

  @Test
  void tanhIsAppliedElementwise() {
    float[] values = {0.0f, 1.0f, -1.0f, 100.0f};

    Gemma3nAltUp.tanhInPlace(values, 4);

    assertThat(values[0]).isZero();
    assertThat(values[1]).isEqualTo(0.7615942f, within(1.0e-6f));
    assertThat(values[2]).isEqualTo(-0.7615942f, within(1.0e-6f));
    assertThat(values[3]).isEqualTo(1.0f, within(1.0e-6f));
  }
}
