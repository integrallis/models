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
package com.integrallis.models.accelerator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class PredictiveKvMapperTest {

  @Test
  void selectsSourceLayersOnHeldOutR2AndLearnsAcrossKvHeads() {
    Samples training = samples(192, 41L);
    Samples selectionHoldout = samples(64, 73L);
    Samples evaluation = samples(48, 97L);

    PredictiveKvMapper mapper =
        PredictiveKvMapper.fit(
            training.sourceKeys(),
            training.sourceValues(),
            training.targetKeys(),
            training.targetValues(),
            selectionHoldout.sourceKeys(),
            selectionHoldout.sourceValues(),
            selectionHoldout.targetKeys(),
            selectionHoldout.targetValues(),
            2,
            2,
            1,
            1.0e-8);

    assertThat(mapper.sourceLayers(0)).containsExactly(1);
    PredictiveKvMapper.MappedCache mapped =
        mapper.map(evaluation.sourceKeys(), evaluation.sourceValues());
    assertThat(
            PredictiveKvMapper.meanPerHeadR2(
                mapped.keys()[0], evaluation.targetKeys()[0], 2, 2))
        .isGreaterThan(0.99999);
    assertThat(
            PredictiveKvMapper.meanPerHeadR2(
                mapped.values()[0], evaluation.targetValues()[0], 2, 2))
        .isGreaterThan(0.99999);
  }

  @Test
  void refusesAnUnderdeterminedProductionFit() {
    Samples training = samples(3, 101L);
    Samples holdout = samples(2, 103L);

    assertThatThrownBy(
            () ->
                PredictiveKvMapper.fit(
                    training.sourceKeys(),
                    training.sourceValues(),
                    training.targetKeys(),
                    training.targetValues(),
                    holdout.sourceKeys(),
                    holdout.sourceValues(),
                    holdout.targetKeys(),
                    holdout.targetValues(),
                    2,
                    2,
                    1,
                    1.0e-8))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("observations per parameter");
  }

  private static Samples samples(int observations, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[][][] sourceKeys = randomLayers(2, observations, 4, random);
    float[][][] sourceValues = randomLayers(2, observations, 4, random);
    float[][][] targetKeys = new float[1][observations][4];
    float[][][] targetValues = new float[1][observations][4];
    for (int row = 0; row < observations; row++) {
      float[] keys = sourceKeys[1][row];
      float[] values = sourceValues[1][row];
      // Target head zero deliberately depends on source head one. A same-head mapper cannot fit it.
      targetKeys[0][row][0] = 11.0f + 2.0f * keys[2] - 0.5f * keys[3];
      targetKeys[0][row][1] = -7.0f + keys[2] + 3.0f * keys[3];
      targetKeys[0][row][2] = 5.0f - keys[0] + 0.25f * keys[1];
      targetKeys[0][row][3] = 3.0f + 0.75f * keys[0] - 2.0f * keys[1];
      targetValues[0][row][0] = -13.0f + values[2] + values[3];
      targetValues[0][row][1] = 17.0f + 0.5f * values[2] - values[3];
      targetValues[0][row][2] = 2.0f + values[0] - 4.0f * values[1];
      targetValues[0][row][3] = -3.0f + 2.0f * values[0] + values[1];
    }
    return new Samples(sourceKeys, sourceValues, targetKeys, targetValues);
  }

  private static float[][][] randomLayers(
      int layers, int observations, int dimensions, SplittableRandom random) {
    float[][][] result = new float[layers][observations][dimensions];
    for (float[][] layer : result) {
      for (float[] row : layer) {
        for (int dimension = 0; dimension < row.length; dimension++) {
          row[dimension] = (float) random.nextDouble(-2.0, 2.0);
        }
      }
    }
    return result;
  }

  private record Samples(
      float[][][] sourceKeys,
      float[][][] sourceValues,
      float[][][] targetKeys,
      float[][][] targetValues) {}
}
