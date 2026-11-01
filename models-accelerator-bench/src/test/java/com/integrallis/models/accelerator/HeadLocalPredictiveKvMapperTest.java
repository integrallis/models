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

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class HeadLocalPredictiveKvMapperTest {

  @Test
  void selectsLayersAndMapsOnlyTheArchitectureAlignedSourceHead() {
    float[][][] fit = source(96, 31L);
    float[][][] selection = source(40, 47L);
    float[][][] fitTargets = targets(fit);
    float[][][] selectionTargets = targets(selection);

    HeadLocalPredictiveKvMapper mapper =
        HeadLocalPredictiveKvMapper.fit(
            fit,
            fit,
            fitTargets,
            fitTargets,
            selection,
            selection,
            selectionTargets,
            selectionTargets,
            2,
            2,
            1,
            1.0e-8);

    assertThat(mapper.sourceLayers(0)).containsExactly(1);
    assertThat(mapper.sourceLayers(1)).containsExactly(0);
    HeadLocalPredictiveKvMapper.MappedCache actual = mapper.map(selection, selection);
    assertThat(RidgeMapper.relativeL2(selectionTargets[0], actual.keys()[0]))
        .isLessThan(1.0e-5);
    assertThat(RidgeMapper.relativeL2(selectionTargets[1], actual.keys()[1]))
        .isLessThan(1.0e-5);
  }

  @Test
  void attentionWeightedFitDoesNotLetIrrelevantTokensDistortTheMap() {
    float[][][] fit = source(96, 131L);
    float[][][] selection = source(40, 149L);
    float[][][] fitTargets = targets(fit);
    float[][][] expected = targets(selection);
    double[][][] keySensitivity = sensitivities(2, 2, 96);
    double[][][] valueSensitivity = sensitivities(2, 2, 96);
    for (int observation = 64; observation < 96; observation++) {
      fitTargets[0][observation][0] = 10_000.0f + observation;
      keySensitivity[0][0][observation] = 0.0;
      valueSensitivity[0][0][observation] = 0.0;
    }

    HeadLocalPredictiveKvMapper mapper =
        HeadLocalPredictiveKvMapper.fit(
            fit,
            fit,
            fitTargets,
            fitTargets,
            selection,
            selection,
            expected,
            expected,
            2,
            2,
            1,
            1.0e-8,
            new HeadLocalPredictiveKvMapper.AttentionSensitivities(
                keySensitivity, valueSensitivity));

    assertThat(RidgeMapper.relativeL2(expected[0], mapper.map(selection, selection).keys()[0]))
        .isLessThan(1.0e-5);
  }

  private static float[][][] source(int observations, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[][][] result = new float[2][observations][4];
    for (int layer = 0; layer < result.length; layer++) {
      for (int observation = 0; observation < observations; observation++) {
        for (int dimension = 0; dimension < result[layer][observation].length; dimension++) {
          result[layer][observation][dimension] = (float) random.nextDouble(-2.0, 2.0);
        }
      }
    }
    return result;
  }

  private static float[][][] targets(float[][][] source) {
    float[][][] result = new float[2][source[0].length][4];
    for (int observation = 0; observation < source[0].length; observation++) {
      for (int head = 0; head < 2; head++) {
        int offset = head * 2;
        result[0][observation][offset] = 3.0f * source[1][observation][offset] + 7.0f;
        result[0][observation][offset + 1] =
            -2.0f * source[1][observation][offset + 1] - 5.0f;
        result[1][observation][offset] = -4.0f * source[0][observation][offset] + 11.0f;
        result[1][observation][offset + 1] =
            0.5f * source[0][observation][offset + 1] - 13.0f;
      }
    }
    return result;
  }

  private static double[][][] sensitivities(int layers, int heads, int observations) {
    double[][][] result = new double[layers][heads][observations];
    for (double[][] layer : result) {
      for (double[] head : layer) {
        java.util.Arrays.fill(head, 1.0);
      }
    }
    return result;
  }
}
