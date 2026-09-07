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

class RidgeMapperTest {

  @Test
  void recoversHeldOutLinearTransformation() {
    double[][] transform = {
      {1.5, -0.25, 0.75},
      {-0.5, 2.0, 0.125},
      {0.25, 0.5, -1.25}
    };
    float[][] training = randomSamples(128, 3, 41L);
    float[][] targets = transform(training, transform);
    RidgeMapper mapper = RidgeMapper.fit(training, targets, 1.0e-8);

    float[][] heldOut = randomSamples(32, 3, 97L);
    float[][] expected = transform(heldOut, transform);

    assertThat(RidgeMapper.relativeL2(expected, mapper.predict(heldOut))).isLessThan(1.0e-6);
  }

  @Test
  void rejectsShapeAndRegularizationErrors() {
    assertThatThrownBy(() -> RidgeMapper.fit(new float[0][], new float[0][], 0.1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("training samples");
    assertThatThrownBy(
            () -> RidgeMapper.fit(new float[][] {{1.0f, 2.0f}}, new float[][] {{1.0f}}, -0.1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("regularization");
  }

  private static float[][] randomSamples(int samples, int dimensions, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[][] values = new float[samples][dimensions];
    for (int sample = 0; sample < samples; sample++) {
      for (int dimension = 0; dimension < dimensions; dimension++) {
        values[sample][dimension] = (float) random.nextDouble(-2.0, 2.0);
      }
    }
    return values;
  }

  private static float[][] transform(float[][] inputs, double[][] transform) {
    float[][] outputs = new float[inputs.length][transform[0].length];
    for (int sample = 0; sample < inputs.length; sample++) {
      for (int output = 0; output < outputs[sample].length; output++) {
        double value = 0.0;
        for (int input = 0; input < inputs[sample].length; input++) {
          value += inputs[sample][input] * transform[input][output];
        }
        outputs[sample][output] = (float) value;
      }
    }
    return outputs;
  }
}
