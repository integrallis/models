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

class PerHeadRidgeMapperTest {

  @Test
  void learnsDifferentTransformationForEachInterleavedHead() {
    int positions = 64;
    int heads = 2;
    float[][] inputs = samples(positions, heads, 71L);
    float[][] targets = transformByHead(inputs, heads);
    PerHeadRidgeMapper mapper = PerHeadRidgeMapper.fit(inputs, targets, heads, 1.0e-8);

    float[][] heldOut = samples(16, heads, 83L);
    float[][] expected = transformByHead(heldOut, heads);

    assertThat(RidgeMapper.relativeL2(expected, mapper.predict(heldOut))).isLessThan(1.0e-6);
  }

  private static float[][] samples(int positions, int heads, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[][] values = new float[positions * heads][2];
    for (float[] value : values) {
      value[0] = (float) random.nextDouble(-2.0, 2.0);
      value[1] = (float) random.nextDouble(-2.0, 2.0);
    }
    return values;
  }

  private static float[][] transformByHead(float[][] inputs, int heads) {
    float[][] outputs = new float[inputs.length][2];
    for (int sample = 0; sample < inputs.length; sample++) {
      float first = inputs[sample][0];
      float second = inputs[sample][1];
      if (sample % heads == 0) {
        outputs[sample][0] = 2.0f * first + second;
        outputs[sample][1] = first - 0.5f * second;
      } else {
        outputs[sample][0] = -first + 0.25f * second;
        outputs[sample][1] = 0.75f * first + 3.0f * second;
      }
    }
    return outputs;
  }
}
