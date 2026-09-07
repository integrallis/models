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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.ops.TensorOps;
import org.junit.jupiter.api.Test;

class QwenKvRidgeExperimentTest {

  @Test
  void removesQwenNeoxRope() {
    float[] original = {0.25f, -0.5f, 0.75f, -1.0f, 1.25f, -1.5f, 1.75f, -2.0f};
    float[] rotated = original.clone();

    TensorOps.ropeNeox(rotated, 0, 37, rotated.length, 1_000_000.0f, 1.0f);
    QwenKvRidgeExperiment.removeRopeNeox(rotated, 37, rotated.length, 1_000_000.0f, 1.0f);

    assertThat(rotated).usingComparatorWithPrecision(1.0e-6f).containsExactly(original);
  }

  @Test
  void cosineDistinguishesAlignedAndOrthogonalSamples() {
    float[][] first = {{1.0f, 0.0f}, {0.0f, 2.0f}};
    float[][] aligned = {{2.0f, 0.0f}, {0.0f, 4.0f}};
    float[][] orthogonal = {{0.0f, 1.0f}, {-2.0f, 0.0f}};

    assertThat(QwenKvRidgeExperiment.cosine(first, aligned)).isCloseTo(1.0, within(1.0e-12));
    assertThat(QwenKvRidgeExperiment.cosine(first, orthogonal)).isCloseTo(0.0, within(1.0e-12));
  }

  @Test
  void concatenatesSelectedSourceLayersInDeclaredOrder() {
    float[][][] layers = {
      {{1.0f, 2.0f}, {3.0f, 4.0f}},
      {{5.0f, 6.0f}, {7.0f, 8.0f}},
      {{9.0f, 10.0f}, {11.0f, 12.0f}}
    };

    float[][] concatenated = QwenKvRidgeExperiment.concatenateLayers(layers, new int[] {2, 0});

    assertThat(concatenated).hasDimensions(2, 4);
    assertThat(concatenated[0]).containsExactly(9.0f, 10.0f, 1.0f, 2.0f);
    assertThat(concatenated[1]).containsExactly(11.0f, 12.0f, 3.0f, 4.0f);
  }
}
