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
package com.integrallis.models.backend.purejava.soprano;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SopranoVocoderMathTest {

  @Test
  void linearlyUpsamplesFramesWithAlignedCorners() {
    float[] input = {1.0f, 10.0f, 3.0f, 20.0f};

    assertThat(SopranoVocoderMath.interpolateAligned(input, 2, 2, 4))
        .containsExactly(
            new float[] {
              1.0f, 10.0f,
              1.5f, 12.5f,
              2.0f, 15.0f,
              2.5f, 17.5f,
              3.0f, 20.0f
            },
            within(1.0e-6f));
  }

  @Test
  void appliesAZeroPaddedDepthwiseConvolution() {
    float[] input = {1.0f, 2.0f, 3.0f};

    assertThat(
            SopranoVocoderMath.depthwiseConv1d(
                input, 3, 1, new float[] {1.0f, 1.0f, 1.0f}, new float[] {0.0f}, 3))
        .containsExactly(new float[] {3.0f, 6.0f, 5.0f}, within(1.0e-6f));
  }
}
