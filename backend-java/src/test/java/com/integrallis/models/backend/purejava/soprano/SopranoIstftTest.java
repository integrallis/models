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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SopranoIstftTest {

  @Test
  void reconstructsAndCenterTrimsOverlappingFrames() {
    int fftSize = 8;
    int bins = fftSize / 2 + 1;
    int frames = 3;
    float[] head = new float[frames * (fftSize + 2)];
    Arrays.fill(head, -100.0f);
    for (int frame = 0; frame < frames; frame++) {
      int row = frame * (fftSize + 2);
      head[row + 1] = 0.0f;
      Arrays.fill(head, row + bins, row + fftSize + 2, 0.0f);
    }

    assertThat(SopranoIstft.decode(head, frames, fftSize, 2, ones(fftSize)))
        .containsExactly(
            new float[] {0.0f, -0.058925565f, -0.083333336f, -0.058925565f}, within(1.0e-6f));
  }

  @Test
  void validatesHeadAndWindowShapes() {
    assertThatThrownBy(() -> SopranoIstft.decode(new float[9], 1, 8, 2, ones(8)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("head");
    assertThatThrownBy(() -> SopranoIstft.decode(new float[10], 1, 8, 2, ones(7)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("window");
  }

  private static float[] ones(int size) {
    float[] values = new float[size];
    Arrays.fill(values, 1.0f);
    return values;
  }
}
