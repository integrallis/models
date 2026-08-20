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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TokenConfidenceSignalTest {

  @Test
  void computesChosenProbabilityAndEntropyFromLogits() {
    TokenConfidenceSignal signal = TokenConfidenceSignal.fromLogits(new float[] {2.0f, 1.0f}, 0);

    assertThat(signal.chosenProbability()).isCloseTo(0.7310586f, within(0.000001f));
    assertThat(signal.entropy()).isCloseTo(0.5822031f, within(0.000001f));
  }

  @Test
  void remainsStableForLargeLogits() {
    TokenConfidenceSignal signal =
        TokenConfidenceSignal.fromLogits(new float[] {1000.0f, 999.0f, 998.0f}, 1);

    assertThat(signal.chosenProbability()).isCloseTo(0.24472848f, within(0.000001f));
    assertThat(signal.entropy()).isCloseTo(0.8323956f, within(0.000001f));
  }

  @Test
  void summarizesASequenceOfSignals() {
    GenerationConfidence confidence =
        GenerationConfidence.fromSignals(
            TokenConfidenceSignal.fromLogits(new float[] {2.0f, 1.0f}, 0),
            TokenConfidenceSignal.fromLogits(new float[] {0.0f, 0.0f}, 1));

    assertThat(confidence.tokenCount()).isEqualTo(2);
    assertThat(confidence.minProbability()).isCloseTo(0.5f, within(0.000001f));
    assertThat(confidence.meanLogProbability()).isCloseTo(-0.5032044f, within(0.000001f));
    assertThat(confidence.maxEntropy()).isCloseTo(0.6931472f, within(0.000001f));
  }

  @Test
  void rejectsInvalidInputs() {
    assertThatThrownBy(() -> TokenConfidenceSignal.fromLogits(new float[0], 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("logits");
    assertThatThrownBy(() -> TokenConfidenceSignal.fromLogits(new float[] {1.0f}, 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("chosenToken");
    assertThatThrownBy(() -> TokenConfidenceSignal.fromLogits(new float[] {Float.NaN}, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("finite");
    assertThatThrownBy(GenerationConfidence::empty)
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("signals");
  }

  private static org.assertj.core.data.Offset<Float> within(float value) {
    return org.assertj.core.data.Offset.offset(value);
  }
}
