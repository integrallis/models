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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The LFM2 hybrid decoder, end to end on a toy model.
 *
 * <p>Three layers in the order convolution, attention, convolution, so both mixers and the boundary
 * between them are exercised. The discriminating tests vary ONE tensor at a time: a convolution
 * kernel and an attention key projection. A hybrid that quietly ran the wrong mixer, or ignored the
 * recurrent state, would still produce finite plausible logits -- so "it runs" is asserted, and
 * then separately that each half actually contributes.
 */
@Tag("unit")
class Lfm2ForwardPassTest {

  @Test
  void runsBothMixersAndProducesFiniteLogits() {
    Lfm2Config config = config();
    assertThat(config.usesAttention(0)).isFalse();
    assertThat(config.usesAttention(1)).isTrue();
    assertThat(config.usesAttention(2)).isFalse();
    assertThat(config.attentionLayers()).isEqualTo(1);

    Lfm2ForwardPass pass = pass(30, 40);
    float[] first = pass.forward(0, 0).clone();
    float[] second = pass.forward(1, 1).clone();

    assertThat(first).hasSize(Lfm2ToyModel.VOCAB);
    for (float value : first) {
      assertThat(Float.isFinite(value)).describedAs("logits must be finite").isTrue();
    }
    for (float value : second) {
      assertThat(Float.isFinite(value)).isTrue();
    }
    assertThat(second).isNotEqualTo(first);
    assertThat(pass.nextPosition()).isEqualTo(2);
  }

  @Test
  void theConvolutionKernelReachesTheOutput() {
    // If the convolutional layers were skipped, or their kernel ignored, these would agree.
    float[] withOneKernel = logits(pass(30, 40));
    float[] withAnother = logits(pass(77, 40));

    assertThat(withAnother).isNotEqualTo(withOneKernel);
  }

  @Test
  void theAttentionKeyProjectionReachesTheOutput() {
    // The complement: the single attention layer must matter too, so the hybrid is not silently
    // running convolutions everywhere.
    float[] withOneKey = logits(pass(30, 40));
    float[] withAnother = logits(pass(30, 91));

    assertThat(withAnother).isNotEqualTo(withOneKey);
  }

  @Test
  void theRecurrentStateCarriesBetweenTokensAndResetClearsIt() {
    // The convolution's whole difference from attention is that it carries a shift register.
    // Feeding
    // the same token twice must not give the same logits, and a reset must return to the first.
    Lfm2ForwardPass pass = pass(30, 40);
    float[] firstOfRun = pass.forward(0, 0).clone();
    float[] secondSameToken = pass.forward(0, 1).clone();

    assertThat(secondSameToken)
        .describedAs("the same token at a later position sees a different state")
        .isNotEqualTo(firstOfRun);

    pass.reset();
    assertThat(pass.nextPosition()).isZero();
    assertThat(pass.forward(0, 0).clone())
        .describedAs("reset must clear the convolution state as well as the position")
        .containsExactly(firstOfRun);
  }

  private static float[] logits(Lfm2ForwardPass pass) {
    pass.forward(0, 0);
    return pass.forward(1, 1).clone();
  }

  private static Lfm2ForwardPass pass(int convKernelSeed, int attentionKeySeed) {
    GgufFile file = Lfm2ToyModel.file(convKernelSeed, attentionKeySeed);
    Lfm2Config config = Lfm2Config.fromMetadata(file.metadata());
    return new Lfm2ForwardPass(config, Lfm2Weights.fromGgufFile(file, config), 8);
  }

  private static Lfm2Config config() {
    GgufFile file = Lfm2ToyModel.file(30, 40);
    return Lfm2Config.fromMetadata(file.metadata());
  }

  private static Map<String, float[]> unused() {
    return new LinkedHashMap<>();
  }
}
