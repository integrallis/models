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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.gguf.GgufEmbeddedFiles;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class SopranoVocoderWeightsIntegrationTest {

  @Test
  void loadsThePublishedStandaloneSopranoDecoder() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    assumeTrue(configured != null && !configured.isBlank(), "MODELS_SOPRANO_GGUF is not set");
    Path model = Path.of(configured);
    assertThat(Files.isRegularFile(model)).isTrue();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(model, arena);
      var embedded = GgufEmbeddedFiles.from(file.metadata());
      var config = SopranoConfig.fromJson(embedded.readUtf8("config.json"));
      var weights = SopranoVocoderWeights.load(file, config);

      assertThat(weights.layerCount()).isEqualTo(8);
      assertThat(weights.head().type()).isIn(GgufTensorType.Q8_0, GgufTensorType.BF16);
      assertThat(weights.head().preparedF32()).isNotNull();
      for (int layer = 0; layer < weights.layerCount(); layer++) {
        assertThat(weights.layer(layer).pointwiseUp().preparedF32()).isNotNull();
        assertThat(weights.layer(layer).pointwiseDown().preparedF32()).isNotNull();
      }
      assertThat(weights.preparedExpandedBytes()).isEqualTo(119_543_808L);
      long expectedSerializedBytes =
          weights.head().type() == GgufTensorType.Q8_0 ? 31_753_824L : 59_771_904L;
      assertThat(weights.preparedSerializedBytes()).isEqualTo(expectedSerializedBytes);
      assertThat(weights.window()).hasSize(2048);

      float[] audio =
          new SopranoVocoder(config, weights).decode(new float[2 * config.hiddenSize()], 2);
      assertThat(audio).hasSize(config.upscale() * config.hopLength());
      for (float sample : audio) {
        assertThat(Float.isFinite(sample)).isTrue();
      }
    }
  }
}
