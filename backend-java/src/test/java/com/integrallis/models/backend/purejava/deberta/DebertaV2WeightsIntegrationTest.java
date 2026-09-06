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
package com.integrallis.models.backend.purejava.deberta;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("model-fixture")
class DebertaV2WeightsIntegrationTest {

  @Test
  void expandsThePinnedFloat16WeightsWithoutChangingTheirValues() throws Exception {
    Path directory = fixtureDirectory();
    DebertaV2Config config = DebertaV2Config.parse(directory.resolve("config.json"));
    try (Arena arena = Arena.ofConfined()) {
      DebertaV2Weights weights =
          DebertaV2Weights.load(
              new SafetensorsTensorSource(SafetensorsBundle.open(directory, arena)), config);

      assertThat(weights.numLayers()).isEqualTo(12);
      assertThat(weights.wordEmbeddings().rows()).isEqualTo(128_100);
      assertThat(weights.wordEmbeddings().columns()).isEqualTo(384);
      assertThat(weights.wordEmbeddings().values())
          .startsWith(
              -0.094482421875f, -0.1954345703125f, -0.0243377685546875f, 0.0040130615234375f);
      assertThat(weights.layer(0).query().values())
          .startsWith(0.10162353515625f, 0.2232666015625f, 0.224853515625f, -0.382568359375f);
      assertThat(weights.classifierBias()).containsExactly(-0.0027370452880859375f);
    }
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.mxbaiRerankerDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.mxbaiRerankerDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isRegularFile(directory.resolve("model.safetensors")));
    return directory;
  }
}
