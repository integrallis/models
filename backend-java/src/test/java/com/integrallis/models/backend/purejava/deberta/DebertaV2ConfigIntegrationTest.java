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

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("model-fixture")
class DebertaV2ConfigIntegrationTest {

  @Test
  void parsesThePinnedMxbaiArchitectureContract() throws Exception {
    Path configPath = fixtureDirectory().resolve("config.json");

    DebertaV2Config config = DebertaV2Config.parse(configPath);

    assertThat(config.hiddenSize()).isEqualTo(384);
    assertThat(config.intermediateSize()).isEqualTo(1536);
    assertThat(config.numLayers()).isEqualTo(12);
    assertThat(config.numHeads()).isEqualTo(6);
    assertThat(config.vocabSize()).isEqualTo(128_100);
    assertThat(config.maxPositions()).isEqualTo(512);
    assertThat(config.positionBuckets()).isEqualTo(256);
    assertThat(config.layerNormEpsilon()).isEqualTo(1.0e-7f);
    assertThat(config.relativeAttention()).isTrue();
    assertThat(config.shareAttentionKey()).isTrue();
    assertThat(config.positionBiasedInput()).isFalse();
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.mxbaiRerankerDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.mxbaiRerankerDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isRegularFile(directory.resolve("config.json")));
    return directory;
  }
}
