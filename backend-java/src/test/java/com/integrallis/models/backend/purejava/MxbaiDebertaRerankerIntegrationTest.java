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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("model-fixture")
class MxbaiDebertaRerankerIntegrationTest {

  @Test
  void matchesPinnedTransformersLogitsAndRanking() {
    try (SafetensorsRerankingModel model = SafetensorsRerankingModel.load(fixtureDirectory())) {
      double relevant =
          model.score(
              "What is the population of Berlin?",
              "Berlin has a population of about 3.7 million people.");
      double irrelevant =
          model.score("What is the population of Berlin?", "The Eiffel Tower is located in Paris.");

      assertThat(relevant).isCloseTo(2.6104862689971924, within(0.001));
      assertThat(irrelevant).isCloseTo(-3.1909432411193848, within(0.001));
      assertThat(relevant).isGreaterThan(irrelevant);
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
