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
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("model-fixture")
class MxbaiDebertaRerankerIntegrationTest {

  @Test
  void matchesPinnedTransformersLogitsAndRanking() {
    try (SafetensorsRerankingModel model = SafetensorsRerankingModel.load(fixtureDirectory())) {
      String query = "What is the population of Berlin?";
      List<String> documents =
          List.of(
              "Berlin has a population of about 3.7 million people.",
              "The Eiffel Tower is located in Paris.",
              "Berlin is well known for its museums and its metropolitan area of about six million people.",
              "Domestic cats sleep for a large part of the day.",
              "New York City had an estimated population of 8,804,190 in 2020.",
              "The Berlin Wall divided the city from 1961 until 1989.");
      double[] expected = {
        2.610485553741455,
        -3.1909432411193848,
        1.1550936698913574,
        -3.9371538162231445,
        -3.6461875438690186,
        -1.0840436220169067
      };

      List<Double> scores = model.scoreAll(query, documents);

      assertThat(scores).hasSize(expected.length);
      for (int index = 0; index < expected.length; index++) {
        assertThat(scores.get(index)).isCloseTo(expected[index], within(0.001));
      }
      assertThat(model.rerank(query, documents))
          .extracting(result -> result.originalIndex())
          .containsExactly(0, 2, 5, 1, 4, 3);
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
