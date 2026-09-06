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
package com.integrallis.models.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.SafetensorsRerankingModel;
import dev.langchain4j.data.segment.TextSegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("model-fixture")
class MxbaiDebertaLangChain4jRerankerIntegrationTest {

  @Test
  void scoringModelUsesTheRealCrossEncoderAndPreservesDocumentOrder() {
    String configured = System.getProperty("models.fixtures.mxbaiRerankerDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.mxbaiRerankerDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isRegularFile(directory.resolve("model.safetensors")));

    try (var model = new ModelsScoringModel(SafetensorsRerankingModel.load(directory))) {
      List<Double> scores =
          model
              .scoreAll(
                  List.of(
                      TextSegment.from("Berlin has a population of about 3.7 million people."),
                      TextSegment.from("The Eiffel Tower is located in Paris.")),
                  "What is the population of Berlin?")
              .content();

      assertThat(scores).hasSize(2);
      assertThat(scores.get(0)).isGreaterThan(scores.get(1));
    }
  }
}
