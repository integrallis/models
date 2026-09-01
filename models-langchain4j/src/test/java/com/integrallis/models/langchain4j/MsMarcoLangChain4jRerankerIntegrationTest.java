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
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.GgufRerankingModel;
import dev.langchain4j.data.segment.TextSegment;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class MsMarcoLangChain4jRerankerIntegrationTest {

  @Test
  void scoringModelUsesTheRealCrossEncoderAndPreservesDocumentOrder() {
    try (var model =
        new ModelsScoringModel(
            GgufRerankingModel.load(
                Path.of(System.getProperty("models.fixtures.msMarcoReranker"))))) {
      List<Double> scores =
          model
              .scoreAll(
                  List.of(
                      TextSegment.from(
                          "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
                      TextSegment.from("Domestic cats sleep for a large part of the day.")),
                  "How many people live in Berlin?")
              .content();

      assertThat(scores.get(0)).isCloseTo(8.846, within(0.15));
      assertThat(scores.get(1)).isCloseTo(-11.225, within(0.15));
      assertThat(scores.get(0)).isGreaterThan(scores.get(1));
    }
  }
}
