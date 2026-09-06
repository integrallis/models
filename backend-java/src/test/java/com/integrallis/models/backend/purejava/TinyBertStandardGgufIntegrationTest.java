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
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Standard rank-pooling and cls.* tensor compatibility against the corrected TinyBERT artifact. */
@Tag("integration")
class TinyBertStandardGgufIntegrationTest {
  private static final String ARTIFACT_PROPERTY = "models.fixtures.tinyBertReranker";
  private static final String ARTIFACT_SHA256 =
      "3c6701e5cd30548a68f5ce0cbbc1ad0833f414021f3a8d5e1436f80738f71ef4";
  private static final String QUERY = "How many people live in Berlin?";
  private static final List<String> DOCUMENTS =
      List.of(
          "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
          "Paris is the capital and most populous city of France.",
          "Berlin is well known for its museums and its metropolitan area of about six million people.",
          "Domestic cats sleep for a large part of the day.",
          "New York City had an estimated population of 8,804,190 in 2020.",
          "The Berlin Wall divided the city from 1961 until 1989.");
  private static final double[] TRANSFORMERS = {
    7.235748291, -11.296918869, 7.422357559, -11.582018852, -10.010071754, -5.596106052
  };

  @Test
  void standardGgufHeadMatchesTransformersScaleAndRanking() throws Exception {
    String configured = System.getProperty(ARTIFACT_PROPERTY);
    assumeTrue(configured != null && !configured.isBlank(), () -> "set -D" + ARTIFACT_PROPERTY);
    Path artifact = Path.of(configured);
    assumeTrue(Files.isRegularFile(artifact), () -> "missing TinyBERT artifact: " + artifact);
    assertThat(
            HexFormat.of()
                .formatHex(
                    MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(artifact))))
        .isEqualTo(ARTIFACT_SHA256);

    try (GgufRerankingModel model = GgufRerankingModel.load(artifact)) {
      List<Double> scores = model.scoreAll(QUERY, DOCUMENTS);

      for (int index = 0; index < scores.size(); index++) {
        assertThat(scores.get(index)).isCloseTo(TRANSFORMERS[index], within(0.08));
      }
      assertThat(model.rerank(QUERY, DOCUMENTS, DOCUMENTS.size()))
          .extracting(result -> result.originalIndex())
          .containsExactly(2, 0, 5, 4, 1, 3);
    }
  }
}
