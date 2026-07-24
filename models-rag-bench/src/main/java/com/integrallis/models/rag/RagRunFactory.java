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
package com.integrallis.models.rag;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;

final class RagRunFactory {
  private RagRunFactory() {}

  static RagRun create(
      String framework,
      GenerationClient client,
      RagCase testCase,
      List<RetrievedDocument> retrieved,
      String prompt,
      double retrievalMillis,
      double endToEndMillis,
      GenerationResult generation) {
    RagEvaluation rawEvaluation = RagEvaluator.evaluate(testCase, retrieved, generation.text());
    GroundedAnswer grounding =
        GroundedAnswerPolicy.productionDefault()
            .apply(
                testCase.question(),
                retrieved.stream()
                    .map(
                        hit ->
                            new GroundingDocument(
                                hit.document().id(),
                                hit.document().text(),
                                hit.score(),
                                hit.rank()))
                    .toList(),
                generation.text());
    GenerationResult groundedGeneration = generation.withText(grounding.text());
    double frameworkOverhead =
        Math.max(0, endToEndMillis - retrievalMillis - generation.totalMillis());
    return new RagRun(
        framework,
        client.backend(),
        client.model(),
        testCase.id(),
        retrieved,
        sha256(prompt),
        retrievalMillis,
        frameworkOverhead,
        endToEndMillis,
        groundedGeneration,
        grounding,
        rawEvaluation,
        RagEvaluator.evaluate(testCase, retrieved, groundedGeneration.text()));
  }

  private static String sha256(String value) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      return HexFormat.of().formatHex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
  }
}
