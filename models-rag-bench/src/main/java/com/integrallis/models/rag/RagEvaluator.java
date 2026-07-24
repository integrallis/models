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

import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Deterministic retrieval, fact, citation, and abstention evaluator. */
public final class RagEvaluator {
  private static final String ABSTENTION = "INSUFFICIENT_CONTEXT";
  private static final Pattern CITATION =
      Pattern.compile("\\[([a-z0-9][a-z0-9-]*)]", Pattern.CASE_INSENSITIVE);
  private static final Pattern CURRENCY_SYMBOL = Pattern.compile("\\$(\\d+(?:\\.\\d+)?)");
  private static final Pattern PLURAL_DOLLARS =
      Pattern.compile("\\b(\\d+(?:\\.\\d+)?)\\s+dollars\\b");

  private RagEvaluator() {}

  public static RagEvaluation evaluate(
      RagCase testCase, List<RetrievedDocument> retrieved, String answer) {
    Set<String> expected = normalizedSet(testCase.relevantDocumentIds());
    Set<String> retrievedIds = new HashSet<>();
    retrieved.forEach(hit -> retrievedIds.add(normalizeId(hit.document().id())));
    Set<String> citations = citations(answer);

    double retrievalRecall = expected.isEmpty() ? 1.0 : fractionPresent(expected, retrievedIds);
    double reciprocalRank = reciprocalRank(expected, retrieved);
    double factCoverage =
        factCoverage(testCase.requiredFacts(), testCase.factAlternatives(), answer);
    double citationRecall =
        expected.isEmpty()
            ? (citations.isEmpty() ? 1.0 : 0.0)
            : fractionPresent(expected, citations);
    double citationPrecision =
        citations.isEmpty()
            ? (expected.isEmpty() ? 1.0 : 0.0)
            : fractionPresent(citations, expected);
    boolean abstained = ABSTENTION.equals(answer.trim());
    boolean correct =
        testCase.answerable()
            ? !abstained && factCoverage == 1.0 && citationRecall == 1.0 && citationPrecision == 1.0
            : abstained && citations.isEmpty();
    return new RagEvaluation(
        retrievalRecall,
        reciprocalRank,
        factCoverage,
        citationRecall,
        citationPrecision,
        abstained,
        correct);
  }

  private static double reciprocalRank(Set<String> expected, List<RetrievedDocument> retrieved) {
    if (expected.isEmpty()) {
      return 1.0;
    }
    for (RetrievedDocument hit : retrieved) {
      if (expected.contains(normalizeId(hit.document().id()))) {
        return 1.0 / hit.rank();
      }
    }
    return 0.0;
  }

  private static double factCoverage(
      List<String> facts, List<RagFactAlternative> alternatives, String answer) {
    if (facts.isEmpty()) {
      return 1.0;
    }
    String normalizedAnswer = normalizeText(answer);
    long present =
        facts.stream()
            .filter(
                fact ->
                    matches(normalizedAnswer, fact)
                        || alternatives.stream()
                            .filter(alternative -> alternative.fact().equals(fact))
                            .anyMatch(
                                alternative ->
                                    matches(normalizedAnswer, alternative.alternative())))
            .count();
    return (double) present / facts.size();
  }

  private static boolean matches(String normalizedAnswer, String fact) {
    return normalizedAnswer.contains(normalizeText(fact));
  }

  private static Set<String> citations(String answer) {
    Set<String> citations = new HashSet<>();
    Matcher matcher = CITATION.matcher(answer);
    while (matcher.find()) {
      citations.add(normalizeId(matcher.group(1)));
    }
    return citations;
  }

  private static Set<String> normalizedSet(List<String> values) {
    Set<String> normalized = new HashSet<>();
    values.forEach(value -> normalized.add(normalizeId(value)));
    return normalized;
  }

  private static double fractionPresent(Set<String> required, Set<String> actual) {
    long present = required.stream().filter(actual::contains).count();
    return (double) present / required.size();
  }

  private static String normalizeId(String value) {
    return value.toLowerCase(Locale.ROOT);
  }

  private static String normalizeText(String value) {
    String normalized = value.toLowerCase(Locale.ROOT).replace(",", "");
    normalized = CURRENCY_SYMBOL.matcher(normalized).replaceAll("$1 dollar");
    normalized = PLURAL_DOLLARS.matcher(normalized).replaceAll("$1 dollar");
    return normalized.replaceAll("[^a-z0-9]+", " ").trim().replaceAll("\\s+", " ");
  }
}
