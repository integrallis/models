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
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Applies deterministic retrieval and source-attribution guardrails to generated RAG answers.
 *
 * <p>A weak retrieval abstains. A high-confidence answer is accepted only when every bracketed
 * citation names a retrieved source, at least one trusted citation is present, and every
 * substantive answer token is supported by the question or retrieved evidence. Otherwise the exact
 * retrieved evidence is returned with trusted source IDs.
 */
public final class GroundedAnswerPolicy {
  public static final String ABSTENTION = "INSUFFICIENT_CONTEXT";
  public static final String POLICY_ID =
      "trusted-citation-lexical-entailment-extractive-fallback-v2";
  public static final float DEFAULT_MINIMUM_RETRIEVAL_SCORE = 2.0f;
  private static final Pattern BRACKETED_TEXT = Pattern.compile("\\[([^\\]\\r\\n]+)]");
  private static final Pattern ABSTENTION_PATTERN =
      Pattern.compile("(?i)^INSUFFICIENT_CONTEXT[.!]?$");
  private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{N}]+");
  private static final Set<String> FUNCTION_WORDS =
      Set.of(
          "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "did", "do",
          "does", "for", "from", "in", "is", "it", "its", "of", "on", "or", "that", "the", "these",
          "this", "those", "to", "was", "were", "with");

  private final float minimumRetrievalScore;

  public GroundedAnswerPolicy(float minimumRetrievalScore) {
    if (!Float.isFinite(minimumRetrievalScore) || minimumRetrievalScore < 0) {
      throw new IllegalArgumentException("minimumRetrievalScore must be finite and non-negative");
    }
    this.minimumRetrievalScore = minimumRetrievalScore;
  }

  public static GroundedAnswerPolicy productionDefault() {
    return new GroundedAnswerPolicy(DEFAULT_MINIMUM_RETRIEVAL_SCORE);
  }

  public float minimumRetrievalScore() {
    return minimumRetrievalScore;
  }

  public GroundedAnswer apply(
      String question, List<GroundingDocument> retrieved, String generatedText) {
    Objects.requireNonNull(question, "question");
    Objects.requireNonNull(retrieved, "retrieved");
    Objects.requireNonNull(generatedText, "generatedText");
    if (question.isBlank()) {
      throw new IllegalArgumentException("question must not be blank");
    }

    if (retrieved.isEmpty()
        || retrieved.stream().map(GroundingDocument::score).max(Float::compare).orElse(0.0f)
            < minimumRetrievalScore) {
      return new GroundedAnswer(generatedText, ABSTENTION, GroundingDecision.RETRIEVAL_ABSTENTION);
    }

    String candidate = generatedText.strip();
    if (ABSTENTION_PATTERN.matcher(candidate).matches()
        || !hasOnlyTrustedCitations(candidate, retrieved)
        || !hasOnlySupportedClaims(question, candidate, retrieved)) {
      return new GroundedAnswer(
          generatedText, extractiveAnswer(retrieved), GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    return new GroundedAnswer(generatedText, candidate, GroundingDecision.MODEL_ANSWER);
  }

  private static boolean hasOnlySupportedClaims(
      String question, String candidate, List<GroundingDocument> retrieved) {
    Set<String> supported = words(question);
    retrieved.forEach(hit -> supported.addAll(words(hit.text())));

    String uncited = BRACKETED_TEXT.matcher(candidate).replaceAll(" ");
    return words(uncited).stream().allMatch(supported::contains);
  }

  private static Set<String> words(String text) {
    Set<String> words = new HashSet<>();
    Matcher matcher = WORD.matcher(text.toLowerCase(Locale.ROOT));
    while (matcher.find()) {
      String word = normalizeWord(matcher.group());
      if (!FUNCTION_WORDS.contains(word)) {
        words.add(word);
      }
    }
    return words;
  }

  private static String normalizeWord(String word) {
    if (word.length() > 4 && word.endsWith("ies")) {
      return word.substring(0, word.length() - 3) + "y";
    }
    if (word.length() > 3 && word.endsWith("s") && !word.endsWith("ss")) {
      return word.substring(0, word.length() - 1);
    }
    return word;
  }

  private static boolean hasOnlyTrustedCitations(
      String candidate, List<GroundingDocument> retrieved) {
    Set<String> trustedIds = new HashSet<>();
    retrieved.forEach(hit -> trustedIds.add(hit.id()));

    boolean foundTrusted = false;
    Matcher matcher = BRACKETED_TEXT.matcher(candidate);
    while (matcher.find()) {
      String citation = matcher.group(1).strip();
      if (!trustedIds.contains(citation)) {
        return false;
      }
      foundTrusted = true;
    }
    return foundTrusted;
  }

  private static String extractiveAnswer(List<GroundingDocument> retrieved) {
    StringBuilder answer = new StringBuilder();
    for (GroundingDocument hit : retrieved) {
      if (!answer.isEmpty()) {
        answer.append(' ');
      }
      String text = hit.text().strip();
      answer.append(text);
      if (!text.endsWith(".") && !text.endsWith("!") && !text.endsWith("?")) {
        answer.append('.');
      }
      answer.append(" [").append(hit.id()).append(']');
    }
    return answer.toString();
  }
}
