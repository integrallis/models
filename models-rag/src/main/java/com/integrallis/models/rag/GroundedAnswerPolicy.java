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
 * citation names a retrieved source and every substantive answer token is supported by the question
 * or retrieved evidence. Explicit model abstentions are preserved. When a supported answer omits
 * citations, retrieved provenance is attached deterministically. Otherwise the exact retrieved
 * evidence is returned with trusted source IDs.
 */
public final class GroundedAnswerPolicy {
  public static final String ABSTENTION = "INSUFFICIENT_CONTEXT";
  public static final String POLICY_ID =
      "trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v12";
  public static final float DEFAULT_MINIMUM_RETRIEVAL_SCORE = 2.0f;
  private static final Pattern BRACKETED_TEXT = Pattern.compile("\\[([^\\]\\r\\n]+)]");
  private static final Pattern ABSTENTION_PATTERN =
      Pattern.compile("(?i)^(?:INSUFFICIENT_CONTEXT[.!]?\\s*)+$");
  private static final Pattern LEADING_CONTEXT_ATTRIBUTION =
      Pattern.compile("(?i)^according to the context provided,?\\s*");
  private static final Pattern CLAUSE_BOUNDARY =
      Pattern.compile("(?i)(?:\\s*,\\s+and\\s+|\\s+and\\s+|[;?!]\\s*|\\.\\s+)");
  private static final Pattern STATEMENT_BOUNDARY = Pattern.compile("(?i)(?:[;?!]\\s*|\\.\\s+)");
  private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{N}]+");
  private static final Set<String> FUNCTION_WORDS =
      Set.of(
          "a",
          "an",
          "and",
          "are",
          "as",
          "at",
          "be",
          "been",
          "being",
          "but",
          "by",
          "did",
          "do",
          "context",
          "does",
          "for",
          "from",
          "how",
          "in",
          "information",
          "is",
          "it",
          "its",
          "of",
          "on",
          "or",
          "source",
          "that",
          "the",
          "these",
          "this",
          "those",
          "to",
          "was",
          "were",
          "what",
          "when",
          "where",
          "which",
          "who",
          "why",
          "with");

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
    if (isExplicitAbstention(candidate)) {
      return new GroundedAnswer(generatedText, ABSTENTION, GroundingDecision.MODEL_ABSTENTION);
    }
    String validationCandidate = removeLeadingContextAttribution(candidate);
    if (validationCandidate.isBlank()
        || !hasOnlySupportedClaims(question, validationCandidate, retrieved)
        || !hasEvidenceAnchorsForEveryQuestionClause(question, validationCandidate, retrieved)) {
      return new GroundedAnswer(
          generatedText, extractiveAnswer(retrieved), GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    if (!hasCitations(candidate)) {
      return new GroundedAnswer(
          generatedText,
          attachCitations(candidate, retrieved),
          GroundingDecision.MODEL_ANSWER_WITH_DERIVED_CITATIONS);
    }
    if (!hasOnlyTrustedCitations(candidate, retrieved)) {
      return new GroundedAnswer(
          generatedText, extractiveAnswer(retrieved), GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    return new GroundedAnswer(generatedText, candidate, GroundingDecision.MODEL_ANSWER);
  }

  private static String removeLeadingContextAttribution(String candidate) {
    return LEADING_CONTEXT_ATTRIBUTION.matcher(candidate).replaceFirst("");
  }

  private static boolean isExplicitAbstention(String candidate) {
    String undecorated = BRACKETED_TEXT.matcher(candidate).replaceAll(" ").strip();
    return ABSTENTION_PATTERN.matcher(undecorated).matches();
  }

  private static boolean hasOnlySupportedClaims(
      String question, String candidate, List<GroundingDocument> retrieved) {
    Set<String> supported = words(question);
    retrieved.forEach(
        hit -> {
          supported.addAll(words(hit.title()));
          supported.addAll(words(hit.text()));
        });

    String uncited = BRACKETED_TEXT.matcher(candidate).replaceAll(" ");
    return words(uncited).stream().allMatch(supported::contains);
  }

  private static Set<String> words(String text) {
    Set<String> words = new HashSet<>();
    Matcher matcher = WORD.matcher(text.toLowerCase(Locale.ROOT));
    while (matcher.find()) {
      String rawWord = matcher.group();
      if (FUNCTION_WORDS.contains(rawWord)) {
        continue;
      }
      String word = normalizeWord(rawWord);
      if (!FUNCTION_WORDS.contains(word)) {
        words.add(word);
      }
    }
    return words;
  }

  private static boolean hasEvidenceAnchorsForEveryQuestionClause(
      String question, String candidate, List<GroundingDocument> retrieved) {
    Set<String> questionWords = words(question);
    Set<String> evidenceWords = new HashSet<>();
    retrieved.forEach(
        hit -> {
          evidenceWords.addAll(words(hit.title()));
          evidenceWords.addAll(words(hit.text()));
        });
    evidenceWords.removeAll(questionWords);

    List<Set<String>> questionClauses = clauses(question);
    List<Set<String>> candidateClauses =
        statementClauses(BRACKETED_TEXT.matcher(candidate).replaceAll(" "));
    if (questionClauses.size() == 1) {
      Set<String> anchors = new HashSet<>(words(candidate));
      anchors.retainAll(evidenceWords);
      return hasConcreteEvidenceAnchor(anchors);
    }

    List<Set<String>> evidenceClauses =
        retrieved.stream()
            .flatMap(hit -> statementClauses(hit.title() + "\n" + hit.text()).stream())
            .toList();
    List<Set<String>> matchingEvidenceClauses =
        questionClauses.stream()
            .map(questionClause -> bestMatchingClause(questionClause, evidenceClauses))
            .toList();
    for (int clauseIndex = 0; clauseIndex < questionClauses.size(); clauseIndex++) {
      Set<String> questionClause = questionClauses.get(clauseIndex);
      Set<String> clauseSpecificWords = new HashSet<>(questionClause);
      Set<String> clauseEvidenceWords = new HashSet<>(matchingEvidenceClauses.get(clauseIndex));
      for (int otherIndex = 0; otherIndex < questionClauses.size(); otherIndex++) {
        if (otherIndex != clauseIndex) {
          clauseSpecificWords.removeAll(questionClauses.get(otherIndex));
          clauseEvidenceWords.removeAll(matchingEvidenceClauses.get(otherIndex));
        }
      }
      clauseEvidenceWords.removeAll(questionWords);
      if (clauseEvidenceWords.isEmpty()) {
        clauseEvidenceWords.addAll(matchingEvidenceClauses.get(clauseIndex));
        clauseEvidenceWords.removeAll(questionWords);
      }
      Set<String> anchors = new HashSet<>();
      for (Set<String> candidateClause : candidateClauses) {
        long overlap = candidateClause.stream().filter(questionClause::contains).count();
        long clauseSpecificOverlap =
            candidateClause.stream().filter(clauseSpecificWords::contains).count();
        int requiredClauseSpecificOverlap = Math.min(1, clauseSpecificWords.size());
        if (overlap >= 1 && clauseSpecificOverlap >= requiredClauseSpecificOverlap) {
          candidateClause.stream().filter(clauseEvidenceWords::contains).forEach(anchors::add);
        }
      }
      if (!hasConcreteEvidenceAnchor(anchors)) {
        return false;
      }
    }
    return true;
  }

  private static List<Set<String>> clauses(String text) {
    return CLAUSE_BOUNDARY
        .splitAsStream(text)
        .map(GroundedAnswerPolicy::words)
        .filter(clause -> !clause.isEmpty())
        .toList();
  }

  private static List<Set<String>> statementClauses(String text) {
    return STATEMENT_BOUNDARY
        .splitAsStream(text)
        .map(GroundedAnswerPolicy::words)
        .filter(clause -> !clause.isEmpty())
        .toList();
  }

  private static Set<String> bestMatchingClause(
      Set<String> questionClause, List<Set<String>> evidenceClauses) {
    Set<String> best = Set.of();
    long bestOverlap = -1;
    for (Set<String> evidenceClause : evidenceClauses) {
      long overlap = evidenceClause.stream().filter(questionClause::contains).count();
      if (overlap > bestOverlap) {
        best = evidenceClause;
        bestOverlap = overlap;
      }
    }
    return best;
  }

  private static boolean hasConcreteEvidenceAnchor(Set<String> anchors) {
    return anchors.size() >= 2
        || anchors.stream()
            .anyMatch(
                anchor -> anchor.length() >= 8 || anchor.codePoints().allMatch(Character::isDigit));
  }

  private static String normalizeWord(String word) {
    if (word.equals("thrown")) {
      return "throw";
    }
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

  private static boolean hasCitations(String candidate) {
    return BRACKETED_TEXT.matcher(candidate).find();
  }

  private static String attachCitations(String candidate, List<GroundingDocument> retrieved) {
    StringBuilder answer = new StringBuilder(candidate);
    for (GroundingDocument hit : retrieved) {
      if (!answer.isEmpty() && !Character.isWhitespace(answer.charAt(answer.length() - 1))) {
        answer.append(' ');
      }
      answer.append('[').append(hit.id()).append(']');
    }
    return answer.toString();
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
