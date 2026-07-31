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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Applies deterministic retrieval and source-attribution guardrails to generated RAG answers.
 *
 * <p>{@link #assess(String, List)} rejects irrelevant, oversized, or suspicious retrieved context
 * before generation. A high-confidence answer is accepted only when every citation names a
 * retrieved source and every substantive answer token is supported by the question or retrieved
 * evidence. Explicit model abstentions are preserved. When a supported answer omits citations,
 * retrieved provenance is attached deterministically. Otherwise the exact retrieved evidence is
 * returned with trusted source IDs.
 */
public final class GroundedAnswerPolicy {
  public static final String ABSTENTION = "INSUFFICIENT_CONTEXT";
  public static final String POLICY_ID =
      "bounded-context-injection-screened-citation-safe-statement-grounding-v20";
  public static final float DEFAULT_MINIMUM_RETRIEVAL_SCORE = 2.0f;
  private static final Pattern CITATION =
      Pattern.compile("\\[([A-Za-z0-9][A-Za-z0-9._:-]{0,127})]");
  private static final Pattern ABSTENTION_PATTERN =
      Pattern.compile("(?i)^(?:INSUFFICIENT_CONTEXT[.!]?\\s*)+$");
  private static final Pattern TERMINAL_ABSTENTION_PATTERN =
      Pattern.compile(
          "(?i)(?:therefore,?\\s+)?(?:the\\s+)?answer\\s+is\\s+"
              + "INSUFFICIENT_CONTEXT[.!]?\\s*$");
  private static final Pattern LEADING_CONTEXT_ATTRIBUTION =
      Pattern.compile("(?i)^according to the context provided,?\\s*");
  private static final Pattern LATEX_MATRIX_DELIMITER =
      Pattern.compile("\\\\(?:begin|end)\\{(?:bmatrix|matrix|pmatrix|vmatrix|Vmatrix)}");
  private static final Pattern OPTIONAL_PLURAL_MARKER = Pattern.compile("(?i)\\(s\\)");
  private static final Pattern NUMBERED_LIST_MARKER = Pattern.compile("(?m)(^|\\s)\\d+[.)]\\s+");
  private static final Pattern CLAUSE_BOUNDARY =
      Pattern.compile("(?i)(?:\\s*,\\s+and\\s+|\\s+and\\s+|[;?!]\\s*|\\.\\s+)");
  private static final Pattern STATEMENT_BOUNDARY = Pattern.compile("(?i)(?:[;?!]\\s*|\\.\\s+)");
  private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{N}]+");
  private static final Pattern NAMED_ENTITY_TOKEN =
      Pattern.compile("\\b[\\p{Lu}][\\p{L}\\p{N}_-]{2,}\\b");
  private static final Pattern ATOMIC_IDENTIFIER =
      Pattern.compile(
          "(?i)\\b(?=[\\p{L}\\p{N}-]*\\p{L})(?=[\\p{L}\\p{N}-]*\\p{N})"
              + "[\\p{L}\\p{N}]+(?:-[\\p{L}\\p{N}]+)+\\b");
  private static final List<Pattern> PROMPT_INJECTION_PATTERNS =
      List.of(
          Pattern.compile(
              "(?i)\\b(?:ignore|disregard|forget)\\s+(?:all\\s+)?"
                  + "(?:previous|prior|above|earlier)\\s+"
                  + "(?:instructions?|rules?|prompts?|directions?)\\b"),
          Pattern.compile(
              "(?i)\\b(?:do not|don't)\\s+(?:follow|obey)\\s+(?:the\\s+)?"
                  + "(?:previous|prior|system|developer)\\s+"
                  + "(?:instructions?|rules?|prompt)\\b"),
          Pattern.compile(
              "(?i)\\b(?:reveal|print|show|disclose|repeat)\\s+(?:the\\s+)?"
                  + "(?:system|developer)\\s+(?:message|prompt|instructions?)\\b"),
          Pattern.compile(
              "(?i)(?:<\\|(?:im_start|im_end|system|assistant|endoftext)\\|>"
                  + "|\\[/?INST]|<<\\s*/?SYS\\s*>>)"));
  private static final Set<String> FUNCTION_WORDS =
      Set.of(
          "a",
          "an",
          "and",
          "answer",
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
          "frame",
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
          "period",
          "provided",
          "question",
          "source",
          "that",
          "the",
          "these",
          "this",
          "those",
          "time",
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
  private final GroundingLimits limits;

  public GroundedAnswerPolicy(float minimumRetrievalScore) {
    this(minimumRetrievalScore, GroundingLimits.productionDefault());
  }

  public GroundedAnswerPolicy(float minimumRetrievalScore, GroundingLimits limits) {
    if (!Float.isFinite(minimumRetrievalScore) || minimumRetrievalScore < 0) {
      throw new IllegalArgumentException("minimumRetrievalScore must be finite and non-negative");
    }
    this.minimumRetrievalScore = minimumRetrievalScore;
    this.limits = Objects.requireNonNull(limits, "limits");
  }

  public static GroundedAnswerPolicy productionDefault() {
    return new GroundedAnswerPolicy(DEFAULT_MINIMUM_RETRIEVAL_SCORE);
  }

  public float minimumRetrievalScore() {
    return minimumRetrievalScore;
  }

  public GroundingLimits limits() {
    return limits;
  }

  /**
   * Validates retrieval relevance, size, and common prompt-injection payloads before generation.
   *
   * <p>This deterministic screen is intentionally conservative. Applications should call it before
   * placing retrieved text in a model prompt; {@link #apply(String, List, String)} repeats the same
   * check so an unsafe context cannot produce an accepted final answer.
   */
  public GroundingContextDecision assess(String question, List<GroundingDocument> retrieved) {
    requireInputs(question, retrieved);
    if (retrieved.isEmpty()) {
      return GroundingContextDecision.EMPTY;
    }
    if (retrieved.size() > limits.maximumRetrievedDocuments()) {
      return GroundingContextDecision.TOO_MANY_DOCUMENTS;
    }
    long retrievedCharacters = 0;
    for (GroundingDocument document : retrieved) {
      retrievedCharacters += (long) document.title().length() + document.text().length();
      if (retrievedCharacters > limits.maximumRetrievedCharacters()) {
        return GroundingContextDecision.TOO_LARGE;
      }
      if (containsPromptInjection(document.title()) || containsPromptInjection(document.text())) {
        return GroundingContextDecision.PROMPT_INJECTION;
      }
    }
    if (retrieved.stream().map(GroundingDocument::score).max(Float::compare).orElse(0.0f)
        < minimumRetrievalScore) {
      return GroundingContextDecision.LOW_CONFIDENCE;
    }
    if (lacksNamedEntityOverlap(question, retrieved)) {
      return GroundingContextDecision.QUESTION_MISMATCH;
    }
    return GroundingContextDecision.ACCEPTED;
  }

  public GroundedAnswer apply(
      String question, List<GroundingDocument> retrieved, String generatedText) {
    Objects.requireNonNull(generatedText, "generatedText");

    if (!assess(question, retrieved).generationAllowed()) {
      return new GroundedAnswer(generatedText, ABSTENTION, GroundingDecision.RETRIEVAL_ABSTENTION);
    }

    String candidate = generatedText.strip();
    if (isExplicitAbstention(candidate)) {
      return new GroundedAnswer(generatedText, ABSTENTION, GroundingDecision.MODEL_ABSTENTION);
    }
    String validationCandidate =
        removeValidationOnlyMarkup(removeLeadingContextAttribution(candidate));
    if (validationCandidate.isBlank()
        || !preservesAtomicIdentifiers(validationCandidate, retrieved)
        || !hasOnlySupportedClaims(question, validationCandidate, retrieved)
        || !hasEvidenceAnchorsForEveryQuestionClause(question, validationCandidate, retrieved)) {
      return new GroundedAnswer(
          generatedText, extractiveAnswer(retrieved), GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    List<GroundingDocument> supportingDocuments =
        supportingDocuments(question, validationCandidate, retrieved, retrieved);
    if (supportingDocuments.isEmpty()) {
      return new GroundedAnswer(
          generatedText, extractiveAnswer(retrieved), GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    if (!hasCitations(candidate)) {
      return new GroundedAnswer(
          generatedText,
          attachCitations(candidate, supportingDocuments),
          GroundingDecision.MODEL_ANSWER_WITH_DERIVED_CITATIONS);
    }
    if (!hasOnlyTrustedCitations(candidate, retrieved)) {
      return new GroundedAnswer(
          generatedText, extractiveAnswer(retrieved), GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    List<GroundingDocument> citedDocuments = citedDocuments(candidate, retrieved);
    if (supportingDocuments(question, validationCandidate, retrieved, citedDocuments).isEmpty()) {
      return new GroundedAnswer(
          generatedText, extractiveAnswer(retrieved), GroundingDecision.EXTRACTIVE_FALLBACK);
    }
    return new GroundedAnswer(generatedText, candidate, GroundingDecision.MODEL_ANSWER);
  }

  private static String removeLeadingContextAttribution(String candidate) {
    return LEADING_CONTEXT_ATTRIBUTION.matcher(candidate).replaceFirst("");
  }

  private static String removeValidationOnlyMarkup(String candidate) {
    String withoutMatrixDelimiters = LATEX_MATRIX_DELIMITER.matcher(candidate).replaceAll(" ");
    String withoutPluralMarkers =
        OPTIONAL_PLURAL_MARKER.matcher(withoutMatrixDelimiters).replaceAll("");
    return NUMBERED_LIST_MARKER.matcher(withoutPluralMarkers).replaceAll("$1");
  }

  private static boolean isExplicitAbstention(String candidate) {
    String undecorated = CITATION.matcher(candidate).replaceAll(" ").strip();
    return ABSTENTION_PATTERN.matcher(undecorated).matches()
        || TERMINAL_ABSTENTION_PATTERN.matcher(undecorated).find();
  }

  private static boolean hasOnlySupportedClaims(
      String question, String candidate, List<GroundingDocument> retrieved) {
    Set<String> supported = words(question);
    retrieved.forEach(
        hit -> {
          supported.addAll(words(hit.title()));
          supported.addAll(words(hit.text()));
        });

    String uncited = CITATION.matcher(candidate).replaceAll(" ");
    return words(uncited).stream().allMatch(word -> isSupportedClaimWord(word, supported));
  }

  private static boolean isSupportedClaimWord(String word, Set<String> supported) {
    return containsEquivalentWord(supported, word);
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

  private static boolean lacksNamedEntityOverlap(
      String question, List<GroundingDocument> retrieved) {
    Set<String> questionEntities = new HashSet<>();
    Matcher matcher = NAMED_ENTITY_TOKEN.matcher(question);
    while (matcher.find()) {
      String entity = matcher.group().toLowerCase(Locale.ROOT);
      if (!FUNCTION_WORDS.contains(entity)) {
        questionEntities.add(normalizeWord(entity));
      }
    }
    if (questionEntities.isEmpty()) {
      return false;
    }

    Set<String> evidenceWords = new HashSet<>();
    retrieved.forEach(
        document -> {
          evidenceWords.addAll(words(document.title()));
          evidenceWords.addAll(words(document.text()));
        });
    return questionEntities.stream()
        .noneMatch(entity -> containsEquivalentWord(evidenceWords, entity));
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
    removeEquivalentWords(evidenceWords, questionWords);

    List<Set<String>> questionClauses = clauses(question);
    List<Set<String>> candidateClauses =
        statementClauses(CITATION.matcher(candidate).replaceAll(" "));
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
          removeEquivalentWords(clauseSpecificWords, questionClauses.get(otherIndex));
          removeEquivalentWords(clauseEvidenceWords, matchingEvidenceClauses.get(otherIndex));
        }
      }
      removeEquivalentWords(clauseEvidenceWords, questionWords);
      if (clauseEvidenceWords.isEmpty()) {
        clauseEvidenceWords.addAll(matchingEvidenceClauses.get(clauseIndex));
        removeEquivalentWords(clauseEvidenceWords, questionWords);
      }
      boolean clauseSatisfied = false;
      for (Set<String> candidateClause : candidateClauses) {
        Set<String> candidateEvidenceAnchors = new HashSet<>();
        candidateClause.stream()
            .filter(word -> containsEquivalentWord(clauseEvidenceWords, word))
            .forEach(candidateEvidenceAnchors::add);
        long overlap =
            candidateClause.stream()
                .filter(word -> containsEquivalentWord(questionClause, word))
                .count();
        long clauseSpecificOverlap =
            candidateClause.stream()
                .filter(word -> containsEquivalentWord(clauseSpecificWords, word))
                .count();
        int requiredClauseSpecificOverlap = Math.min(1, clauseSpecificWords.size());
        boolean questionBoundAnchor =
            overlap >= 2
                && clauseSpecificOverlap >= requiredClauseSpecificOverlap
                && candidateEvidenceAnchors.stream()
                    .anyMatch(
                        anchor ->
                            anchor.length() >= 5
                                || anchor.codePoints().allMatch(Character::isDigit));
        if (questionBoundAnchor || hasConcreteEvidenceAnchor(candidateEvidenceAnchors)) {
          clauseSatisfied = true;
          break;
        }
      }
      if (!clauseSatisfied) {
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
      long overlap =
          evidenceClause.stream()
              .filter(word -> containsEquivalentWord(questionClause, word))
              .count();
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

  private static boolean containsEquivalentWord(Set<String> words, String candidate) {
    if (words.contains(candidate)) {
      return true;
    }
    Set<String> candidateForms = inflectionForms(candidate);
    for (String word : words) {
      if (candidateForms.stream().anyMatch(inflectionForms(word)::contains)) {
        return true;
      }
    }
    return false;
  }

  private static void removeEquivalentWords(Set<String> words, Set<String> removals) {
    words.removeIf(word -> containsEquivalentWord(removals, word));
  }

  private static Set<String> inflectionForms(String word) {
    Set<String> forms = new HashSet<>();
    forms.add(word);
    if (word.equals("expiration")) {
      forms.add("expire");
    } else if (word.equals("retention")) {
      forms.add("retain");
    }
    if (word.length() > 5 && word.endsWith("ing")) {
      String withoutIng = word.substring(0, word.length() - 3);
      forms.add(withoutIng);
      forms.add(withoutIng + "e");
      int stemLength = withoutIng.length();
      if (stemLength > 1
          && withoutIng.charAt(stemLength - 1) == withoutIng.charAt(stemLength - 2)) {
        forms.add(withoutIng.substring(0, stemLength - 1));
      }
    }
    if (word.length() > 4 && word.endsWith("ed")) {
      forms.add(word.substring(0, word.length() - 1));
      String withoutEd = word.substring(0, word.length() - 2);
      forms.add(withoutEd);
      int stemLength = withoutEd.length();
      if (stemLength > 1 && withoutEd.charAt(stemLength - 1) == withoutEd.charAt(stemLength - 2)) {
        forms.add(withoutEd.substring(0, stemLength - 1));
      }
    }
    return forms;
  }

  private static String normalizeWord(String word) {
    String number = normalizeNumberWord(word);
    if (!number.equals(word)) {
      return number;
    }
    if (word.equals("thrown")) {
      return "throw";
    }
    if (word.length() > 4 && word.endsWith("ies")) {
      return word.substring(0, word.length() - 3) + "y";
    }
    if (word.length() > 4
        && (word.endsWith("ses")
            || word.endsWith("xes")
            || word.endsWith("zes")
            || word.endsWith("ches")
            || word.endsWith("shes"))) {
      return word.substring(0, word.length() - 2);
    }
    if (word.length() > 3 && word.endsWith("s") && !word.endsWith("ss")) {
      return word.substring(0, word.length() - 1);
    }
    return word;
  }

  private static String normalizeNumberWord(String word) {
    return switch (word) {
      case "zero" -> "0";
      case "one" -> "1";
      case "two" -> "2";
      case "three" -> "3";
      case "four" -> "4";
      case "five" -> "5";
      case "six" -> "6";
      case "seven" -> "7";
      case "eight" -> "8";
      case "nine" -> "9";
      case "ten" -> "10";
      default -> word;
    };
  }

  private static boolean hasOnlyTrustedCitations(
      String candidate, List<GroundingDocument> retrieved) {
    Set<String> trustedIds = new HashSet<>();
    retrieved.forEach(hit -> trustedIds.add(hit.id()));

    boolean foundTrusted = false;
    Matcher matcher = CITATION.matcher(candidate);
    while (matcher.find()) {
      String citation = matcher.group(1).strip();
      if (!trustedIds.contains(citation)) {
        return false;
      }
      foundTrusted = true;
    }
    return foundTrusted;
  }

  private static List<GroundingDocument> citedDocuments(
      String candidate, List<GroundingDocument> retrieved) {
    Set<String> citedIds = new HashSet<>();
    Matcher matcher = CITATION.matcher(candidate);
    while (matcher.find()) {
      citedIds.add(matcher.group(1).strip());
    }
    return retrieved.stream().filter(document -> citedIds.contains(document.id())).toList();
  }

  private static List<GroundingDocument> supportingDocuments(
      String question,
      String candidate,
      List<GroundingDocument> evidenceUniverse,
      List<GroundingDocument> allowedDocuments) {
    Set<String> questionWords = words(question);
    Set<String> evidenceWords = documentWords(evidenceUniverse);
    LinkedHashMap<String, GroundingDocument> supporting = new LinkedHashMap<>();

    String uncited = CITATION.matcher(candidate).replaceAll(" ");
    for (Set<String> claim : clauses(uncited)) {
      Set<String> evidenceBoundWords = new HashSet<>();
      for (String word : claim) {
        if (containsEquivalentWord(evidenceWords, word)) {
          evidenceBoundWords.add(word);
        } else if (!containsEquivalentWord(questionWords, word)) {
          return List.of();
        }
      }
      if (evidenceBoundWords.isEmpty()) {
        continue;
      }

      GroundingDocument document =
          allowedDocuments.stream()
              .filter(
                  candidateDocument -> {
                    Set<String> candidateDocumentWords =
                        words(candidateDocument.title() + "\n" + candidateDocument.text());
                    return evidenceBoundWords.stream()
                        .allMatch(word -> containsEquivalentWord(candidateDocumentWords, word));
                  })
              .findFirst()
              .orElse(null);
      if (document == null) {
        return List.of();
      }
      supporting.putIfAbsent(document.id(), document);
    }
    return List.copyOf(supporting.values());
  }

  private static Set<String> documentWords(List<GroundingDocument> documents) {
    Set<String> result = new HashSet<>();
    documents.forEach(
        document -> {
          result.addAll(words(document.title()));
          result.addAll(words(document.text()));
        });
    return result;
  }

  private static boolean preservesAtomicIdentifiers(
      String candidate, List<GroundingDocument> retrieved) {
    Set<String> candidateIdentifiers = atomicIdentifiers(candidate);
    Set<String> candidateWords = words(candidate);
    for (GroundingDocument document : retrieved) {
      for (String identifier : atomicIdentifiers(document.title() + " " + document.text())) {
        if (candidateIdentifiers.contains(identifier)) {
          continue;
        }
        for (String segment : identifier.split("-")) {
          String normalizedSegment = normalizeWord(segment);
          if (candidateWords.contains(normalizedSegment)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  private static Set<String> atomicIdentifiers(String text) {
    Set<String> identifiers = new HashSet<>();
    Matcher matcher = ATOMIC_IDENTIFIER.matcher(text);
    while (matcher.find()) {
      identifiers.add(matcher.group().toLowerCase(Locale.ROOT));
    }
    return identifiers;
  }

  private static boolean hasCitations(String candidate) {
    return CITATION.matcher(candidate).find();
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

  private String extractiveAnswer(List<GroundingDocument> retrieved) {
    StringBuilder answer = new StringBuilder();
    int documentCount = Math.min(limits.maximumExtractiveDocuments(), retrieved.size());
    for (int index = 0; index < documentCount; index++) {
      GroundingDocument hit = retrieved.get(index);
      String separator = answer.isEmpty() ? "" : " ";
      String citation = " [" + hit.id() + "]";
      int textBudget =
          limits.maximumExtractiveCharacters()
              - answer.length()
              - separator.length()
              - citation.length();
      if (textBudget <= 0) {
        break;
      }

      String text = hit.text().strip();
      if (!text.endsWith(".") && !text.endsWith("!") && !text.endsWith("?")) {
        text += ".";
      }
      if (text.length() > textBudget) {
        text = truncate(text, textBudget);
      }
      answer.append(separator).append(text).append(citation);
    }
    return answer.isEmpty() ? ABSTENTION : answer.toString();
  }

  private static String truncate(String text, int maximumCharacters) {
    if (text.length() <= maximumCharacters) {
      return text;
    }
    if (maximumCharacters <= 3) {
      return text.substring(0, maximumCharacters);
    }
    String prefix = text.substring(0, maximumCharacters - 3).stripTrailing();
    return prefix + "...";
  }

  private static boolean containsPromptInjection(String text) {
    return PROMPT_INJECTION_PATTERNS.stream().anyMatch(pattern -> pattern.matcher(text).find());
  }

  private static void requireInputs(String question, List<GroundingDocument> retrieved) {
    Objects.requireNonNull(question, "question");
    Objects.requireNonNull(retrieved, "retrieved");
    if (question.isBlank()) {
      throw new IllegalArgumentException("question must not be blank");
    }
    retrieved.forEach(document -> Objects.requireNonNull(document, "retrieved document"));
  }
}
