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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/** Audits visible facts and completion separately without rewriting original benchmark evidence. */
public final class RagVisibleAnswerAudit {
  public static final String POLICY_ID = "visible-answer-completion-and-bounded-facts-v1";
  private static final Pattern TAG = Pattern.compile("(?i)<(/?)(think|thinking|reasoning)>");

  private RagVisibleAnswerAudit() {}

  public record Assessment(
      String caseId,
      String visibleText,
      String reasoningText,
      boolean reasoningComplete,
      boolean budgetReached,
      boolean completeVisibleAnswer,
      boolean polarityConsistent,
      RagEvaluation visibleEvaluation,
      boolean requiredFactsAndCitationsPass) {}

  public record Audit(
      String policyId,
      String generatedAt,
      boolean newInference,
      String sourceReport,
      String sourceReportSha256,
      String sourceBackendVersion,
      String assessmentGroundingPolicy,
      String artifactSha256,
      String workload,
      String corpusSha256,
      int attempts,
      long distinctQuestions,
      long completeVisibleAnswers,
      long requiredFactsAndCitationsPasses,
      long budgetReached,
      String limitation,
      List<Assessment> assessments) {
    public Audit {
      assessments = List.copyOf(assessments);
    }
  }

  static Assessment assess(
      RagCase question,
      List<RetrievedDocument> retrieved,
      String raw,
      int outputTokens,
      int maxTokens) {
    var visible = new StringBuilder();
    var reasoning = new StringBuilder();
    var stack = new ArrayDeque<String>();
    var tags = TAG.matcher(raw);
    int cursor = 0;
    boolean malformed = false;
    while (tags.find()) {
      (stack.isEmpty() ? visible : reasoning).append(raw, cursor, tags.start());
      String name = tags.group(2).toLowerCase(Locale.ROOT);
      if (tags.group(1).isEmpty()) stack.push(name);
      else if (stack.isEmpty() || !stack.pop().equals(name)) {
        malformed = true;
        break;
      }
      cursor = tags.end();
    }
    if (!malformed) (stack.isEmpty() ? visible : reasoning).append(raw, cursor, raw.length());
    String answer = malformed ? "" : visible.toString().strip();
    boolean closed = !malformed && stack.isEmpty();
    boolean capped = outputTokens >= maxTokens;
    boolean complete = closed && !capped && !answer.isBlank();
    var evaluation = RagEvaluator.evaluateVisible(question, retrieved, answer);
    boolean polarityConsistent =
        GroundedAnswerPolicy.preservesClaimPolarity(
            answer,
            retrieved.stream()
                .map(
                    hit ->
                        new GroundingDocument(
                            hit.document().id(),
                            hit.document().title(),
                            hit.document().text(),
                            hit.score(),
                            hit.rank()))
                .toList());
    return new Assessment(
        question.id(),
        answer,
        reasoning.toString(),
        closed,
        capped,
        complete,
        polarityConsistent,
        evaluation,
        complete && polarityConsistent && evaluation.correct());
  }

  /** Usage: RagVisibleAnswerAudit INPUT_REPORT NEW_AUDIT_REPORT. Runs no inference. */
  public static void main(String[] args) throws Exception {
    if (args.length != 2)
      throw new IllegalArgumentException(
          "usage: RagVisibleAnswerAudit INPUT_REPORT NEW_AUDIT_REPORT");
    Path input = Path.of(args[0]);
    Path output = Path.of(args[1]);
    if (Files.exists(output))
      throw new IllegalArgumentException("audit output must be a new file: " + output);
    byte[] bytes = Files.readAllBytes(input);
    var mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    var report = mapper.readValue(bytes, RagBenchmarkReport.class);
    if (report.schemaVersion() != RagBenchmarkReport.CURRENT_SCHEMA_VERSION)
      throw new IllegalArgumentException("unsupported report schema");
    var corpus = RagCorpus.load(RagWorkload.parse(report.settings().workload()));
    if (!corpus.fingerprint().equals(report.settings().corpusSha256()))
      throw new IllegalArgumentException("report corpus differs from this scorer's pinned corpus");
    var assessments = new ArrayList<Assessment>();
    for (var run : report.runs()) {
      var question =
          corpus.cases().stream()
              .filter(c -> c.id().equals(run.caseId()))
              .findFirst()
              .orElseThrow();
      assessments.add(
          assess(
              question,
              run.retrieved(),
              run.grounding().rawText(),
              run.generation().outputTokens(),
              report.settings().maxOutputTokens()));
    }
    var audit =
        new Audit(
            POLICY_ID,
            Instant.now().toString(),
            false,
            input.toString(),
            HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)),
            report.backendVersion(),
            GroundedAnswerPolicy.POLICY_ID,
            report.artifactSha256(),
            report.settings().workload(),
            corpus.fingerprint(),
            assessments.size(),
            assessments.stream().map(Assessment::caseId).distinct().count(),
            assessments.stream().filter(Assessment::completeVisibleAnswer).count(),
            assessments.stream().filter(Assessment::requiredFactsAndCitationsPass).count(),
            assessments.stream().filter(Assessment::budgetReached).count(),
            "Deterministic required-fact/citation matching is not a general hallucination or semantic-entailment judge. Budget reached is conservative: reports lack an explicit finish reason. Repetitions are not independent accuracy samples.",
            List.copyOf(assessments));
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) Files.createDirectories(parent);
    Files.write(output, mapper.writeValueAsBytes(audit), StandardOpenOption.CREATE_NEW);
  }
}
