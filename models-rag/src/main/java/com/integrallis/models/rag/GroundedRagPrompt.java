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

import java.util.List;
import java.util.Objects;

/** A retrieval-screened, framework-neutral prompt for grounded generation. */
public final class GroundedRagPrompt {
  private static final String INSTRUCTIONS =
      "You answer questions using only the supplied context.\n"
          + "Rules:\n"
          + "- If the context does not contain the answer, reply exactly INSUFFICIENT_CONTEXT.\n"
          + "- Otherwise answer in one short sentence.\n"
          + "- Copy each supporting source ID exactly from the square brackets at the start of "
          + "its CONTEXT entry, and put those citations at the end of the sentence.\n"
          + "- Only IDs present in CONTEXT are valid citations; do not invent or substitute one.\n"
          + "- Do not use prior knowledge.\n\n";

  private final GroundingContextDecision decision;
  private final String request;

  private GroundedRagPrompt(GroundingContextDecision decision, String request) {
    this.decision = Objects.requireNonNull(decision, "decision");
    this.request = request;
  }

  /**
   * Screens retrieved evidence and prepares the canonical prompt only when generation is allowed.
   *
   * <p>Call {@link #generationAllowed()} before reading the prompt. The prompt accessors throw when
   * retrieval was rejected, which prevents unsafe context from being passed to a model by mistake.
   */
  public static GroundedRagPrompt prepare(
      GroundedAnswerPolicy policy, String question, List<GroundingDocument> evidence) {
    Objects.requireNonNull(policy, "policy");
    List<GroundingDocument> copiedEvidence = List.copyOf(evidence);
    GroundingContextDecision decision = policy.assess(question, copiedEvidence);
    return decision.generationAllowed()
        ? new GroundedRagPrompt(decision, renderRequest(question, copiedEvidence))
        : new GroundedRagPrompt(decision, null);
  }

  static GroundedRagPrompt formatUnchecked(String question, List<GroundingDocument> evidence) {
    Objects.requireNonNull(question, "question");
    List<GroundingDocument> copiedEvidence = List.copyOf(evidence);
    if (question.isBlank()) {
      throw new IllegalArgumentException("question must not be blank");
    }
    return new GroundedRagPrompt(
        GroundingContextDecision.ACCEPTED, renderRequest(question, copiedEvidence));
  }

  /** The retrieval decision made before prompt construction. */
  public GroundingContextDecision decision() {
    return decision;
  }

  /** Whether the evidence passed retrieval validation and prompt text is available. */
  public boolean generationAllowed() {
    return decision.generationAllowed();
  }

  /** Canonical system instructions, suitable for a framework system message. */
  public String instructions() {
    requireGenerationAllowed();
    return INSTRUCTIONS;
  }

  /** Canonical context and question, suitable for a framework user message. */
  public String request() {
    requireGenerationAllowed();
    return request;
  }

  /** Canonical raw prompt for runtimes that accept one text string. */
  public String text() {
    return instructions() + request();
  }

  private void requireGenerationAllowed() {
    if (!generationAllowed()) {
      throw new IllegalStateException(
          "RAG prompt is unavailable because retrieved evidence was rejected: " + decision);
    }
  }

  private static String renderRequest(String question, List<GroundingDocument> evidence) {
    StringBuilder prompt = new StringBuilder("CONTEXT\n");
    for (GroundingDocument document : evidence) {
      prompt
          .append('[')
          .append(document.id())
          .append("] ")
          .append(document.title())
          .append('\n')
          .append(document.text())
          .append("\n\n");
    }
    return prompt.append("QUESTION\n").append(question).append("\n\nANSWER\n").toString();
  }
}
