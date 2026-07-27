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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class RagPromptRendererTest {

  @Test
  void promptCarriesStrictGroundingRuleSourceIdsAndQuestion() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?", List.of(new RetrievedDocument(document, 1.0f, 1)));

    assertThat(prompt)
        .contains("reply exactly INSUFFICIENT_CONTEXT")
        .contains("Copy each supporting source ID exactly")
        .contains("[source-1] Policy")
        .contains("The answer is quartz.")
        .contains("QUESTION\nWhat is the answer?\n\nANSWER\n")
        .doesNotContain("[source-id]")
        .doesNotContain("null");
  }

  @Test
  void chatmlProfileUsesNativeSystemAndUserTurns() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.CHATML);

    assertThat(prompt)
        .startsWith("<|im_start|>system\nYou answer questions")
        .contains("<|im_end|>\n<|im_start|>user\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER\n<|im_end|>\n<|im_start|>assistant\n");
  }

  @Test
  void chatmlNoThinkProfilePrefillsAnEmptyReasoningBlock() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.CHATML_NO_THINK);

    assertThat(prompt)
        .startsWith("<|im_start|>system\nYou answer questions")
        .contains("<|im_end|>\n<|im_start|>user\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  }

  @Test
  void chatmlDirectProfilePrefillsAConciseAnswerLeadIn() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.parse("chatml-direct"));

    assertThat(prompt)
        .startsWith("<|im_start|>system\nYou answer questions")
        .contains("<|im_end|>\n<|im_start|>user\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER\n<|im_end|>\n<|im_start|>assistant\nThe context states that ");
  }

  @Test
  void chatmlAnswerProfilePrefillsOnlyTheAnswerLabel() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.parse("chatml-answer"));

    assertThat(prompt)
        .startsWith("<|im_start|>system\nYou answer questions")
        .contains("<|im_end|>\n<|im_start|>user\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER\n<|im_end|>\n<|im_start|>assistant\nAnswer: ");
  }

  @Test
  void zephyrProfileUsesNativeSystemAndUserTurns() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.ZEPHYR);

    assertThat(prompt)
        .startsWith("<|system|>\nYou answer questions")
        .contains("</s>\n<|user|>\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER\n</s>\n<|assistant|>");
  }

  @Test
  void llama3ProfileUsesHeaderAndEndOfTurnTokensWithoutDuplicatingBos() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.LLAMA3);

    assertThat(prompt)
        .startsWith("<|start_header_id|>system<|end_header_id|>\n\nYou answer questions")
        .contains(
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCONTEXT\n" + "[source-1] Policy")
        .endsWith("ANSWER<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        .doesNotContain("<|begin_of_text|>");
  }

  @Test
  void gemmaProfileMergesSystemInstructionsIntoTheUserTurn() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.GEMMA);

    assertThat(prompt)
        .startsWith("<start_of_turn>user\nYou answer questions")
        .contains(
            "Do not use prior knowledge.\n\nCONTEXT\n[source-1] Policy\nThe answer is quartz.")
        .endsWith("ANSWER<end_of_turn>\n<start_of_turn>model\n")
        .doesNotContain("<start_of_turn>system")
        .doesNotContain("<bos>");
  }

  @Test
  void phi3ProfileUsesRoleAndEndTokens() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.PHI3);

    assertThat(prompt)
        .startsWith("<|system|>\nYou answer questions")
        .contains("<|end|>\n<|user|>\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER<|end|>\n<|assistant|>\n");
  }

  @Test
  void deepseekCoderProfileUsesInstructionAndResponseSections() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.DEEPSEEK);

    assertThat(prompt)
        .startsWith("### Instruction:\nYou answer questions")
        .contains("Do not use prior knowledge.\n\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER\n\n### Response:\n");
  }

  @Test
  void h2oProfileUsesPromptAndAnswerTokens() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.parse("h2o"));

    assertThat(prompt)
        .startsWith("<|prompt|>You answer questions")
        .contains("Do not use prior knowledge.\n\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER</s><|answer|>");
  }

  @Test
  void h2oDirectProfilePrefillsAConciseAnswerLeadIn() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.parse("h2o-direct"));

    assertThat(prompt)
        .startsWith("<|prompt|>You answer questions")
        .endsWith("ANSWER</s><|answer|>The context states that ");
  }

  @Test
  void miniCpm5NoThinkProfileEmitsTheTemplateOwnedBosToken() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.MINICPM5_NO_THINK);

    assertThat(prompt)
        .startsWith("<s><|im_start|>system\nYou answer questions")
        .contains("<|im_end|>\n<|im_start|>user\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  }
}
