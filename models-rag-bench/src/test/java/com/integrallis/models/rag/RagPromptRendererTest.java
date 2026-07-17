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
        .contains("[source-1] Policy")
        .contains("The answer is quartz.")
        .contains("QUESTION\nWhat is the answer?\n\nANSWER\n")
        .doesNotContain("null");
  }

  @Test
  void chatmlProfileWrapsTheCanonicalPromptAsAUserTurn() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.CHATML);

    assertThat(prompt)
        .startsWith("<|im_start|>user\nYou answer questions")
        .contains("[source-1] Policy")
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
        .endsWith("ANSWER\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  }
}
