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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * EdgeRock's prompt envelope, as the publisher ships it.
 *
 * <p>There are two template enums: {@code ChatTemplate} in models-runtime and this one, which is
 * what {@code RagBenchmarkCli} parses {@code --prompt-template} through. Adding the value to the
 * former alone left the benchmark rejecting {@code edgerock} at argument parsing, after the
 * qualification host was already provisioned -- so both halves are asserted here.
 */
@Tag("unit")
class RagPromptTemplateEdgeRockTest {

  @Test
  void theIdentifierTheBenchmarkParsesResolves() {
    assertThat(RagPromptTemplate.parse("edgerock")).isEqualTo(RagPromptTemplate.EDGEROCK);
    assertThat(RagPromptTemplate.EDGEROCK.id()).isEqualTo("edgerock");
  }

  @Test
  void theAssistantTurnIsOpenedIntoAThinkingBlockNotClosedOutOfOne() {
    String rendered = RagPromptTemplate.EDGEROCK.apply("What is the capital of France?");

    assertThat(rendered)
        .as("ChatML with the assistant turn opened into <think>, per Modelfile.Q4_K_M")
        .isEqualTo(
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n");
    // The distinction that matters: CHATML_NO_THINK closes an empty block, which instructs the
    // opposite, and CHATML leaves the choice to the model.
    assertThat(rendered).doesNotContain("</think>");
    assertThat(RagPromptTemplate.CHATML_NO_THINK.apply("x")).contains("<think>\n\n</think>");
    assertThat(RagPromptTemplate.CHATML.apply("x")).doesNotContain("<think>");
  }

  @Test
  void theRoleAwareEnvelopeAlsoOpensTheThinkingBlock() {
    String rendered = RagPromptTemplate.EDGEROCK.apply("Use the context.", "Who is it?");

    assertThat(rendered).startsWith("<|im_start|>system\nUse the context.<|im_end|>\n");
    assertThat(rendered).endsWith("<|im_start|>assistant\n<think>\n");
  }
}
