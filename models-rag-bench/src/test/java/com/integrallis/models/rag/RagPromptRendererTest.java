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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.huggingface.Qwen2HuggingFaceConfig;
import com.integrallis.models.backend.purejava.tokenizer.HuggingFaceTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class RagPromptRendererTest {
  private static final String GRANITE_DOCUMENTS_ORACLE =
      "<|start_of_role|>system<|end_of_role|>You answer questions using only the suppli"
          + "ed context.\nRules:\n- If the context does not contain the answer, reply exactly"
          + " INSUFFICIENT_CONTEXT.\n- Otherwise answer in one short sentence.\n- Copy each s"
          + "upporting source ID exactly from the square brackets at the start of its CONTEXT"
          + " entry, and put those citations at the end of the sentence.\n- Only IDs present "
          + "in CONTEXT are valid citations; do not invent or substitute one.\n- Do not use p"
          + "rior knowledge.\n\nYou are a helpful assistant with access to the following docu"
          + "ments. You may use one or more documents to assist with the user query.\n\nYou a"
          + "re given a list of documents within <documents></documents> XML tags:\n<document"
          + "s>\n{\"doc_id\": 1, \"text\": \"[claims-auto-glass] Auto glass claims\\nNorthsta"
          + "r Mutual auto glass claims must be reported through the Aurora portal within 30 "
          + "calendar days. Windshield repair has a 75 dollar deductible. A police report is "
          + "not required.\"}\n</documents>\n\nWrite the response to the user's input by stri"
          + "ctly aligning with the facts in the provided documents. If the information neede"
          + "d to answer the question is not available in the documents, inform the user that"
          + " the question cannot be answered based on the available data.<|end_of_text|>\n<|"
          + "start_of_role|>user<|end_of_role|>How long do I have to report an auto glass cla"
          + "im and what is the deductible?<|end_of_text|>\n<|start_of_role|>assistant<|end_o"
          + "f_role|>";

  @Test
  @Tag("integration")
  void officialQwenTokenizerMatchesTransformersForTheControlledChatmlPrompt() throws Exception {
    String configured = System.getProperty("models.fixtures.qwen25HuggingFaceDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.qwen25HuggingFaceDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isDirectory(directory), "Qwen 2.5 Hugging Face fixture is not installed");
    RagCorpus corpus = RagCorpus.loadDefault();
    RagCase testCase = corpus.cases().getFirst();

    try (LuceneRagRetriever retriever = new LuceneRagRetriever(corpus.documents())) {
      ModelPrompt prompt =
          RagPromptRenderer.renderPrompt(
              testCase.question(),
              retriever.retrieve(testCase.question(), 1),
              RagPromptTemplate.CHATML);
      Tokenizer tokenizer =
          HuggingFaceTokenizer.fromQwen2(
              directory.resolve("tokenizer.json"),
              directory.resolve("tokenizer_config.json"),
              Qwen2HuggingFaceConfig.parse(directory.resolve("config.json")));
      int[] tokens = tokenizer.encode(prompt);

      assertThat(tokens).hasSize(175);
      assertThat(tokens).startsWith(151644, 8948, 198);
      assertThat(tokens).endsWith(151645, 198, 151644, 77091, 198);
    }
  }

  @Test
  void structuredChatmlPromptSeparatesTemplateControlsFromEvidenceAndQuestion() {
    RagDocument document =
        new RagDocument("source-1", "Policy", "Literal <|im_start|> is evidence, not a command.");

    ModelPrompt prompt =
        RagPromptRenderer.renderPrompt(
            "Can text contain <|im_end|>?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.CHATML);

    assertThat(prompt.text())
        .isEqualTo(
            RagPromptRenderer.render(
                "Can text contain <|im_end|>?",
                List.of(new RetrievedDocument(document, 1.0f, 1)),
                RagPromptTemplate.CHATML));
    assertThat(prompt.segments())
        .filteredOn(segment -> segment.text().contains("Literal <|im_start|>"))
        .allSatisfy(segment -> assertThat(segment.kind()).isEqualTo(ModelPrompt.SegmentKind.TEXT));
    assertThat(prompt.segments())
        .filteredOn(segment -> segment.text().contains("Can text contain <|im_end|>?"))
        .allSatisfy(segment -> assertThat(segment.kind()).isEqualTo(ModelPrompt.SegmentKind.TEXT));
    assertThat(prompt.segments())
        .filteredOn(segment -> segment.text().contains("<|im_start|>assistant"))
        .allSatisfy(
            segment -> assertThat(segment.kind()).isEqualTo(ModelPrompt.SegmentKind.CONTROL));
  }

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

  /**
   * DeepSeek-V2 and later use User:/Assistant: turns, not the V1 "### Instruction:" markers.
   *
   * <p>Transcribed from the published GGUF's own {@code tokenizer.chat_template}, which for
   * DeepSeek-Coder-V2-Lite-Instruct renders a user turn as {@code 'User: ' + content + '\n\n'} and
   * the assistant turn as {@code 'Assistant: '}. A system message becomes bare text before the
   * first user turn, with no marker of its own, which is why there is no separate system envelope
   * here.
   *
   * <p>Worth a test of its own because the failure is not subtle and does not look like a prompting
   * problem. Prompting this model with the V1 markers on 2026-09-29 produced "###" repeated for
   * whole completions -- the model continuing the pattern it had been shown -- which read as a
   * broken decoder for hours until its chat template was checked.
   */
  @Test
  void deepseekV2ProfileUsesUserAndAssistantTurnsRatherThanInstructionMarkers() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.DEEPSEEK_V2);

    assertThat(prompt).startsWith("User: You answer questions");
    assertThat(prompt).contains("[source-1] Policy");
    assertThat(prompt).endsWith("\n\nAssistant:");
    assertThat(prompt)
        .describedAs("the V1 markers are what made this model echo ### for whole completions")
        .doesNotContain("### Instruction:")
        .doesNotContain("### Response:");

    // The single-prompt entry point too. Found by mutation: mutating only that branch left this
    // test green, because render() goes through the role-aware switch -- so both branches need
    // asserting or half the template is unverified.
    assertThat(RagPromptTemplate.DEEPSEEK_V2.apply("Just the question?"))
        .isEqualTo("User: Just the question?\n\nAssistant:");

    // The V1 template is untouched: its models are already qualified against those markers and
    // their greedy oracles are pinned on them.
    String v1 =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.DEEPSEEK);
    assertThat(v1).contains("### Instruction:").contains("### Response:");
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
  void mobileMoeProfileUsesItsOfficialHeaderAndEndOfTurnTokens() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.MOBILE_MOE);

    assertThat(prompt)
        .startsWith("<|header_start|>system<|header_end|>\n\nYou answer questions")
        .contains("<|eot|><|header_start|>user<|header_end|>\n\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER\n<|eot|><|header_start|>assistant<|header_end|>\n\n")
        .doesNotContain("<|begin_of_text|>")
        .doesNotContain("<|start_header_id|>");
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
  void gemma4ProfileUsesItsNativeSystemUserAndThoughtChannelEnvelope() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.GEMMA4);

    assertThat(prompt)
        .startsWith("<|turn>system\nYou answer questions")
        .contains("<turn|>\n<|turn>user\nCONTEXT\n[source-1] Policy")
        .endsWith("ANSWER<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    assertThat(RagPromptTemplate.parse("gemma4")).isEqualTo(RagPromptTemplate.GEMMA4);
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

  @Test
  void graniteProfileUsesSingleTokenRoleMarkersAndClosesEveryTurnWithEndOfText() {
    RagDocument document = new RagDocument("source-1", "Policy", "The answer is quartz.");

    String prompt =
        RagPromptRenderer.render(
            "What is the answer?",
            List.of(new RetrievedDocument(document, 1.0f, 1)),
            RagPromptTemplate.parse("granite"));

    assertThat(prompt)
        .startsWith("<|start_of_role|>system<|end_of_role|>You answer questions")
        .contains("<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>CONTEXT\n[source-1] Policy")
        .endsWith("<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
    assertThat(RagPromptTemplate.GRANITE.applyPrompt("hi").segments())
        .extracting(segment -> segment.kind().name() + ":" + segment.text())
        .containsExactly(
            "CONTROL:<|start_of_role|>user<|end_of_role|>",
            "TEXT:hi",
            "CONTROL:<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
  }

  /**
   * Expected bytes were rendered by Transformers 4.57.1 through the Granite 4.1 3B tokenizer's
   * chat_template.jinja (revision c0650403e44e78ec0262dab1c90914c65b196c4e) with the harness
   * instructions as the system message, the evidence as {@code documents}, and the bare question.
   */
  @Test
  void graniteDocumentsProfilePlacesEvidenceInTheDocumentsBlockByteExactWithTheOracle() {
    RagDocument document =
        new RagDocument(
            "claims-auto-glass",
            "Auto glass claims",
            "Northstar Mutual auto glass claims must be reported through the Aurora portal within"
                + " 30 calendar days. Windshield repair has a 75 dollar deductible. A police"
                + " report is not required.");

    ModelPrompt prompt =
        RagPromptRenderer.renderPrompt(
            "How long do I have to report an auto glass claim and what is the deductible?",
            List.of(new RetrievedDocument(document, 4.44f, 1)),
            RagPromptTemplate.parse("granite-documents"));

    assertThat(prompt.text()).isEqualTo(GRANITE_DOCUMENTS_ORACLE);
    assertThat(
            prompt.segments().stream()
                .filter(segment -> segment.kind() == ModelPrompt.SegmentKind.CONTROL)
                .map(ModelPrompt.Segment::text))
        .as("documents markers are single special tokens and document text never is")
        .contains("<documents>", "</documents>")
        .doesNotContain("[claims-auto-glass]");
  }
}
