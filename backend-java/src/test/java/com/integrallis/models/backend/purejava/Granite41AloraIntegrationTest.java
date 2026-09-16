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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Real-artifact gates 1 and 2 of the Granite 4.1 3B activated-adapter hybrid candidate.
 *
 * <p>The pinned expectations come from llama.cpp {@code b9960-a935fbffe} on the exact {@code
 * granite-4.1-3b-Q4_K_M.gguf} artifact and were frozen in {@code
 * benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid/preflight.md} before this test first
 * ran. The adapter test opens IBM's upstream answerability aLoRA at the assistant marker over a
 * documents prompt rendered exactly as the published Granite 4.1 chat template renders it.
 */
@Tag("integration")
@EnabledIfSystemProperty(named = Granite41AloraIntegrationTest.BASE_PROPERTY, matches = ".+")
class Granite41AloraIntegrationTest {
  static final String BASE_PROPERTY = "models.fixtures.granite41AloraBase";
  static final String ADAPTER_PROPERTY = "models.fixtures.granite41AloraAdapter";
  static final String INVOCATION = "<|start_of_role|>assistant<|end_of_role|>";
  private static final int[] INVOCATION_TOKENS = {100264, 78191, 100265};
  private static final int[] ORACLE_PROMPT_TOKENS = {791, 4062, 14198, 39935};
  private static final int[] ORACLE_GREEDY_TOKENS = {35308, 927, 279, 16053, 5679, 1210, 578, 734};

  /**
   * A second oracle prompt with digits, quotes, a percent sign, and a thousands separator. The
   * first prompt cannot expose a missing pre-tokenizer; this one splits digits in groups of three
   * and keeps a space attached to punctuation exactly as the dbrx pattern requires.
   */
  private static final String ORACLE_PUNCTUATION_PROMPT =
      "In the 1850s, \"Islamist\" parties won 75% of 1,024 seats.";

  private static final int[] ORACLE_PUNCTUATION_PROMPT_TOKENS = {
    644, 279, 220, 9741, 15, 82, 11, 330, 94893, 380, 1, 9875, 2834, 220, 2075, 4, 315, 220, 16, 11,
    19592, 16712, 13
  };
  private static final int[] ORACLE_PUNCTUATION_GREEDY_TOKENS = {
    763, 279, 220, 1049, 15, 82, 11, 330
  };

  private static final int[] DOCUMENTS_MARKER_TOKENS = {100282, 100283};
  private static final String DOCUMENT =
      "{\"doc_id\": 1, \"text\": \"Tim Cook has served as the chief executive officer of Apple "
          + "since August 2011, when he succeeded Steve Jobs.\"}";

  @Test
  void matchesTheLlamaCppOracleAndPinsTheMarkerToTheGraniteTokenizer() {
    ModelOracleTestSupport.assertPanamaEnabled();
    try (PureJavaBackend backend =
        PureJavaBackend.load(Path.of(System.getProperty(BASE_PROPERTY)))) {
      Tokenizer tokenizer = backend.tokenizer();

      assertThat(backend.metadata().modelFamily()).isEqualTo("granite");

      int[] promptTokens = tokenizer.encode("The quick brown fox");
      assertThat(promptTokens)
          .as("Granite 4.1 must tokenize the oracle prompt identically and add no BOS")
          .containsExactly(ORACLE_PROMPT_TOKENS);
      assertThat(ModelOracleTestSupport.greedyTokens(backend, promptTokens, 8))
          .as("greedy token IDs must match llama.cpp b9960 for the pinned Granite 4.1 3B GGUF")
          .containsExactly(ORACLE_GREEDY_TOKENS);

      int[] punctuationTokens = tokenizer.encode(ORACLE_PUNCTUATION_PROMPT);
      assertThat(punctuationTokens)
          .as("the dbrx pre-tokenizer must split digits by three and keep space-punctuation runs")
          .containsExactly(ORACLE_PUNCTUATION_PROMPT_TOKENS);
      assertThat(ModelOracleTestSupport.greedyTokens(backend, punctuationTokens, 8))
          .as("greedy IDs after a digit-heavy prompt must match llama.cpp b9960")
          .containsExactly(ORACLE_PUNCTUATION_GREEDY_TOKENS);

      try (var session = backend.openSession()) {
        float[] logits = backend.prefill(session, punctuationTokens, 0);
        int[] generated = new int[8];
        int position = punctuationTokens.length;
        for (int index = 0; index < generated.length; index++) {
          generated[index] = argmax(logits);
          if (index + 1 < generated.length) {
            logits = backend.forward(session, generated[index], position++);
          }
        }
        assertThat(generated)
            .as("a batched session prefill must continue greedily exactly like the oracle")
            .containsExactly(ORACLE_PUNCTUATION_GREEDY_TOKENS);
      }

      int[] invocationTokens = tokenizer.encodeControl(INVOCATION);
      assertThat(invocationTokens).containsExactly(INVOCATION_TOKENS);
      assertThat(tokenizer.decode(invocationTokens)).isEqualTo(INVOCATION);

      assertThat(tokenizer.encodeControl("<documents></documents>"))
          .as("Granite 4.1 declares the documents markers as special tokens")
          .containsExactly(DOCUMENTS_MARKER_TOKENS);
      int[] renderedPromptTokens = tokenizer.encode(answerabilityPrompt());
      assertThat(lastIndexOf(renderedPromptTokens, INVOCATION_TOKENS))
          .as("the marker must close the rendered documents prompt")
          .isEqualTo(renderedPromptTokens.length - INVOCATION_TOKENS.length);
      assertThat(lastIndexOf(renderedPromptTokens, DOCUMENTS_MARKER_TOKENS))
          .as("the empty <documents></documents> pair must render as the two special tokens")
          .isGreaterThan(0);
    }
  }

  @Test
  @EnabledIfSystemProperty(named = ADAPTER_PROPERTY, matches = ".+")
  void opensTheUpstreamAnswerabilityAdapterAtTheAssistantMarker() {
    try (PureJavaBackend backend =
            PureJavaBackend.loadActivatedAdapter(
                Path.of(System.getProperty(BASE_PROPERTY)),
                Path.of(System.getProperty(ADAPTER_PROPERTY)));
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn =
            model.openToolTurn(
                answerabilityPrompt(), ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
      assertThat(model.adapter().invocationTokens())
          .containsExactly(Arrays.stream(INVOCATION_TOKENS).boxed().toArray(Integer[]::new));
      assertThat(model.adapter().invocationText()).isEqualTo(INVOCATION);
      assertThat(model.adapter().rank()).isEqualTo(16);
      assertThat(turn.physicallySharesPrefix()).isTrue();
      assertThat(turn.sharedPrefixTokens()).isGreaterThan(0);
    }
  }

  /** Mirrors the published documents template: markers are control, everything else is text. */
  static ModelPrompt answerabilityPrompt() {
    return ModelPrompt.builder()
        .control("<|start_of_role|>system<|end_of_role|>")
        .text(
            "You are a helpful assistant with access to the following documents. You may use one "
                + "or more documents to assist with the user query.\n\n"
                + "You are given a list of documents within ")
        .control("<documents></documents>")
        .text(" XML tags:\n")
        .control("<documents>")
        .text("\n" + DOCUMENT + "\n")
        .control("</documents>")
        .text(
            "\n\nWrite the response to the user's input by strictly aligning with the facts in "
                + "the provided documents. If the information needed to answer the question is "
                + "not available in the documents, inform the user that the question cannot be "
                + "answered based on the available data.")
        .control("<|end_of_text|>\n")
        .control("<|start_of_role|>user<|end_of_role|>")
        .text("Who is the CEO of Apple?")
        .control("<|end_of_text|>\n")
        .control(INVOCATION)
        .build();
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static int lastIndexOf(int[] values, int[] target) {
    for (int start = values.length - target.length; start >= 0; start--) {
      boolean matches = true;
      for (int offset = 0; offset < target.length; offset++) {
        if (values[start + offset] != target[offset]) {
          matches = false;
          break;
        }
      }
      if (matches) return start;
    }
    return -1;
  }
}
