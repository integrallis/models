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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Real-artifact preflight for IBM's Granite 3.2 query-rewrite aLoRA candidate. */
@Tag("integration")
@EnabledIfSystemProperty(named = GraniteAloraIntegrationTest.BASE_PROPERTY, matches = ".+")
class GraniteAloraIntegrationTest {
  static final String BASE_PROPERTY = "models.fixtures.granite32AloraBase";
  static final String ADAPTER_PROPERTY = "models.fixtures.granite32AloraAdapter";
  static final String INVOCATION =
      "<|start_of_role|>rewrite: Reword the final utterance from the USER into a single "
          + "utterance that doesn't need the prior conversation history to understand the user's "
          + "intent. If the final utterance is a clear and standalone question, please DO NOT "
          + "attempt to rewrite it, rather output the last utterance as is. Your output format "
          + "should be in JSON: { \"rewritten_question\": <REWRITE> }<|end_of_role|>";
  private static final int[] INVOCATION_TOKENS = {
    49152, 22179, 44, 24120, 655, 322, 1158, 46879, 723, 645, 322, 14126, 1991, 312, 3982, 46879,
    723, 688, 4163, 1330, 1849, 322, 9553, 19509, 8142, 372, 7648, 322, 1256, 1182, 8927, 32, 1670,
    322, 1158, 46879, 723, 438, 312, 4233, 461, 27933, 7000, 30, 4322, 4085, 2370, 11549, 372,
    21817, 561, 30, 9283, 1688, 322, 2401, 46879, 723, 619, 438, 32, 10604, 1688, 2179, 1395, 526,
    328, 3398, 44, 301, 313, 268, 15557, 81, 4594, 563, 333, 3722, 48, 320, 49153
  };

  @Test
  void loadsGraniteAndPinsTheUpstreamActivationMarkerToItsTokenizer() {
    try (PureJavaBackend backend =
        PureJavaBackend.load(Path.of(System.getProperty(BASE_PROPERTY)))) {
      Tokenizer tokenizer = backend.tokenizer();
      int[] invocationTokens = tokenizer.encodeControl(INVOCATION);
      int[] renderedPromptTokens =
          tokenizer.encode(
              ModelPrompt.builder()
                  .control("<|start_of_role|>system<|end_of_role|><|end_of_text|>\n")
                  .control(
                      "<|start_of_role|>user<|end_of_role|>Who is the CEO of Apple?<|end_of_text|>\n")
                  .control(
                      "<|start_of_role|>assistant<|end_of_role|>Tim Cook is the CEO of Apple.<|end_of_text|>\n")
                  .control(
                      "<|start_of_role|>user<|end_of_role|>and for Microsoft?<|end_of_text|>\n")
                  .control(INVOCATION)
                  .build());

      assertThat(backend.metadata().modelFamily()).isEqualTo("granite");
      assertThat(invocationTokens).containsExactly(INVOCATION_TOKENS);
      assertThat(tokenizer.decode(invocationTokens)).isEqualTo(INVOCATION);
      assertThat(lastIndexOf(renderedPromptTokens, INVOCATION_TOKENS)).isGreaterThan(0);
    }
  }

  @Test
  @EnabledIfSystemProperty(named = ADAPTER_PROPERTY, matches = ".+")
  void opensTheActualUpstreamAdapterAtThePinnedPromptBoundary() {
    ModelPrompt prompt =
        ModelPrompt.builder()
            .control("<|start_of_role|>system<|end_of_role|><|end_of_text|>\n")
            .control(
                "<|start_of_role|>user<|end_of_role|>Who is the CEO of Apple?<|end_of_text|>\n")
            .control(
                "<|start_of_role|>assistant<|end_of_role|>Tim Cook is the CEO of Apple.<|end_of_text|>\n")
            .control("<|start_of_role|>user<|end_of_role|>and for Microsoft?<|end_of_text|>\n")
            .control(INVOCATION)
            .build();
    try (PureJavaBackend backend =
            PureJavaBackend.loadActivatedAdapter(
                Path.of(System.getProperty(BASE_PROPERTY)),
                Path.of(System.getProperty(ADAPTER_PROPERTY)));
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn =
            model.openToolTurn(prompt, ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
      assertThat(model.adapter().invocationTokens())
          .containsExactly(
              java.util.Arrays.stream(INVOCATION_TOKENS).boxed().toArray(Integer[]::new));
      assertThat(model.adapter().invocationText()).isEqualTo(INVOCATION);
      assertThat(turn.physicallySharesPrefix()).isTrue();
      assertThat(turn.sharedPrefixTokens()).isGreaterThan(0);
    }
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
