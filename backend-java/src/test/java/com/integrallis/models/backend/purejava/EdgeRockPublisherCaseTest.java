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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * EdgeRock against the publisher's own functional cases, run on our decoder.
 *
 * <p>The cases and their expected answers come from {@code receipts/GGUF-Q4_K_M.json} in {@code
 * hkust-aaron/EdgeRock-4B-GGUF}; the prompts are extracted verbatim. The publisher's recorded
 * outputs are CUDA runs and their timings are therefore <b>theirs, not ours</b> -- only the
 * expected answers are used here.
 *
 * <p>The two long cases are the reason this test is worth more than a short continuation. At 9664
 * and 10462 input tokens against a 1024-token sliding window, the identifier being asked for sits
 * thousands of tokens outside anything the attention layers can see. Nothing but the Rock Memory
 * Anchors can carry it. A decoder with a broken memory module can still answer the arithmetic case
 * and will fail these, which is exactly the discrimination a 16-token continuation cannot make.
 */
@Tag("integration")
@EnabledIfSystemProperty(named = EdgeRockPublisherCaseTest.FIXTURE_PROPERTY, matches = ".+")
@EnabledIfSystemProperty(named = EdgeRockPublisherCaseTest.CASES_PROPERTY, matches = ".+")
class EdgeRockPublisherCaseTest {

  static final String FIXTURE_PROPERTY = "models.fixtures.edgeRock4bGguf";
  static final String CASES_PROPERTY = "models.fixtures.edgeRockCases";

  /** From Modelfile.Q4_K_M: ChatML, with the assistant turn opened into a thinking block. */
  private static String render(String user) {
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
  }

  @Test
  void answersTheShortCaseThatNeedsNoMemory() throws IOException {
    // 24 input tokens: inside the sliding window, so this passes with or without working anchors.
    // It is here to separate "the decoder is broken" from "the memory is broken" when a long case
    // fails -- on its own it is not evidence about the memory at all.
    assertCase("arithmetic", 512);
  }

  @Test
  void recallsAnIdentifierFromOutsideTheAttentionWindow() throws IOException {
    assertCase("memory_update_8k", 640);
  }

  @Test
  void recallsTheRightOneOfTwoCompetingKeys() throws IOException {
    assertCase("memory_two_keys_8k", 640);
  }

  private void assertCase(String id, int maxTokens) throws IOException {
    Path cases = Path.of(System.getProperty(CASES_PROPERTY));
    Path promptFile = cases.resolve(id + ".prompt");
    Path expectedFile = cases.resolve(id + ".expected");
    assertThat(promptFile).as("extracted prompt for %s", id).exists();

    String prompt = Files.readString(promptFile);
    String expected = Files.readString(expectedFile).trim();

    Path model = Path.of(System.getProperty(FIXTURE_PROPERTY));
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    // The publisher runs these at --ctx-size 16384; the longest is 10462 input tokens.
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "16384");
    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      int[] tokens = backend.tokenizer().encode(render(prompt));
      long start = System.nanoTime();

      float[] logits = null;
      int position = 0;
      for (int token : tokens) {
        logits = backend.forward(token, position++);
      }
      long prefilled = System.nanoTime();

      StringBuilder text = new StringBuilder();
      for (int step = 0; step < maxTokens; step++) {
        int next = argmax(logits);
        String piece = backend.tokenizer().decode(new int[] {next});
        text.append(piece);
        if (text.indexOf("<|im_end|>") >= 0 || text.indexOf("<|endoftext|>") >= 0) {
          break;
        }
        logits = backend.forward(next, position++);
      }
      long done = System.nanoTime();

      double prefillSeconds = (prefilled - start) / 1e9;
      double decodeSeconds = (done - prefilled) / 1e9;
      System.out.printf(
          "[edgerock:%s] prompt=%d tokens prefill=%.1fs (%.1f tok/s) decode=%.1fs%n",
          id, tokens.length, prefillSeconds, tokens.length / prefillSeconds, decodeSeconds);
      System.out.printf("[edgerock:%s] answer tail: %s%n", id, tail(text.toString()));

      assertThat(text.toString())
          .as("%s: expected the publisher's recorded answer %s", id, expected)
          .contains(expected);
    } finally {
      if (previous == null) {
        System.clearProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      } else {
        System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }
  }

  private static String tail(String text) {
    String trimmed = text.replace("\n", "\\n");
    return trimmed.length() <= 200 ? trimmed : trimmed.substring(trimmed.length() - 200);
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }
}
