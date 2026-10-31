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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * EdgeRock's long-memory cases, on the route the catalog qualifies.
 *
 * <p>The cases and their expected answers are the publisher's own, extracted from {@code
 * receipts/GGUF-Q4_K_M.json}. Their recorded outputs are CUDA runs, so their timings are theirs;
 * only the expected answers are used. Preferring the creator's own cases and template over anything
 * reconstructed here is deliberate: they have already settled which template, which stop tokens and
 * what a correct answer looks like.
 *
 * <p>These are the cases worth running. At 9664 and 10462 input tokens against a 1024-token sliding
 * window, the identifier being asked for sits thousands of tokens outside anything the attention
 * layers can see. The short arithmetic case passes with the memory deleted; these should not.
 *
 * <p>One limit on what a pass proves, stated because it is easy to overclaim: this does not isolate
 * the anchors. EdgeRock's 24 recurrent layers have unbounded context, so a recalled identifier may
 * have travelled through the Gated DeltaNet state rather than through memory. Isolating them needs
 * an ablation that turns the anchors off, which the all-or-nothing construction guard forbids by
 * design.
 */
@Tag("slow")
@EnabledIfSystemProperty(named = EdgeRockNativePublisherCaseTest.FIXTURE_PROPERTY, matches = ".+")
@EnabledIfSystemProperty(named = EdgeRockNativePublisherCaseTest.CASES_PROPERTY, matches = ".+")
class EdgeRockNativePublisherCaseTest {

  static final String FIXTURE_PROPERTY = "models.fixtures.edgeRock4bGguf";
  static final String CASES_PROPERTY = "models.fixtures.edgeRockCases";

  /** From Modelfile.Q4_K_M: ChatML, with the assistant turn opened into a thinking block. */
  private static String render(String user) {
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
  }

  @Test
  void recallsAnIdentifierUpdatedFarOutsideTheAttentionWindow() throws IOException {
    assertCase("memory_update_8k", 700);
  }

  @Test
  void recallsTheRightOneOfTwoCompetingKeys() throws IOException {
    assertCase("memory_two_keys_8k", 700);
  }

  private void assertCase(String id, int maxTokens) throws IOException {
    Path cases = Path.of(System.getProperty(CASES_PROPERTY));
    String prompt = Files.readString(cases.resolve(id + ".prompt"));
    String expected = Files.readString(cases.resolve(id + ".expected")).trim();
    Path model = Path.of(System.getProperty(FIXTURE_PROPERTY));
    Path library = Path.of(System.getProperty("models.native.kernels.library"));

    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    // The publisher runs these at --ctx-size 16384; the longer prompt is 10462 input tokens.
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "16384");
    try (PureJavaBackend backend =
        PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.open(library, 8))) {
      int[] tokens = backend.tokenizer().encode(render(prompt));

      long start = System.nanoTime();
      float[] logits = backend.prefill(tokens, 0);
      long prefilled = System.nanoTime();

      StringBuilder text = new StringBuilder();
      int position = tokens.length;
      int generated = 0;
      for (int step = 0; step < maxTokens; step++) {
        int next = argmax(logits);
        // Stop on the token, not only on its spelling. Scanning decoded text alone let a run
        // continue past the answer and emit "<|im_start|>assistant\n<answer>...</answer>" over
        // and over, which reads like a model looping when it is the harness failing to stop.
        if (backend.tokenizer().isEndOfGeneration(next)) {
          break;
        }
        text.append(backend.tokenizer().decode(new int[] {next}));
        generated++;
        if (text.indexOf("<|im_end|>") >= 0
            || text.indexOf("<|endoftext|>") >= 0
            || text.indexOf("<|im_start|>") >= 0) {
          break;
        }
        logits = backend.forward(next, position++);
      }
      long done = System.nanoTime();

      double prefillSeconds = (prefilled - start) / 1e9;
      double decodeSeconds = (done - prefilled) / 1e9;
      System.out.printf(
          "[edgerock:native:%s] prompt=%d prefill=%.1fs (%.1f tok/s) "
              + "decode=%d tokens %.1fs (%.2f tok/s)%n",
          id,
          tokens.length,
          prefillSeconds,
          tokens.length / prefillSeconds,
          generated,
          decodeSeconds,
          generated / decodeSeconds);
      System.out.printf("[edgerock:native:%s] tail: %s%n", id, tail(text.toString()));

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
    String flat = text.replace("\n", "\\n");
    return flat.length() <= 240 ? flat : flat.substring(flat.length() - 240);
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
}
