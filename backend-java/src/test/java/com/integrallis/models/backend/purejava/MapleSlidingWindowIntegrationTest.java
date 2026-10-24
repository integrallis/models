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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * maple's bounded attention and expert clamp, exercised past the declared window.
 *
 * <p>The prompt has to be longer than 512 tokens or this test cannot fail for the reason it exists.
 * Below the window, bounded and full attention are the same computation: a four-token load probe
 * reported finite logits and an argmax while the graph was running full attention on all 24 layers,
 * and that probe is what cleared maple into the catalog. So the token count is asserted against the
 * declared window rather than assumed from the prose length -- a tokenizer change that shortened
 * this prompt under 512 would otherwise leave a passing test that checks nothing.
 *
 * <p>Tagged slow on purpose. This is a 20B TQ1_0 artifact and the prefill is minutes, not seconds.
 */
@Tag("model-fixture")
@Tag("slow")
class MapleSlidingWindowIntegrationTest {

  private static final ModelFixtureRequirement MAPLE_PREVIEW_TQ1_0 =
      ModelFixtureRequirement.of("hf://deepgrove/maple-preview-GGUF")
          .version("[1.0.0,2.0.0)")
          .variant("tq1_0")
          .backend("pure-java")
          .capability("chat");

  /** maple's declared window, and its declared pattern: three local layers then one global. */
  private static final int DECLARED_WINDOW = 512;

  private static final int LAYERS = 24;
  private static final int EXPECTED_LOCAL_LAYERS = 18;

  @Test
  void boundsAttentionAndClampsExpertsOnAPromptLongerThanTheWindow() {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(MAPLE_PREVIEW_TQ1_0).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as("%s must be installed before running this test", descriptor.localPath().orElseThrow())
        .isTrue();

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor.localPath().orElseThrow())) {
      BackendDiagnostics diagnostics = backend.diagnostics();

      // Reported, not inferred. Before 2026-10-24 both of these were silently off and no output
      // inspection could tell: the model simply answered a little worse.
      assertThat(diagnostics.optimization("sliding-window-attention"))
          .get()
          .satisfies(
              decision -> {
                assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
                assertThat(decision.settings())
                    .containsEntry("declared-window", Integer.toString(DECLARED_WINDOW))
                    .containsEntry("local-layers", EXPECTED_LOCAL_LAYERS + "/" + LAYERS)
                    .containsEntry("pattern-source", "declared-per-layer");
              });
      assertThat(diagnostics.optimization("expert-swiglu-clamp"))
          .get()
          .satisfies(
              decision -> {
                assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
                assertThat(decision.settings())
                    .containsEntry("clamped-layers", LAYERS + "/" + LAYERS);
              });

      int[] tokens = backend.tokenizer().encode(longPrompt());
      assertThat(tokens.length)
          .as(
              "the prompt must exceed the %d-token window, or bounded and full attention are the"
                  + " same computation and this test proves nothing",
              DECLARED_WINDOW)
          .isGreaterThan(DECLARED_WINDOW);

      float[] logits = backend.prefill(tokens, 0);
      assertThat(logits).hasSize(backend.metadata().vocabSize());
      for (int index = 0; index < logits.length; index++) {
        assertThat(Float.isFinite(logits[index]))
            .as("logit %d must be finite past the sliding window", index)
            .isTrue();
      }

      // A flat distribution is the shape of a mask that admitted nothing or everything. Asserting
      // a POSITIVE top logit would be wrong -- nothing guarantees the sign -- so this asserts
      // non-degeneracy instead: a real spread, and a top that stands above the mean.
      int best = 0;
      double total = 0.0;
      for (int index = 0; index < logits.length; index++) {
        total += logits[index];
        if (logits[index] > logits[best]) {
          best = index;
        }
      }
      float mean = (float) (total / logits.length);
      assertThat(logits[best])
          .as("the next-token distribution must not be flat past the window")
          .isGreaterThan(mean);
      assertThat(logits[best]).isNotEqualTo(logits[(best + 1) % logits.length]);
    }
  }

  /**
   * Deterministic prose, lexically varied so the tokenizer cannot collapse it into a short repeated
   * run, and long enough to clear the window with room to spare.
   */
  private static String longPrompt() {
    String[] words = {
      "harbor", "lantern", "gravel", "meridian", "copper", "thistle", "quarry", "vellum",
      "anvil", "cistern", "furrow", "palisade", "mortar", "seldom", "tundra", "basalt",
      "clover", "rafter", "pebble", "sextant", "willow", "cinder", "marrow", "trellis"
    };
    StringBuilder prompt = new StringBuilder();
    for (int item = 0; item < 60; item++) {
      prompt
          .append("Item ")
          .append(item + 1)
          .append(": the ")
          .append(words[item % words.length])
          .append(" beside the ")
          .append(words[(item * 7 + 3) % words.length])
          .append(" was counted twice before the ")
          .append(words[(item * 11 + 5) % words.length])
          .append(" settled. ");
    }
    return prompt.toString().trim();
  }
}
