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

import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Exact checkpoint, independent tokenizer and greedy-continuation reference. */
@Tag("slow")
class FlintModelFixtureIntegrationTest {
  @Test
  void matchesPinnedReferenceOnCompletionAndChat() throws Exception {
    var requirement =
        ModelFixtureRequirement.of("hf://ggml-org/Flint-GGUF")
            .version("1.0.0")
            .variant("q4_k_m")
            .backend("pure-java")
            .capability("chat");
    var fixture = ModelFixtureRegistry.fromClasspath().resolve(requirement).orElseThrow();
    Properties reference = new Properties();
    try (var input =
        new InputStreamReader(
            getClass().getResourceAsStream("/flint/reference.properties"),
            StandardCharsets.UTF_8)) {
      reference.load(input);
    }
    try (var backend = PureJavaBackend.load(fixture.localPath().orElseThrow())) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("flint");
      assertThat(backend.metadata().numLayers()).isEqualTo(49);
      assertThat(ChatTemplate.FLINT.endOfTurnStopsGeneration(backend.tokenizer())).isTrue();
      assertThat(backend.tokenizer().isEndOfGeneration(150003)).isTrue();
      assertThat(backend.tokenizer().isEndOfGeneration(150005)).isTrue();
      assertThat(backend.tokenizer().isEndOfGeneration(150018)).isTrue();
      for (String kind : new String[] {"completion", "chat"}) {
        backend.reset();
        String text = reference.getProperty(kind + ".prompt");
        int[] tokens =
            kind.equals("chat")
                ? backend
                    .tokenizer()
                    .encode(ChatTemplate.FLINT.render(List.of(ChatMessage.user(text))))
                : backend.tokenizer().encode(text);
        assertThat(tokens)
            .as(kind + " tokenizer")
            .containsExactly(ints(reference, kind + ".tokens"));
        int[] expected = ints(reference, kind + ".continuation");
        long started = System.nanoTime();
        float[] logits = backend.prefill(tokens, 0);
        long prefill = System.nanoTime();
        for (int step = 0; step < expected.length; step++) {
          int best = 0;
          for (int i = 0; i < logits.length; i++) {
            if (!Float.isFinite(logits[i])) throw new AssertionError("Nonfinite Flint logit " + i);
            if (logits[i] > logits[best]) best = i;
          }
          assertThat(best).as(kind + " reference token " + step).isEqualTo(expected[step]);
          if (step + 1 < expected.length) logits = backend.forward(best, tokens.length + step);
        }
        System.out.printf(
            "Flint %s: prefill %.3f s; %d subsequent tokens %.3f s%n",
            kind,
            (prefill - started) / 1e9,
            expected.length - 1,
            (System.nanoTime() - prefill) / 1e9);
      }
    }
  }

  private static int[] ints(Properties values, String key) {
    return Arrays.stream(values.getProperty(key).split(","))
        .map(String::trim)
        .mapToInt(Integer::parseInt)
        .toArray();
  }
}
