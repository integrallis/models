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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Exact-artifact load and public generation smoke; oracle continuation is recorded separately. */
@Tag("integration")
@Tag("slow")
class Bonsai2ModelFixtureIntegrationTest {
  @Test
  void generatesFromTheSignedHadamardTernaryArtifact() {
    var requirement =
        ModelFixtureRequirement.of("hf://prism-ml/Ternary-Bonsai-2-27B-gguf")
            .version("[2.0.0,3.0.0)")
            .variant("ptq1_0")
            .backend("pure-java")
            .capability("chat");
    var fixture = ModelFixtureRegistry.fromClasspath().resolve(requirement).orElseThrow();
    try (var backend = PureJavaBackend.load(fixture.localPath().orElseThrow())) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen35");
      int[] tokens = backend.tokenizer().encode("The quick brown fox");
      assertThat(tokens).containsExactly(760, 3841, 13477, 37550);
      // PrismML-Eng/llama.cpp e8fd209c975c80aa7fb36a8423571ff3b1b7e955, exact PTQ1_0 bytes.
      int[] expected = {33075, 888, 279, 15217, 5388, 13, 271, 760};
      float[] logits = backend.prefill(tokens, 0);
      for (int step = 0; step < expected.length; step++) {
        assertThat(logits).hasSize(backend.metadata().vocabSize());
        int best = 0;
        for (int i = 0; i < logits.length; i++) {
          assertThat(Float.isFinite(logits[i])).isTrue();
          if (logits[i] > logits[best]) best = i;
        }
        assertThat(best).as("oracle continuation token %s", step).isEqualTo(expected[step]);
        if (step + 1 < expected.length) logits = backend.forward(best, tokens.length + step);
      }
      try (var session = backend.openSession()) {
        float[] fresh = backend.prefill(session, tokens, 0);
        int best = 0;
        for (int i = 1; i < fresh.length; i++) if (fresh[i] > fresh[best]) best = i;
        assertThat(best).isEqualTo(expected[0]);
      }
    }
  }
}
