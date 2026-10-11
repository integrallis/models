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
import static org.assertj.core.data.Offset.offset;

import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Public embedding path over the exact text GGUF; this does not qualify the multimodal projector.
 */
@Tag("integration")
class EmbeddingGemma2ModelFixtureIntegrationTest {
  @Test
  void embedsWithThePublishedWidthAndIsolatesSequences() {
    var requirement =
        ModelFixtureRequirement.of("hf://ggml-org/embeddinggemma-2-GGUF")
            .version("[2.0.0,3.0.0)")
            .variant("q8_0")
            .backend("pure-java")
            .capability("embedding");
    var fixture = ModelFixtureRegistry.fromClasspath().resolve(requirement).orElseThrow();
    try (var backend = PureJavaBackend.load(fixture.localPath().orElseThrow());
        var embedding = GgufEmbeddingBackend.builder(backend).build()) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("gemma-embedding2");
      assertThat(embedding.dimension()).isEqualTo(768);
      assertThat(backend.tokenizer().encode("The cat sat on the mat."))
          .containsExactly(2, 818, 5866, 2838, 580, 506, 1756, 236761, 1);
      float[] first = embedding.embed("The cat sat on the mat.");
      assertThat(first).hasSize(768);
      double norm = 0;
      for (float value : first) {
        assertThat(Float.isFinite(value)).isTrue();
        norm += (double) value * value;
      }
      assertThat(Math.sqrt(norm)).isCloseTo(1.0, offset(1e-3));
      embedding.embed("Quantum chromodynamics describes the strong interaction.");
      assertThat(embedding.embed("The cat sat on the mat.")).containsExactly(first);
    }
  }
}
