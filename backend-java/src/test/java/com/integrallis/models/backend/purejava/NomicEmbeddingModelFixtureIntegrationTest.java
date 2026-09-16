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

import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-artifact contract checks for Nomic's RoPE and SwiGLU BERT encoder. */
@Tag("integration")
class NomicEmbeddingModelFixtureIntegrationTest {

  private static final ModelFixtureRequirement NOMIC =
      ModelFixtureRequirement.of("hf://second-state/Nomic-embed-text-v1.5-Embedding-GGUF")
          .version("[1.5.0,2.0.0)")
          .variant("f16")
          .backend("pure-java")
          .capability("embedding");

  private static ModelFixtureDescriptor fixture() {
    return ModelFixtureRegistry.fromClasspath().resolve(NOMIC).orElseThrow();
  }

  @Test
  void wordPieceTokenizationMatchesThePinnedLlamaCppOracle() throws IOException {
    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(fixture().localPath().orElseThrow(), arena);
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());

      assertThat(file.metadata().getString("general.architecture")).contains("nomic-bert");
      assertThat(tokenizer.encode("A commuter needs an accessible route to the airport."))
          .containsExactly(101, 1037, 14334, 3791, 2019, 7801, 2799, 2000, 1996, 3199, 1012, 102);
    }
  }

  @Test
  void exposesTheNomicEmbeddingContract() {
    try (PureJavaBackend backend = PureJavaBackend.load(fixture().localPath().orElseThrow());
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend).normalize(true).build()) {
      float[] actual = embedding.embed("A commuter needs an accessible route to the airport.");

      assertThat(backend.metadata().modelFamily()).isEqualTo("nomic-bert");
      assertThat(backend.supportsSequenceEmbedding()).isTrue();
      assertThat(actual).hasSize(768);
      assertThat(l2Norm(actual)).isBetween(0.9999, 1.0001);
    }
  }

  private static double l2Norm(float[] vector) {
    double squared = 0;
    for (float value : vector) {
      squared += (double) value * value;
    }
    return Math.sqrt(squared);
  }
}
