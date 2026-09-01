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

/** Real-artifact tokenizer compatibility tests for Granite's multilingual BERT encoder. */
@Tag("integration")
class GraniteEmbeddingModelFixtureIntegrationTest {

  private static final ModelFixtureRequirement GRANITE =
      ModelFixtureRequirement.of("hf://bartowski/granite-embedding-107m-multilingual-GGUF")
          .version("[1.0.0,2.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("embedding");

  private static ModelFixtureDescriptor fixture() {
    return ModelFixtureRegistry.fromClasspath().resolve(GRANITE).orElseThrow();
  }

  @Test
  void unigramTokenizationMatchesThePinnedLlamaCppOracle() throws IOException {
    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(fixture().localPath().orElseThrow(), arena);
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());

      assertThat(file.metadata().getString("tokenizer.ggml.model")).contains("t5");
      assertThat(tokenizer.encode("retrieval augmented generation"))
          .containsExactly(0, 456, 97351, 1405, 9620, 1183, 71, 58093, 2);
      assertThat(tokenizer.encode("café naïve résumé"))
          .containsExactly(0, 26216, 24, 9392, 272, 233482, 2);
      assertThat(tokenizer.encode("東京の電車")).containsExactly(0, 6, 22888, 154, 130497, 2);
    }
  }

  @Test
  void exposesTheExpectedGraniteEmbeddingContract() {
    try (PureJavaBackend backend = PureJavaBackend.load(fixture().localPath().orElseThrow());
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend).normalize(true).build()) {
      float[] actual = embedding.embed("retrieval augmented generation");

      assertThat(backend.metadata().modelFamily()).isEqualTo("bert");
      assertThat(backend.supportsSequenceEmbedding()).isTrue();
      assertThat(actual).hasSize(384);
      assertThat(l2Norm(actual)).isBetween(0.9999, 1.0001);
    }
  }

  private static double l2Norm(float[] vector) {
    double squared = 0.0;
    for (float value : vector) {
      squared += (double) value * value;
    }
    return Math.sqrt(squared);
  }
}
