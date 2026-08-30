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
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-artifact compatibility and independent-reference tests for MiniLM's BERT encoder. */
@Tag("integration")
class BertMiniLmModelFixtureIntegrationTest {

  private static final String TEXT = "A commuter needs an accessible route to the airport.";
  private static final ModelFixtureRequirement MINILM =
      ModelFixtureRequirement.of("hf://second-state/All-MiniLM-L6-v2-Embedding-GGUF")
          .version("[2.0.0,3.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("embedding");

  private static ModelFixtureDescriptor fixture() {
    return ModelFixtureRegistry.fromClasspath().resolve(MINILM).orElseThrow();
  }

  @Test
  void wordPieceTokenizationMatchesThePinnedLlamaCppOracle() throws IOException {
    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(fixture().localPath().orElseThrow(), arena);
      var tokenizer = GgufTokenizer.fromMetadata(file.metadata());

      assertThat(tokenizer.encode(TEXT))
          .containsExactly(101, 1037, 14334, 3791, 2019, 7801, 2799, 2000, 1996, 3199, 1012, 102);
      assertThat(tokenizer.encode("Transit-friendly APIs handle naïve café riders."))
          .containsExactly(101, 6671, 1011, 5379, 17928, 2015, 5047, 15743, 7668, 8195, 1012, 102);
      assertThat(tokenizer.encode("unaffordable rerouting"))
          .containsExactly(101, 14477, 4246, 8551, 3085, 2128, 22494, 3436, 102);
    }
  }

  @Test
  void normalizedEmbeddingMatchesThePinnedLlamaCppOracle() throws IOException {
    float[] expected = referenceEmbedding();
    try (PureJavaBackend backend = PureJavaBackend.load(fixture().localPath().orElseThrow());
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend).normalize(true).build()) {
      float[] actual = embedding.embed(TEXT);

      assertThat(backend.metadata().modelFamily()).isEqualTo("bert");
      assertThat(backend.supportsSequenceEmbedding()).isTrue();
      assertThat(actual).hasSize(384);
      assertThat(cosine(actual, expected)).isGreaterThan(0.9995f);
    }
  }

  @Test
  void embeddingsAreRepeatableAndPreserveSemanticNeighborhoods() {
    try (PureJavaBackend backend = PureJavaBackend.load(fixture().localPath().orElseThrow());
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend).normalize(true).build()) {
      float[] anchor = embedding.embed(TEXT);
      float[] repeated = embedding.embed(TEXT);
      float[] related =
          embedding.embed("How can a wheelchair rider take public transit to the airport?");
      float[] unrelated =
          embedding.embed("The Java memory model defines happens-before relationships.");

      assertThat(repeated).containsExactly(anchor);
      assertThat(cosine(anchor, related)).isGreaterThan(cosine(anchor, unrelated));
    }
  }

  @Test
  void refusesSequencesBeyondTheTrainedContext() {
    String text = "transit ".repeat(600);
    try (PureJavaBackend backend = PureJavaBackend.load(fixture().localPath().orElseThrow());
        GgufEmbeddingBackend embedding = GgufEmbeddingBackend.builder(backend).build()) {
      assertThat(org.assertj.core.api.Assertions.catchThrowable(() -> embedding.embed(text)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("context length of 512");
    }
  }

  private static float[] referenceEmbedding() throws IOException {
    try (var input =
        BertMiniLmModelFixtureIntegrationTest.class.getResourceAsStream(
            "/oracles/all-minilm-l6-v2-q4-k-m.txt")) {
      if (input == null) {
        throw new IOException("missing MiniLM embedding oracle");
      }
      String[] values =
          new String(input.readAllBytes(), StandardCharsets.UTF_8).trim().split("\\s+");
      float[] result = new float[values.length];
      for (int index = 0; index < values.length; index++) {
        result[index] = Float.parseFloat(values[index]);
      }
      return result;
    }
  }

  private static float cosine(float[] left, float[] right) {
    assertThat(left).hasSameSizeAs(right);
    double dot = 0;
    double leftNorm = 0;
    double rightNorm = 0;
    for (int index = 0; index < left.length; index++) {
      dot += (double) left[index] * right[index];
      leftNorm += (double) left[index] * left[index];
      rightNorm += (double) right[index] * right[index];
    }
    return (float) (dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm)));
  }
}
