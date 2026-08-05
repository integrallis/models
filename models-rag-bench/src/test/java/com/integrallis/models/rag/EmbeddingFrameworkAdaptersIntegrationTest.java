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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.models.api.Pooling;
import com.integrallis.models.backend.purejava.GgufEmbeddingBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.langchain4j.ModelsEmbeddingModel;
import com.integrallis.models.spring.ai.ModelsSpringAiEmbeddingModel;
import dev.langchain4j.data.segment.TextSegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/**
 * Drives both framework embedding adapters with real Qwen3-Embedding weights.
 *
 * <p>No stand-ins: the point is to prove the adapters carry genuine semantics through to each
 * framework's own types, not merely that they move float arrays around. A stubbed backend can
 * satisfy every shape assertion while producing vectors that mean nothing.
 */
@Tag("integration")
class EmbeddingFrameworkAdaptersIntegrationTest {

  private static final int EMBEDDING_DIM = 1024;

  private static final String PASSWORD = "How do I reset my password?";
  private static final String LOGIN = "I forgot my login credentials and need to recover access.";
  private static final String PHYSICS = "Quantum chromodynamics describes the strong interaction.";

  private static Path fixturePath() {
    String configured = System.getProperty("models.fixtures.directory");
    Path directory =
        configured == null || configured.isBlank()
            ? Path.of(System.getProperty("user.home"), ".jvllm", "models")
            : Path.of(configured);
    return directory.resolve("Qwen3-Embedding-0.6B-Q8_0.gguf");
  }

  private static EmbeddingBackend openBackend() {
    Path model = fixturePath();
    assumeTrue(
        Files.exists(model),
        "Qwen3-Embedding fixture missing; run :backend-java:downloadQwen3Embedding06BQ80Model");
    return GgufEmbeddingBackend.builder(PureJavaBackend.load(model))
        .pooling(Pooling.LAST_TOKEN)
        .normalize(true)
        .build();
  }

  private static float cosine(float[] left, float[] right) {
    double dot = 0;
    for (int index = 0; index < left.length; index++) {
      dot += (double) left[index] * right[index];
    }
    // Both adapters normalize, so cosine reduces to the dot product.
    return (float) dot;
  }

  @Test
  void springAiAdapterCarriesSemanticsIntoEmbeddingResponses() {
    try (ModelsSpringAiEmbeddingModel model = new ModelsSpringAiEmbeddingModel(openBackend())) {
      assertThat(model.dimensions()).isEqualTo(EMBEDDING_DIM);

      EmbeddingResponse response =
          model.call(new EmbeddingRequest(List.of(PASSWORD, LOGIN, PHYSICS), null));

      assertThat(response.getResults()).hasSize(3);
      float[] password = response.getResults().get(0).getOutput();
      float[] login = response.getResults().get(1).getOutput();
      float[] physics = response.getResults().get(2).getOutput();

      assertThat(password).hasSize(EMBEDDING_DIM);
      assertThat(cosine(password, login)).isGreaterThan(cosine(password, physics));
      // Indices must survive the batch, or callers cannot match a vector to its input.
      assertThat(response.getResults().get(2).getIndex()).isEqualTo(2);
    }
  }

  @Test
  void springAiAdapterEmbedsDocumentsThroughTheSameModel() {
    try (ModelsSpringAiEmbeddingModel model = new ModelsSpringAiEmbeddingModel(openBackend())) {
      float[] fromDocument = model.embed(new Document(PASSWORD));
      float[] fromString = model.embed(PASSWORD);

      assertThat(fromDocument).containsExactly(fromString);
    }
  }

  @Test
  void langChain4jAdapterCarriesSemanticsIntoEmbeddings() {
    try (ModelsEmbeddingModel model = new ModelsEmbeddingModel(openBackend())) {
      assertThat(model.dimension()).isEqualTo(EMBEDDING_DIM);

      List<dev.langchain4j.data.embedding.Embedding> embeddings =
          model
              .embedAll(
                  List.of(
                      TextSegment.from(PASSWORD),
                      TextSegment.from(LOGIN),
                      TextSegment.from(PHYSICS)))
              .content();

      assertThat(embeddings).hasSize(3);
      float[] password = embeddings.get(0).vector();
      float[] login = embeddings.get(1).vector();
      float[] physics = embeddings.get(2).vector();

      assertThat(password).hasSize(EMBEDDING_DIM);
      assertThat(cosine(password, login)).isGreaterThan(cosine(password, physics));
    }
  }

  @Test
  void bothAdaptersAgreeOnTheSameText() {
    // One backend, two framework surfaces: an application must be able to move between them
    // without re-indexing, which requires byte-identical vectors.
    float[] springAi;
    float[] langChain4j;
    try (ModelsSpringAiEmbeddingModel model = new ModelsSpringAiEmbeddingModel(openBackend())) {
      springAi = model.embed(PASSWORD);
    }
    try (ModelsEmbeddingModel model = new ModelsEmbeddingModel(openBackend())) {
      langChain4j = model.embed(PASSWORD).content().vector();
    }

    assertThat(springAi).containsExactly(langChain4j);
  }
}
