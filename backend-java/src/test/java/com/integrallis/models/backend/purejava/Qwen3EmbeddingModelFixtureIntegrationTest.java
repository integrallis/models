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

import com.integrallis.models.api.Pooling;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Exercises the embedding path against the real Qwen3-Embedding-0.6B weights.
 *
 * <p>The synthetic nano-model tests prove the plumbing: shapes, isolation, normalization. They
 * cannot prove the vectors mean anything, because random weights carry no semantics. These tests
 * assert the property that actually matters — that related text lands closer together than
 * unrelated text — which only a trained model can satisfy.
 */
@Tag("integration")
class Qwen3EmbeddingModelFixtureIntegrationTest {

  private static final ModelFixtureRequirement QWEN3_EMBEDDING_0_6B_Q8_0 =
      ModelFixtureRequirement.of("hf://Qwen/Qwen3-Embedding-0.6B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("embedding");

  /** Published width of Qwen3-Embedding-0.6B. */
  private static final int EMBEDDING_DIM = 1024;

  private static final String CAT = "The cat sat on the mat.";
  private static final String FELINE = "A feline rested upon the rug.";
  private static final String PHYSICS = "Quantum chromodynamics describes the strong interaction.";
  private static final String PASSWORD = "How do I reset my password?";
  private static final String LOGIN = "I forgot my login credentials and need to recover access.";

  private static PureJavaBackend loadModel() {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(QWEN3_EMBEDDING_0_6B_Q8_0).orElseThrow();
    return PureJavaBackend.load(descriptor.localPath().orElseThrow());
  }

  private static GgufEmbeddingBackend embedding(Pooling pooling, boolean normalize) {
    return GgufEmbeddingBackend.builder(loadModel()).pooling(pooling).normalize(normalize).build();
  }

  private static float cosine(float[] left, float[] right) {
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

  @Test
  void loadsAsAQwen3ArchitectureEmbeddingModel() {
    try (PureJavaBackend backend = loadModel()) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen3");
      assertThat(backend.metadata().embeddingDim()).isEqualTo(EMBEDDING_DIM);
      assertThat(backend.supportsHiddenState()).isTrue();
    }
  }

  @Test
  void producesVectorsOfThePublishedWidth() {
    try (GgufEmbeddingBackend embedding = embedding(Pooling.LAST_TOKEN, true)) {
      assertThat(embedding.dimension()).isEqualTo(EMBEDDING_DIM);
      assertThat(embedding.embed(CAT)).hasSize(EMBEDDING_DIM);
    }
  }

  @Test
  void placesRelatedTextCloserThanUnrelatedText() {
    // The property everything downstream depends on. Retrieval, semantic caching and semantic
    // routing are all just thresholds over this ordering.
    try (GgufEmbeddingBackend embedding = embedding(Pooling.LAST_TOKEN, true)) {
      float[][] vectors = embedding.embedAll(List.of(CAT, FELINE, PHYSICS));

      float related = cosine(vectors[0], vectors[1]);
      float unrelated = cosine(vectors[0], vectors[2]);

      assertThat(related).isGreaterThan(unrelated);
      // A real margin, not a coin flip; observed ~0.51 on this model.
      assertThat(related - unrelated).isGreaterThan(0.25f);
    }
  }

  @Test
  void separatesParaphrasedSupportQueriesFromUnrelatedOnes() {
    // A support-desk shape: the paraphrase pair is what a semantic cache must treat as a hit,
    // and the unrelated pair is what it must not.
    try (GgufEmbeddingBackend embedding = embedding(Pooling.LAST_TOKEN, true)) {
      float[][] vectors = embedding.embedAll(List.of(PASSWORD, LOGIN, PHYSICS));

      float paraphrase = cosine(vectors[0], vectors[1]);
      float unrelated = cosine(vectors[0], vectors[2]);

      assertThat(paraphrase).isGreaterThan(unrelated);
      assertThat(paraphrase - unrelated).isGreaterThan(0.20f);
    }
  }

  @Test
  void meanPoolingAlsoOrdersSemantically() {
    // Both pooling modes must be usable; they differ in scale, not in ordering.
    try (GgufEmbeddingBackend embedding = embedding(Pooling.MEAN, true)) {
      float[][] vectors = embedding.embedAll(List.of(CAT, FELINE, PHYSICS));

      assertThat(cosine(vectors[0], vectors[1])).isGreaterThan(cosine(vectors[0], vectors[2]));
    }
  }

  @Test
  void identicalTextEmbedsToACosineOfOne() {
    try (GgufEmbeddingBackend embedding = embedding(Pooling.LAST_TOKEN, true)) {
      float[] first = embedding.embed(CAT);
      float[] second = embedding.embed(CAT);

      assertThat(first).containsExactly(second);
      assertThat(cosine(first, second))
          .isCloseTo(1.0f, org.assertj.core.data.Offset.offset(1.0e-5f));
    }
  }

  @Test
  void normalizedVectorsAreUnitLength() {
    try (GgufEmbeddingBackend embedding = embedding(Pooling.LAST_TOKEN, true)) {
      float[] vector = embedding.embed(CAT);

      double sumOfSquares = 0;
      for (float value : vector) {
        sumOfSquares += (double) value * value;
      }
      assertThat(Math.sqrt(sumOfSquares))
          .isCloseTo(1.0, org.assertj.core.data.Offset.offset(1.0e-5));
    }
  }

  @Test
  void embeddingOneTextDoesNotDependOnWhatCameBefore() {
    // Real proof of sequence isolation: with a trained model a leaked KV cache would visibly
    // move the vector, unlike the synthetic case where any change is hard to interpret.
    try (GgufEmbeddingBackend isolated = embedding(Pooling.LAST_TOKEN, true);
        GgufEmbeddingBackend reused = embedding(Pooling.LAST_TOKEN, true)) {
      float[] alone = isolated.embed(PHYSICS);

      reused.embed(CAT);
      reused.embed(PASSWORD);
      float[] afterOthers = reused.embed(PHYSICS);

      assertThat(afterOthers).containsExactly(alone);
    }
  }

  @Test
  void unnormalizedVectorsPreserveTheSameOrdering() {
    // Normalization changes magnitude, not direction, so the ranking must survive it.
    try (GgufEmbeddingBackend raw = embedding(Pooling.LAST_TOKEN, false)) {
      float[][] vectors = raw.embedAll(List.of(CAT, FELINE, PHYSICS));

      assertThat(cosine(vectors[0], vectors[1])).isGreaterThan(cosine(vectors[0], vectors[2]));
    }
  }
}
