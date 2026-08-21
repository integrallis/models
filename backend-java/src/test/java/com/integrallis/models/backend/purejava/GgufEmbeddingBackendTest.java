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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.api.Pooling;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Contract tests for the GGUF embedding backend against a synthetic nano model. */
@Tag("unit")
class GgufEmbeddingBackendTest {

  // The nano vocabulary is t0..t31, so inputs are written in its own tokens.
  private static final String TEXT = "t5 t7 t11";
  private static final String OTHER_TEXT = "t13 t17";
  private static final float FLOAT_REDUCTION_TOLERANCE = 1.0e-7f;

  private static Path modelPath;

  /**
   * Builds the synthetic model on first use.
   *
   * <p>Lazy rather than {@code @BeforeAll}: the nested classes here are static, so JUnit discovers
   * them as standalone test classes and never runs the enclosing class lifecycle.
   */
  private static synchronized Path modelPath() {
    if (modelPath == null) {
      try {
        Path directory = Files.createTempDirectory("gguf-embedding-test");
        directory.toFile().deleteOnExit();
        modelPath = PureJavaBackendTest.buildNanoModelFile(directory, new Random(42));
        modelPath.toFile().deleteOnExit();
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
    }
    return modelPath;
  }

  private static GgufEmbeddingBackend embedding(Pooling pooling, boolean normalize) {
    return GgufEmbeddingBackend.builder(PureJavaBackend.load(modelPath()))
        .pooling(pooling)
        .normalize(normalize)
        .build();
  }

  private static double magnitude(float[] vector) {
    double sumOfSquares = 0;
    for (float value : vector) {
      sumOfSquares += (double) value * value;
    }
    return Math.sqrt(sumOfSquares);
  }

  @Nested
  static class Shape {

    @Test
    void reportsTheModelEmbeddingWidth() {
      try (GgufEmbeddingBackend embed = embedding(Pooling.LAST_TOKEN, true)) {
        assertThat(embed.dimension()).isPositive();
        assertThat(embed.embed(TEXT)).hasSize(embed.dimension());
      }
    }

    @Test
    void embedAllPreservesRowOrderWithinFloatTolerance() {
      try (GgufEmbeddingBackend embed = embedding(Pooling.LAST_TOKEN, true)) {
        float[][] rows = embed.embedAll(List.of(TEXT, OTHER_TEXT, TEXT));

        assertThat(rows).hasDimensions(3, embed.dimension());
        // A row may cross the JVM's interpreted/JIT boundary, which can change the final bit of a
        // floating-point reduction without changing the embedding.
        assertThat(rows[0]).containsExactly(rows[2], within(FLOAT_REDUCTION_TOLERANCE));
        assertThat(rows[0]).isNotEqualTo(rows[1]);
      }
    }
  }

  @Nested
  static class Determinism {

    @Test
    void repeatedCallsAgreeExactly() {
      try (GgufEmbeddingBackend embed = embedding(Pooling.LAST_TOKEN, true)) {
        assertThat(embed.embed(TEXT)).containsExactly(embed.embed(TEXT));
      }
    }

    @Test
    void oneTextDoesNotLeakIntoTheNext() {
      // Each text is an independent sequence. Without resetting the KV cache the second text
      // would attend to the first, making results depend on call order.
      try (GgufEmbeddingBackend isolated = embedding(Pooling.LAST_TOKEN, true);
          GgufEmbeddingBackend reused = embedding(Pooling.LAST_TOKEN, true)) {
        float[] alone = isolated.embed(OTHER_TEXT);

        reused.embed(TEXT);
        float[] afterAnother = reused.embed(OTHER_TEXT);

        assertThat(afterAnother).containsExactly(alone);
      }
    }
  }

  @Nested
  static class Normalization {

    @Test
    void normalizedVectorsAreUnitLength() {
      // Unit length is what lets downstream cosine similarity collapse to a dot product.
      try (GgufEmbeddingBackend embed = embedding(Pooling.LAST_TOKEN, true)) {
        assertThat(magnitude(embed.embed(TEXT))).isCloseTo(1.0, Offset.offset(1.0e-5));
      }
    }

    @Test
    void normalizationCanBeDisabled() {
      try (GgufEmbeddingBackend raw = embedding(Pooling.LAST_TOKEN, false)) {
        assertThat(magnitude(raw.embed(TEXT))).isNotCloseTo(1.0, Offset.offset(1.0e-3));
      }
    }
  }

  @Nested
  static class PoolingModes {

    @Test
    void meanAndLastTokenDisagreeOnMultiTokenInput() {
      // Two different reductions of the same states. Choosing the wrong one degrades retrieval
      // silently rather than failing, which is why pooling has to travel with the model.
      try (GgufEmbeddingBackend last = embedding(Pooling.LAST_TOKEN, true);
          GgufEmbeddingBackend mean = embedding(Pooling.MEAN, true)) {
        assertThat(last.embed(TEXT)).isNotEqualTo(mean.embed(TEXT));
      }
    }

    @Test
    void meanPoolingIsDeterministic() {
      try (GgufEmbeddingBackend embed = embedding(Pooling.MEAN, true)) {
        assertThat(embed.embed(TEXT)).containsExactly(embed.embed(TEXT));
      }
    }
  }

  @Nested
  static class Contract {

    @Test
    void returnsCallerOwnedArrays() {
      // The SPI documents results as caller-owned; mutating one must not affect later calls.
      try (GgufEmbeddingBackend embed = embedding(Pooling.LAST_TOKEN, true)) {
        float[] first = embed.embed(TEXT);
        float original = first[0];
        first[0] = Float.NaN;

        assertThat(embed.embed(TEXT)[0]).isEqualTo(original);
      }
    }

    @Test
    void rejectsNullInput() {
      try (GgufEmbeddingBackend embed = embedding(Pooling.LAST_TOKEN, true)) {
        assertThatThrownBy(() -> embed.embed(null)).isInstanceOf(NullPointerException.class);
        assertThatThrownBy(() -> embed.embedAll(null)).isInstanceOf(NullPointerException.class);
      }
    }

    @Test
    void embedsEmptyTextWithoutFailing() {
      // Ingest runs batch whatever a corpus contains; a blank row must not abort the run.
      try (GgufEmbeddingBackend embed = embedding(Pooling.LAST_TOKEN, true)) {
        assertThat(embed.embed("")).hasSize(embed.dimension());
      }
    }

    @Test
    void closingReleasesTheUnderlyingBackend() {
      GgufEmbeddingBackend embed = embedding(Pooling.LAST_TOKEN, true);
      embed.close();

      assertThatThrownBy(() -> embed.embed(TEXT)).isInstanceOf(IllegalStateException.class);
    }
  }
}
