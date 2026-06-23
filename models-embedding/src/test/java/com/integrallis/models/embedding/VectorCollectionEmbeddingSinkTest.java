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
package com.integrallis.models.embedding;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * Pins the contract of {@link VectorCollectionEmbeddingSink} (ROADMAP II.12 F4): the bridge wires
 * an {@link EmbeddingBackend} into a {@link VectorCollection} with no extra copy beyond
 * vectors-db's documented staging-buffer clone, and dimension mismatches are caught at
 * construction.
 */
class VectorCollectionEmbeddingSinkTest {

  private static final int DIM = 4;

  private static VectorCollection collection(int dim) {
    return VectorCollection.builder()
        .dimension(dim)
        .metric(SimilarityFunction.COSINE)
        .indexType(IndexType.FLAT)
        .build();
  }

  @Test
  void putEmbedsAndStoresWithMatchingDimension() {
    VectorCollection c = collection(DIM);
    AtomicInteger calls = new AtomicInteger();
    EmbeddingBackend backend =
        new EmbeddingBackend() {
          @Override
          public int dimension() {
            return DIM;
          }

          @Override
          public float[] embed(String text) {
            calls.incrementAndGet();
            return new float[] {1.0f + text.length(), 0.5f, -0.25f, 0.1f};
          }
        };
    try (c) {
      VectorCollectionEmbeddingSink sink = new VectorCollectionEmbeddingSink(backend, c);
      sink.put("a", "hello");
      sink.put("b", "world");
      c.commit();

      assertThat(calls.get()).as("each put must invoke the backend exactly once").isEqualTo(2);
      assertThat(c.size()).isEqualTo(2);
      assertThat(c.contains("a")).isTrue();
      assertThat(c.contains("b")).isTrue();
    }
  }

  @Test
  void putAllUsesTheBatchPathSoTheBackendOnlySeesOneCall() {
    VectorCollection c = collection(DIM);
    AtomicInteger singleCalls = new AtomicInteger();
    AtomicInteger batchCalls = new AtomicInteger();
    EmbeddingBackend backend =
        new EmbeddingBackend() {
          @Override
          public int dimension() {
            return DIM;
          }

          @Override
          public float[] embed(String text) {
            singleCalls.incrementAndGet();
            return new float[] {text.length(), 0f, 0f, 0f};
          }

          @Override
          public float[][] embedAll(List<String> texts) {
            batchCalls.incrementAndGet();
            float[][] out = new float[texts.size()][];
            for (int i = 0; i < texts.size(); i++) {
              out[i] = new float[] {texts.get(i).length(), 0f, 0f, 0f};
            }
            return out;
          }
        };
    try (c) {
      VectorCollectionEmbeddingSink sink = new VectorCollectionEmbeddingSink(backend, c);
      sink.putAll(List.of("id-1", "id-2", "id-3"), List.of("alpha", "beta", "gamma"));
      c.commit();

      assertThat(batchCalls.get())
          .as("putAll must reach the batch path, not the single-embed loop")
          .isEqualTo(1);
      assertThat(singleCalls.get())
          .as("the single-embed path must not be invoked when embedAll is overridden")
          .isZero();
      assertThat(c.size()).isEqualTo(3);
    }
  }

  @Test
  void searchRoundTripsThroughTheSink() {
    VectorCollection c = collection(DIM);
    EmbeddingBackend backend =
        new EmbeddingBackend() {
          @Override
          public int dimension() {
            return DIM;
          }

          @Override
          public float[] embed(String text) {
            // One-hot per character — different texts land in different positions in the unit
            // cube so cosine ranks them deterministically.
            float[] v = new float[DIM];
            v[Math.floorMod(text.hashCode(), DIM)] = 1.0f;
            return v;
          }
        };
    try (c) {
      VectorCollectionEmbeddingSink sink = new VectorCollectionEmbeddingSink(backend, c);
      sink.putAll(List.of("apple", "banana", "cherry"), List.of("apple", "banana", "cherry"));
      c.commit();

      // Search using the same backend to embed the query — closes the loop.
      float[] q = backend.embed("apple");
      SearchResult hits = c.search(SearchRequest.builder(q, 1).build());
      assertThat(hits.hits()).hasSize(1);
      assertThat(hits.hits().get(0).id())
          .as("an exact-match query must return its own document first")
          .isEqualTo("apple");
    }
  }

  @Test
  void constructorRejectsDimensionMismatch() {
    VectorCollection c = collection(DIM);
    EmbeddingBackend mismatched =
        new EmbeddingBackend() {
          @Override
          public int dimension() {
            return DIM * 2;
          }

          @Override
          public float[] embed(String text) {
            return new float[DIM * 2];
          }
        };
    try (c) {
      assertThatThrownBy(() -> new VectorCollectionEmbeddingSink(mismatched, c))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("dimension");
    }
  }

  @Test
  void putAllRejectsMismatchedListSizes() {
    VectorCollection c = collection(DIM);
    EmbeddingBackend backend = constantBackend(DIM, 1.0f);
    try (c) {
      VectorCollectionEmbeddingSink sink = new VectorCollectionEmbeddingSink(backend, c);
      assertThatThrownBy(() -> sink.putAll(List.of("a", "b"), List.of("only-one-text")))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("size");
    }
  }

  @Test
  void putAllRejectsBackendThatReturnsWrongRowCount() {
    VectorCollection c = collection(DIM);
    EmbeddingBackend buggy =
        new EmbeddingBackend() {
          @Override
          public int dimension() {
            return DIM;
          }

          @Override
          public float[] embed(String text) {
            return new float[DIM];
          }

          @Override
          public float[][] embedAll(List<String> texts) {
            return new float[texts.size() + 1][DIM]; // off-by-one to simulate a buggy backend
          }
        };
    try (c) {
      VectorCollectionEmbeddingSink sink = new VectorCollectionEmbeddingSink(buggy, c);
      assertThatThrownBy(() -> sink.putAll(List.of("a", "b"), List.of("x", "y")))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("embedAll contract");
    }
  }

  private static EmbeddingBackend constantBackend(int dim, float value) {
    return new EmbeddingBackend() {
      @Override
      public int dimension() {
        return dim;
      }

      @Override
      public float[] embed(String text) {
        float[] v = new float[dim];
        java.util.Arrays.fill(v, value);
        return v;
      }
    };
  }
}
