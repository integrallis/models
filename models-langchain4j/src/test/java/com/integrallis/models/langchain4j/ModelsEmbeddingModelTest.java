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
package com.integrallis.models.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.EmbeddingBackend;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelsEmbeddingModelTest {

  private static final int DIM = 4;

  /** Deterministic stand-in: each text maps to a vector derived from its hash. */
  private static final class ScriptedBackend implements EmbeddingBackend {
    private final List<String> seen = new ArrayList<>();
    private boolean closed;

    @Override
    public int dimension() {
      return DIM;
    }

    @Override
    public float[] embed(String text) {
      seen.add(text);
      float[] vector = new float[DIM];
      for (int index = 0; index < DIM; index++) {
        vector[index] = (text.hashCode() % (index + 7)) / 10.0f;
      }
      return vector;
    }

    @Override
    public void close() {
      closed = true;
    }
  }

  @Nested
  static class Contract {

    @Test
    void reportsTheBackendDimension() {
      assertThat(new ModelsEmbeddingModel(new ScriptedBackend()).dimension()).isEqualTo(DIM);
    }

    @Test
    void embedsASingleString() {
      ScriptedBackend backend = new ScriptedBackend();

      Embedding embedding = new ModelsEmbeddingModel(backend).embed("hello").content();

      assertThat(embedding.vector()).hasSize(DIM);
      assertThat(backend.seen).containsExactly("hello");
    }

    @Test
    void embedsATextSegmentUsingItsText() {
      ScriptedBackend backend = new ScriptedBackend();

      new ModelsEmbeddingModel(backend).embed(TextSegment.from("the text"));

      assertThat(backend.seen).containsExactly("the text");
    }

    @Test
    void serializesConcurrentCallsToANonThreadSafeBackend() throws Exception {
      var backend = new RejectConcurrentBackend();
      var embedding = new ModelsEmbeddingModel(backend);

      try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
        try {
          var first = executor.submit(() -> embedding.embed("first"));
          assertThat(backend.firstEntered.await(1, TimeUnit.SECONDS)).isTrue();
          var second = executor.submit(() -> embedding.embed("second"));

          assertThatThrownBy(() -> second.get(200, TimeUnit.MILLISECONDS))
              .isInstanceOf(TimeoutException.class);
          backend.releaseFirst.countDown();

          assertThat(first.get().content().vector()).hasSize(DIM);
          assertThat(second.get().content().vector()).hasSize(DIM);
          assertThat(backend.concurrentEntry).isFalse();
        } finally {
          backend.releaseFirst.countDown();
        }
      }
    }
  }

  @Nested
  static class Batches {

    @Test
    void embedAllPreservesSegmentOrder() {
      ScriptedBackend backend = new ScriptedBackend();
      List<TextSegment> segments =
          List.of(TextSegment.from("alpha"), TextSegment.from("beta"), TextSegment.from("alpha"));

      List<Embedding> embeddings = new ModelsEmbeddingModel(backend).embedAll(segments).content();

      assertThat(embeddings).hasSize(3);
      assertThat(backend.seen).containsExactly("alpha", "beta", "alpha");
      // Identical text embeds identically regardless of batch position.
      assertThat(embeddings.get(0).vector()).containsExactly(embeddings.get(2).vector());
      assertThat(embeddings.get(0).vector()).isNotEqualTo(embeddings.get(1).vector());
    }

    @Test
    void embedsAnEmptyBatchWithoutFailing() {
      assertThat(new ModelsEmbeddingModel(new ScriptedBackend()).embedAll(List.of()).content())
          .isEmpty();
    }
  }

  @Nested
  static class Lifecycle {

    @Test
    void closingReleasesTheBackend() {
      ScriptedBackend backend = new ScriptedBackend();
      ModelsEmbeddingModel embedding = new ModelsEmbeddingModel(backend);

      embedding.close();

      assertThat(backend.closed).isTrue();
    }

    @Test
    void rejectsANullBackend() {
      assertThatThrownBy(() -> new ModelsEmbeddingModel(null))
          .isInstanceOf(NullPointerException.class);
    }
  }

  private static final class RejectConcurrentBackend implements EmbeddingBackend {
    private final AtomicBoolean active = new AtomicBoolean();
    private final CountDownLatch firstEntered = new CountDownLatch(1);
    private final CountDownLatch releaseFirst = new CountDownLatch(1);
    private volatile boolean concurrentEntry;

    @Override
    public int dimension() {
      return DIM;
    }

    @Override
    public float[] embed(String text) {
      if (!active.compareAndSet(false, true)) {
        concurrentEntry = true;
        throw new IllegalStateException("concurrent backend entry");
      }
      try {
        if (text.equals("first")) {
          firstEntered.countDown();
          try {
            if (!releaseFirst.await(1, TimeUnit.SECONDS)) {
              throw new IllegalStateException("timed out waiting to release first embedding");
            }
          } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", exception);
          }
        }
        return new float[DIM];
      } finally {
        active.set(false);
      }
    }
  }
}
