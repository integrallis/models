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

import com.integrallis.models.api.EmbeddingBackend;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * LangChain4j {@link EmbeddingModel} backed by an in-process {@link EmbeddingBackend}.
 *
 * <p>Keeps embeddings inside the JVM, which matters more than it first appears: swapping a chat
 * model is a prompt-regression exercise, while swapping an embedding model invalidates every stored
 * vector and forces a full re-index. A pinned local model has no retirement date.
 *
 * <p>Pooling and normalization are fixed by the backend, deliberately: they are properties of the
 * embedding model, and varying them per request would silently produce vectors that no longer match
 * those already in the store.
 */
public final class ModelsEmbeddingModel implements EmbeddingModel, AutoCloseable {

  private final EmbeddingBackend backend;

  public ModelsEmbeddingModel(EmbeddingBackend backend) {
    this.backend = Objects.requireNonNull(backend, "backend");
  }

  @Override
  public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    Objects.requireNonNull(segments, "segments");
    List<Embedding> embeddings = new ArrayList<>(segments.size());
    for (TextSegment segment : segments) {
      embeddings.add(
          new Embedding(backend.embed(Objects.requireNonNull(segment, "segment").text())));
    }
    return Response.from(embeddings);
  }

  @Override
  public int dimension() {
    return backend.dimension();
  }

  /** Releases the underlying model. */
  @Override
  public void close() {
    try {
      backend.close();
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Exception failure) {
      throw new IllegalStateException("failed to close embedding backend", failure);
    }
  }
}
