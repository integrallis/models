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

import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.VectorCollection;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Embeds text and writes the resulting vectors to a {@link VectorCollection}.
 *
 * <p>A successfully constructed sink owns its {@link EmbeddingBackend}; {@link #close()} releases
 * that backend exactly once. The caller retains ownership of the collection. Every returned vector
 * is checked against the declared dimension before the collection is mutated.
 *
 * <p>The embedding array is passed directly to {@link Document#of(String, float[], String)}.
 * Vectors performs its documented defensive copy at the staging boundary.
 */
public final class VectorCollectionEmbeddingSink implements AutoCloseable {

  private final EmbeddingBackend backend;
  private final VectorCollection collection;
  private final int dimension;
  private final AtomicBoolean closed = new AtomicBoolean();

  /**
   * Constructs the sink. Validates that the backend's dimension matches the collection's.
   *
   * @throws IllegalArgumentException if the dimensions disagree
   */
  public VectorCollectionEmbeddingSink(EmbeddingBackend backend, VectorCollection collection) {
    this.backend = Objects.requireNonNull(backend, "backend must not be null");
    this.collection = Objects.requireNonNull(collection, "collection must not be null");
    int backendDim = backend.dimension();
    int collDim = collection.config().dimension();
    if (backendDim != collDim) {
      throw new IllegalArgumentException(
          "EmbeddingBackend dimension ("
              + backendDim
              + ") does not match VectorCollection dimension ("
              + collDim
              + ")");
    }
    this.dimension = backendDim;
  }

  /**
   * Embeds {@code text} and inserts the resulting vector under {@code id}, with {@code text}
   * carried as the document's text field. Does NOT commit — the caller batches commits as it sees
   * fit.
   */
  public void put(String id, String text) {
    ensureOpen();
    Objects.requireNonNull(id, "id must not be null");
    Objects.requireNonNull(text, "text must not be null");
    float[] vector = validateVector(backend.embed(text), 0);
    collection.add(Document.of(id, vector, text));
  }

  /**
   * Embeds {@code texts} as a batch (one backend call) and inserts each result paired with the
   * corresponding id. Lists must be the same length; nulls are rejected. Does NOT commit.
   *
   * @param ids ids, parallel to {@code texts}
   * @param texts texts to embed
   * @throws IllegalArgumentException if list sizes differ
   */
  public void putAll(List<String> ids, List<String> texts) {
    ensureOpen();
    Objects.requireNonNull(ids, "ids must not be null");
    Objects.requireNonNull(texts, "texts must not be null");
    if (ids.size() != texts.size()) {
      throw new IllegalArgumentException(
          "ids.size() (" + ids.size() + ") != texts.size() (" + texts.size() + ")");
    }
    if (texts.isEmpty()) return;
    for (int i = 0; i < texts.size(); i++) {
      Objects.requireNonNull(ids.get(i), "ids must not contain null");
      Objects.requireNonNull(texts.get(i), "texts must not contain null");
    }
    float[][] vectors = backend.embedAll(texts);
    if (vectors == null) {
      throw new IllegalStateException("backend returned a null vector batch");
    }
    if (vectors.length != texts.size()) {
      throw new IllegalStateException(
          "backend returned "
              + vectors.length
              + " vectors for "
              + texts.size()
              + " texts — embedAll contract violation");
    }
    for (int i = 0; i < vectors.length; i++) {
      validateVector(vectors[i], i);
    }
    List<Document> docs = new ArrayList<>(texts.size());
    for (int i = 0; i < texts.size(); i++) {
      docs.add(Document.of(ids.get(i), vectors[i], texts.get(i)));
    }
    collection.addAll(docs);
  }

  /**
   * Convenience: {@link #put(String, String)} followed by {@link VectorCollection#commit()}. Use
   * sparingly — committing once per insert is the slowest possible path. Prefer {@code putAll} +
   * one explicit commit.
   */
  public void putAndCommit(String id, String text) {
    put(id, text);
    collection.commit();
  }

  /**
   * Closes the owned embedding backend. The caller-owned collection remains open.
   *
   * <p>Repeated calls have no effect.
   */
  @Override
  public void close() {
    if (closed.compareAndSet(false, true)) {
      backend.close();
    }
  }

  /** Returns the backing {@link EmbeddingBackend}. */
  public EmbeddingBackend backend() {
    return backend;
  }

  /** Returns the backing {@link VectorCollection}. */
  public VectorCollection collection() {
    return collection;
  }

  private float[] validateVector(float[] vector, int index) {
    if (vector == null) {
      throw new IllegalStateException("backend returned a null vector for text at index " + index);
    }
    if (vector.length != dimension) {
      throw new IllegalStateException(
          "backend returned dimension "
              + vector.length
              + " for text at index "
              + index
              + "; expected "
              + dimension);
    }
    return vector;
  }

  private void ensureOpen() {
    if (closed.get()) {
      throw new IllegalStateException("embedding sink is closed");
    }
  }
}
