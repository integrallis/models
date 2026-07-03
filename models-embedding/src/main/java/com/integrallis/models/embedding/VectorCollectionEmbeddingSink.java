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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.VectorCollection;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Bridges an {@link EmbeddingBackend} into a {@link VectorCollection}: the caller hands over an
 * {@code id} and {@code text}, the sink invokes the backend, and the resulting {@code float[]} is
 * placed directly on the collection via {@link Document#of(String, float[], String)}. No
 * intermediate copy between the embedder's output and the collection's staging buffer (the staging
 * buffer's defensive clone is the only allocation; the embedder's array is consumed once and
 * reusable as soon as {@code put} returns).
 *
 * <p>Closes the documented audit gap II.12 F4 — the "zero-copy {@code models ↔ vectors} pipeline".
 * Before this sink existed, every framework adapter (Spring AI, LangChain4j, …) had to define its
 * own bridge code, each with its own copy semantics. The sink consolidates that surface.
 *
 * <p>Dimension validation runs at construction time: the backend's reported {@link
 * EmbeddingBackend#dimension()} must equal the collection's configured dimension, otherwise the
 * sink throws {@link IllegalArgumentException} before any embed() call is made (the alternative —
 * defer until first put() — surfaces a cryptic vectors-db error at insert time).
 */
public final class VectorCollectionEmbeddingSink {

  private final EmbeddingBackend backend;
  private final VectorCollection collection;

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
  }

  /**
   * Embeds {@code text} and inserts the resulting vector under {@code id}, with {@code text}
   * carried as the document's text field. Does NOT commit — the caller batches commits as it sees
   * fit.
   */
  public void put(String id, String text) {
    Objects.requireNonNull(id, "id must not be null");
    Objects.requireNonNull(text, "text must not be null");
    float[] vector = backend.embed(text);
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
    Objects.requireNonNull(ids, "ids must not be null");
    Objects.requireNonNull(texts, "texts must not be null");
    if (ids.size() != texts.size()) {
      throw new IllegalArgumentException(
          "ids.size() (" + ids.size() + ") != texts.size() (" + texts.size() + ")");
    }
    if (texts.isEmpty()) return;
    float[][] vectors = backend.embedAll(texts);
    if (vectors.length != texts.size()) {
      throw new IllegalStateException(
          "backend returned "
              + vectors.length
              + " vectors for "
              + texts.size()
              + " texts — embedAll contract violation");
    }
    List<Document> docs = new ArrayList<>(texts.size());
    for (int i = 0; i < texts.size(); i++) {
      String id = Objects.requireNonNull(ids.get(i), "ids must not contain null");
      String text = Objects.requireNonNull(texts.get(i), "texts must not contain null");
      docs.add(Document.of(id, vectors[i], text));
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

  /** Returns the backing {@link EmbeddingBackend}. */
  public EmbeddingBackend backend() {
    return backend;
  }

  /** Returns the backing {@link VectorCollection}. */
  public VectorCollection collection() {
    return collection;
  }
}
