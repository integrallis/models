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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Minimal SPI for an embedding model that produces fixed-dimension float vectors. The interface is
 * deliberately scoped so a future fp16/bf16 implementation can plug in alongside the float[] path
 * without breaking callers (see {@link #embed(String)} for the ownership contract).
 *
 * <p>Distinct from {@code com.integrallis.models.api.InferenceBackend}: that SPI runs a generative
 * forward pass (token in → logits out). This SPI is for embedding models — input text in, fixed
 * sentence-level vector out — which has different lifecycle and batching semantics.
 *
 * <h2>Vector ownership</h2>
 *
 * <p>The {@code float[]} returned by {@link #embed(String)} and the {@code float[][]} returned by
 * {@link #embedAll(List)} are <b>caller-owned</b>: the implementation must not retain or mutate the
 * array after returning. Callers (notably {@code VectorCollectionEmbeddingSink}) hand the array
 * directly to {@code vectors-db}, which is documented to defensively clone at the staging boundary
 * — so the caller may reuse its own buffers freely once the call returns. This is the "no upcast
 * round-trip" path the audit calls out under ROADMAP II.12 F4: the embedder produces a native
 * {@code float[]} and the storage layer consumes it without an intermediate copy beyond the
 * documented staging clone.
 */
public interface EmbeddingBackend extends AutoCloseable {

  /** Returns the fixed embedding dimension produced by this backend. */
  int dimension();

  /**
   * Embeds a single text into a fixed-dimension float vector.
   *
   * @param text the input text; must not be null
   * @return a fresh {@code float[]} of length {@link #dimension()}; the implementation must not
   *     retain or mutate the array after returning
   * @throws NullPointerException if {@code text} is null
   */
  float[] embed(String text);

  /**
   * Embeds a batch of texts. The default implementation calls {@link #embed(String)} per item;
   * implementations that can amortise per-call setup cost (e.g. ONNX session pooling, GPU batching)
   * should override.
   *
   * @param texts the input texts; the list and its elements must not be null
   * @return a 2-D matrix where row {@code i} corresponds to {@code texts.get(i)}; each row is a
   *     fresh array of length {@link #dimension()}
   * @throws NullPointerException if {@code texts} or any element is null
   */
  default float[][] embedAll(List<String> texts) {
    Objects.requireNonNull(texts, "texts must not be null");
    float[][] out = new float[texts.size()][];
    for (int i = 0; i < texts.size(); i++) {
      out[i] = embed(Objects.requireNonNull(texts.get(i), "texts must not contain null"));
    }
    return out;
  }

  /**
   * Default {@link #close()} is a no-op for backends that hold no native resources. ONNX/native
   * implementations should override.
   */
  @Override
  default void close() {}

  /** Convenience: returns a {@link List} view of the rows produced by {@link #embedAll}. */
  default List<float[]> embedAllAsList(List<String> texts) {
    float[][] rows = embedAll(texts);
    List<float[]> view = new ArrayList<>(rows.length);
    for (float[] row : rows) view.add(row);
    return view;
  }
}
