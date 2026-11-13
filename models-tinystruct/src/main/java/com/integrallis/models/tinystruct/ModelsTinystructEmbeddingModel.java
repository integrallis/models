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
package com.integrallis.models.tinystruct;

import com.integrallis.models.api.EmbeddingBackend;
import java.util.List;
import java.util.Objects;
import org.tinystruct.data.component.Builder;
import org.tinystruct.data.component.Builders;

/**
 * In-process embedding adapter for tinystruct applications backed by an {@link EmbeddingBackend}.
 *
 * <p>Provides both raw {@code float[]} vectors and structured {@link Builder} / {@link Builders}
 * representations suitable for tinystruct JSON serialization and {@code @Action} responses.
 */
public final class ModelsTinystructEmbeddingModel implements AutoCloseable {

  private static final String DEFAULT_MODEL_NAME = "local";

  private final EmbeddingBackend backend;
  private final String modelName;
  private final Object backendMonitor = new Object();

  /** Creates an embedding adapter with the default model name ({@code "local"}). */
  public ModelsTinystructEmbeddingModel(EmbeddingBackend backend) {
    this(backend, DEFAULT_MODEL_NAME);
  }

  /** Creates an embedding adapter with an explicit model identifier. */
  public ModelsTinystructEmbeddingModel(EmbeddingBackend backend, String modelName) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.modelName = requireText(modelName, "modelName");
  }

  /** Returns the configured model identifier. */
  public String modelName() {
    return modelName;
  }

  /** Returns the embedding vector dimension of the underlying backend. */
  public int dimensions() {
    return backend.dimension();
  }

  /** Computes the embedding vector for a single text input. */
  public float[] embed(String text) {
    Objects.requireNonNull(text, "text");
    synchronized (backendMonitor) {
      return backend.embed(text);
    }
  }

  /**
   * Computes embeddings for a batch of text inputs and returns them as a tinystruct {@link
   * Builders} list.
   *
   * @param texts the input texts to embed
   * @return a {@link Builders} containing one {@link Builder} per input with {@code index}, {@code
   *     model}, {@code dimensions}, and {@code embedding}
   */
  public Builders embedAll(List<String> texts) {
    Objects.requireNonNull(texts, "texts");
    Builders results = new Builders();
    synchronized (backendMonitor) {
      for (int index = 0; index < texts.size(); index++) {
        String text = Objects.requireNonNull(texts.get(index), "texts must not contain null");
        float[] vector = backend.embed(text);
        Builder item = new Builder();
        item.put("index", index);
        item.put("model", modelName);
        item.put("dimensions", vector.length);
        item.put("embedding", toDoubleArray(vector));
        results.add(item);
      }
    }
    return results;
  }

  @Override
  public void close() {
    synchronized (backendMonitor) {
      try {
        backend.close();
      } catch (RuntimeException failure) {
        throw failure;
      } catch (Exception failure) {
        throw new IllegalStateException("failed to close embedding backend", failure);
      }
    }
  }

  private static double[] toDoubleArray(float[] vector) {
    double[] values = new double[vector.length];
    for (int index = 0; index < vector.length; index++) {
      values[index] = vector[index];
    }
    return values;
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }
}
