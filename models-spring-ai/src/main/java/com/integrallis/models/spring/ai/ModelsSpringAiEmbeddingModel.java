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
package com.integrallis.models.spring.ai;

import com.integrallis.models.api.EmbeddingBackend;
import io.micrometer.observation.ObservationRegistry;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import org.springframework.ai.embedding.observation.DefaultEmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationContext;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationDocumentation;

/**
 * Spring AI {@link EmbeddingModel} backed by an in-process {@link EmbeddingBackend}.
 *
 * <p>This is what lets a Spring AI application keep embeddings inside the JVM. It matters more than
 * it first appears: swapping a chat model is a prompt-regression exercise, while swapping an
 * embedding model invalidates every stored vector and forces a full re-index. A pinned local model
 * has no retirement date.
 *
 * <p>Pooling and normalization are fixed by the backend, deliberately: they are properties of the
 * embedding model, and letting a caller vary them per request would silently produce vectors that
 * no longer match the ones already in the store.
 */
public final class ModelsSpringAiEmbeddingModel implements EmbeddingModel, AutoCloseable {

  private static final String DEFAULT_MODEL_NAME = "local";
  private static final String PROVIDER = "integrallis";
  private static final EmbeddingModelObservationConvention DEFAULT_OBSERVATION_CONVENTION =
      new DefaultEmbeddingModelObservationConvention();

  private final EmbeddingBackend backend;
  private final Object backendMonitor = new Object();
  private final String modelName;
  private final ObservationRegistry observationRegistry;
  private EmbeddingModelObservationConvention observationConvention =
      DEFAULT_OBSERVATION_CONVENTION;

  public ModelsSpringAiEmbeddingModel(EmbeddingBackend backend) {
    this(backend, DEFAULT_MODEL_NAME, ObservationRegistry.NOOP);
  }

  public ModelsSpringAiEmbeddingModel(
      EmbeddingBackend backend, String modelName, ObservationRegistry observationRegistry) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.modelName = requireText(modelName, "modelName");
    this.observationRegistry = Objects.requireNonNull(observationRegistry, "observationRegistry");
  }

  @Override
  public EmbeddingResponse call(EmbeddingRequest request) {
    Objects.requireNonNull(request, "request");
    var observedRequest =
        new EmbeddingRequest(
            request.getInstructions(),
            EmbeddingOptions.builder().model(modelName).dimensions(dimensions()).build());
    var context =
        EmbeddingModelObservationContext.builder()
            .embeddingRequest(observedRequest)
            .provider(PROVIDER)
            .build();
    return EmbeddingModelObservationDocumentation.EMBEDDING_MODEL_OPERATION
        .observation(
            observationConvention,
            DEFAULT_OBSERVATION_CONVENTION,
            () -> context,
            observationRegistry)
        .observe(() -> embedAll(observedRequest, context));
  }

  @Override
  public float[] embed(String text) {
    Objects.requireNonNull(text, "text");
    return call(new EmbeddingRequest(List.of(text), null)).getResult().getOutput();
  }

  @Override
  public float[] embed(Document document) {
    Objects.requireNonNull(document, "document");
    return embed(getEmbeddingContent(document));
  }

  @Override
  public int dimensions() {
    return backend.dimension();
  }

  public void setObservationConvention(EmbeddingModelObservationConvention observationConvention) {
    this.observationConvention =
        Objects.requireNonNull(observationConvention, "observationConvention");
  }

  /** Releases the underlying model. */
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

  private EmbeddingResponse embedAll(
      EmbeddingRequest request, EmbeddingModelObservationContext context) {
    List<String> instructions = request.getInstructions();
    List<Embedding> embeddings = new ArrayList<>(instructions.size());
    synchronized (backendMonitor) {
      for (int index = 0; index < instructions.size(); index++) {
        // The index correlates each vector with its input; callers depend on it to match rows.
        embeddings.add(new Embedding(backend.embed(instructions.get(index)), index));
      }
    }
    var response =
        new EmbeddingResponse(embeddings, new EmbeddingResponseMetadata(modelName, null));
    context.setResponse(response);
    return response;
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }
}
