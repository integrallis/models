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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.EmbeddingBackend;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationHandler;
import io.micrometer.observation.ObservationRegistry;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationContext;

@Tag("unit")
class ModelsSpringAiEmbeddingModelTest {

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

  private static ModelsSpringAiEmbeddingModel model(ScriptedBackend backend) {
    return new ModelsSpringAiEmbeddingModel(backend);
  }

  @Nested
  static class Contract {

    @Test
    void reportsTheBackendDimension() {
      assertThat(model(new ScriptedBackend()).dimensions()).isEqualTo(DIM);
    }

    @Test
    void embedsASingleString() {
      ScriptedBackend backend = new ScriptedBackend();

      float[] vector = model(backend).embed("hello");

      assertThat(vector).hasSize(DIM);
      assertThat(backend.seen).containsExactly("hello");
    }

    @Test
    void embedsADocumentUsingItsText() {
      ScriptedBackend backend = new ScriptedBackend();

      model(backend).embed(new Document("the text"));

      assertThat(backend.seen).containsExactly("the text");
    }
  }

  @Nested
  static class BatchResponses {

    @Test
    void indexesResultsInRequestOrder() {
      // Spring AI callers rely on index to correlate a vector with its input.
      ScriptedBackend backend = new ScriptedBackend();

      EmbeddingResponse response =
          model(backend).call(new EmbeddingRequest(List.of("alpha", "beta", "gamma"), null));

      assertThat(response.getResults()).hasSize(3);
      assertThat(response.getResults().get(0).getIndex()).isZero();
      assertThat(response.getResults().get(1).getIndex()).isEqualTo(1);
      assertThat(response.getResults().get(2).getIndex()).isEqualTo(2);
      assertThat(backend.seen).containsExactly("alpha", "beta", "gamma");
    }

    @Test
    void embedsAnEmptyRequestWithoutFailing() {
      EmbeddingResponse response =
          model(new ScriptedBackend()).call(new EmbeddingRequest(List.of(), null));

      assertThat(response.getResults()).isEmpty();
    }

    @Test
    void identicalInputsEmbedIdentically() {
      ScriptedBackend backend = new ScriptedBackend();

      EmbeddingResponse response =
          model(backend).call(new EmbeddingRequest(List.of("same", "same"), null));

      assertThat(response.getResults().get(0).getOutput())
          .containsExactly(response.getResults().get(1).getOutput());
    }
  }

  @Nested
  static class Lifecycle {

    @Test
    void closingReleasesTheBackend() {
      ScriptedBackend backend = new ScriptedBackend();
      ModelsSpringAiEmbeddingModel embedding = model(backend);

      embedding.close();

      assertThat(backend.closed).isTrue();
    }

    @Test
    void rejectsANullBackend() {
      assertThatThrownBy(() -> new ModelsSpringAiEmbeddingModel(null))
          .isInstanceOf(NullPointerException.class);
    }
  }

  @Nested
  static class Observability {

    @Test
    void observesConvenienceEmbeddingCallsWithThePinnedModelIdentity() {
      ScriptedBackend backend = new ScriptedBackend();
      ObservationRegistry registry = ObservationRegistry.create();
      List<EmbeddingModelObservationContext> stopped = new ArrayList<>();
      registry
          .observationConfig()
          .observationHandler(
              new ObservationHandler<EmbeddingModelObservationContext>() {
                @Override
                public boolean supportsContext(Observation.Context context) {
                  return context instanceof EmbeddingModelObservationContext;
                }

                @Override
                public void onStop(EmbeddingModelObservationContext context) {
                  stopped.add(context);
                }
              });
      var embedding =
          new ModelsSpringAiEmbeddingModel(backend, "embeddinggemma-300m-q8_0", registry);

      embedding.embed("hello");

      assertThat(stopped)
          .singleElement()
          .satisfies(
              context -> {
                assertThat(context.getOperationMetadata().provider()).isEqualTo("integrallis");
                assertThat(context.getRequest().getOptions().getModel())
                    .isEqualTo("embeddinggemma-300m-q8_0");
                assertThat(context.getRequest().getOptions().getDimensions()).isEqualTo(DIM);
                assertThat(context.getResponse().getMetadata().getModel())
                    .isEqualTo("embeddinggemma-300m-q8_0");
              });
      assertThat(backend.seen).containsExactly("hello");
    }
  }
}
