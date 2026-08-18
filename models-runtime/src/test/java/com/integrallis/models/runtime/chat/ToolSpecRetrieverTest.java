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
package com.integrallis.models.runtime.chat;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.models.api.ToolSpec;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ToolSpecRetrieverTest {

  private static final ToolSpec LIGHTS =
      new ToolSpec(
          "set_lights",
          "Turn room lights on or off.",
          "{\"type\":\"object\",\"properties\":{\"room\":{\"type\":\"string\"}}}");
  private static final ToolSpec WEATHER =
      new ToolSpec(
          "get_weather",
          "Get the forecast for a city.",
          "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}");
  private static final ToolSpec TIMER =
      new ToolSpec(
          "set_timer",
          "Start a countdown timer.",
          "{\"type\":\"object\",\"properties\":{\"minutes\":{\"type\":\"integer\"}}}");

  @Test
  void ranksToolsByCosineSimilarityToTheQuery() {
    RecordingEmbeddingBackend embeddings =
        new RecordingEmbeddingBackend(
            List.of(vector(1.0f, 0.0f), vector(0.0f, 1.0f), vector(0.7f, 0.7f)),
            vector(0.95f, 0.05f));
    ToolSpecRetriever retriever =
        new ToolSpecRetriever(embeddings, List.of(LIGHTS, WEATHER, TIMER));

    List<ToolSpecRetriever.Match> matches = retriever.select("switch on the kitchen lights", 2);

    assertThat(matches)
        .extracting(match -> match.tool().name())
        .containsExactly("set_lights", "set_timer");
    assertThat(matches)
        .extracting(ToolSpecRetriever.Match::score)
        .isSortedAccordingTo((a, b) -> Float.compare(b, a));
    assertThat(embeddings.embeddedDocuments)
        .containsExactly(
            "set_lights\nTurn room lights on or off.\n"
                + "{\"type\":\"object\",\"properties\":{\"room\":{\"type\":\"string\"}}}",
            "get_weather\nGet the forecast for a city.\n"
                + "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}",
            "set_timer\nStart a countdown timer.\n"
                + "{\"type\":\"object\",\"properties\":{\"minutes\":{\"type\":\"integer\"}}}");
    assertThat(embeddings.embeddedQueries).containsExactly("switch on the kitchen lights");
  }

  @Test
  void tiesKeepDeclarationOrder() {
    RecordingEmbeddingBackend embeddings =
        new RecordingEmbeddingBackend(
            List.of(vector(1.0f, 0.0f), vector(1.0f, 0.0f)), vector(1.0f, 0.0f));
    ToolSpecRetriever retriever = new ToolSpecRetriever(embeddings, List.of(LIGHTS, WEATHER));

    assertThat(retriever.select("status", 2))
        .extracting(match -> match.tool().name())
        .containsExactly("set_lights", "get_weather");
  }

  @Test
  void rejectsInvalidEmbeddingRows() {
    RecordingEmbeddingBackend embeddings =
        new RecordingEmbeddingBackend(List.of(vector(0.0f, 0.0f)), vector(1.0f, 0.0f));

    assertThatThrownBy(() -> new ToolSpecRetriever(embeddings, List.of(LIGHTS)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("zero");
  }

  private static float[] vector(float... values) {
    return values;
  }

  private static final class RecordingEmbeddingBackend implements EmbeddingBackend {
    private final List<float[]> documentVectors;
    private final float[] queryVector;
    private final List<String> embeddedDocuments = new ArrayList<>();
    private final List<String> embeddedQueries = new ArrayList<>();

    private RecordingEmbeddingBackend(List<float[]> documentVectors, float[] queryVector) {
      this.documentVectors = documentVectors;
      this.queryVector = queryVector;
    }

    @Override
    public int dimension() {
      return queryVector.length;
    }

    @Override
    public float[] embed(String text) {
      embeddedQueries.add(text);
      return queryVector.clone();
    }

    @Override
    public float[][] embedAll(List<String> texts) {
      embeddedDocuments.addAll(texts);
      float[][] vectors = new float[documentVectors.size()][];
      for (int index = 0; index < documentVectors.size(); index++) {
        vectors[index] = documentVectors.get(index).clone();
      }
      return vectors;
    }
  }
}
