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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.langchain4j.JavaVectorsEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VectorsLangChain4jInteroperabilityTest {

  interface Assistant {
    String answer(String question);
  }

  @Test
  void vectorsRetrievalAndModelsGenerationComposeInOneAiService() {
    EmbeddingModel embeddingModel = new KeywordEmbeddingModel();
    AtomicReference<String> generatedPrompt = new AtomicReference<>();
    TextGenerationModel generationModel = recordingModel(generatedPrompt);

    try (VectorCollection collection =
            VectorCollection.builder()
                .dimension(2)
                .metric(SimilarityFunction.COSINE)
                .indexType(IndexType.FLAT)
                .build();
        JavaVectorsEmbeddingStore store =
            JavaVectorsEmbeddingStore.builder(collection).commitAfterAdd(true).build()) {
      TextSegment context = TextSegment.from("The deployment region is Phoenix.");
      store.add(embeddingModel.embed(context).content(), context);

      var retriever =
          EmbeddingStoreContentRetriever.builder()
              .embeddingModel(embeddingModel)
              .embeddingStore(store)
              .maxResults(1)
              .build();
      Assistant assistant =
          AiServices.builder(Assistant.class)
              .chatModel(new ModelsChatModel(generationModel))
              .contentRetriever(retriever)
              .build();

      assertThat(assistant.answer("Where is the deployment region?")).isEqualTo("Phoenix");
      assertThat(generatedPrompt.get())
          .contains("Where is the deployment region?")
          .contains("The deployment region is Phoenix.");
    }
  }

  private static TextGenerationModel recordingModel(AtomicReference<String> generatedPrompt) {
    return new TextGenerationModel() {
      @Override
      public String modelName() {
        return "interoperability-test";
      }

      @Override
      public BackendDiagnostics diagnostics() {
        return BackendDiagnostics.unavailable("interoperability-test");
      }

      @Override
      public void generate(String prompt, SamplingOptions options, TokenStream stream) {
        generatedPrompt.set(prompt);
        stream.onToken("Phoenix");
        stream.onComplete();
      }
    };
  }

  private static final class KeywordEmbeddingModel implements EmbeddingModel {
    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
      return Response.from(segments.stream().map(this::embedding).toList());
    }

    @Override
    public int dimension() {
      return 2;
    }

    private Embedding embedding(TextSegment segment) {
      String normalized = segment.text().toLowerCase(java.util.Locale.ROOT);
      float region = normalized.contains("region") ? 1.0f : 0.0f;
      float other = normalized.contains("unrelated") ? 1.0f : 0.0f;
      return Embedding.from(new float[] {region, other});
    }
  }
}
