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

import com.integrallis.models.api.RerankingModel;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import java.util.List;
import java.util.Objects;

/** LangChain4j {@link ScoringModel} backed by an in-process Models cross-encoder. */
public final class ModelsScoringModel implements ScoringModel, AutoCloseable {

  private final RerankingModel model;
  private final Object modelMonitor = new Object();

  public ModelsScoringModel(RerankingModel model) {
    this.model = Objects.requireNonNull(model, "model");
  }

  @Override
  public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
    Objects.requireNonNull(segments, "segments");
    Objects.requireNonNull(query, "query");
    List<String> documents =
        segments.stream()
            .map(
                segment -> Objects.requireNonNull(segment, "segments must not contain null").text())
            .toList();
    synchronized (modelMonitor) {
      return Response.from(model.scoreAll(query, documents));
    }
  }

  /** Releases the mapped reranking model. */
  @Override
  public void close() {
    synchronized (modelMonitor) {
      try {
        model.close();
      } catch (RuntimeException failure) {
        throw failure;
      } catch (Exception failure) {
        throw new IllegalStateException("failed to close reranking model", failure);
      }
    }
  }
}
