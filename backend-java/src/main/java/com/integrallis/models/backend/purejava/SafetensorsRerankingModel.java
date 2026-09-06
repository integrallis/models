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
package com.integrallis.models.backend.purejava;

import com.integrallis.models.api.RerankingModel;
import com.integrallis.models.backend.purejava.deberta.DebertaV2Engine;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Objects;

/** Pure-Java DeBERTa-v2 cross-encoder loaded from Hugging Face Safetensors. */
public final class SafetensorsRerankingModel implements RerankingModel {

  private final DebertaV2Engine engine;
  private boolean closed;

  private SafetensorsRerankingModel(DebertaV2Engine engine) {
    this.engine = engine;
  }

  /** Maps, validates, and expands a supported DeBERTa-v2 reranker directory. */
  public static SafetensorsRerankingModel load(Path directory) {
    Objects.requireNonNull(directory, "directory");
    try {
      return new SafetensorsRerankingModel(DebertaV2Engine.load(directory));
    } catch (IOException failure) {
      throw new UncheckedIOException("Failed to load reranking model: " + directory, failure);
    }
  }

  @Override
  public synchronized double score(String query, String document) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(document, "document");
    checkOpen();
    return engine.score(query, document);
  }

  @Override
  public synchronized void close() {
    closed = true;
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("reranking model is closed");
    }
  }
}
