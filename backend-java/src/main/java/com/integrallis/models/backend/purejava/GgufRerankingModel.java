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
import com.integrallis.models.backend.purejava.bert.BertClassificationHead;
import com.integrallis.models.backend.purejava.bert.BertConfig;
import com.integrallis.models.backend.purejava.bert.BertForwardPass;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Objects;

/** Pure-Java BERT cross-encoder loaded directly from a corrected GGUF artifact. */
public final class GgufRerankingModel implements RerankingModel {

  private final Arena arena;
  private final GgufTokenizer tokenizer;
  private final BertForwardPass encoder;
  private final BertClassificationHead classifier;
  private final int contextLength;
  private boolean closed;

  private GgufRerankingModel(
      Arena arena,
      GgufTokenizer tokenizer,
      BertForwardPass encoder,
      BertClassificationHead classifier,
      int contextLength) {
    this.arena = arena;
    this.tokenizer = tokenizer;
    this.encoder = encoder;
    this.classifier = classifier;
    this.contextLength = contextLength;
  }

  /** Maps, validates, and loads a BERT reranker. The returned model owns the mapping. */
  public static GgufRerankingModel load(Path modelPath) {
    Objects.requireNonNull(modelPath, "modelPath");
    Arena arena = Arena.ofShared();
    try {
      GgufFile file = GgufParser.parse(modelPath, arena);
      BertConfig config = BertConfig.fromMetadata(file.metadata());
      return new GgufRerankingModel(
          arena,
          GgufTokenizer.fromMetadata(file.metadata()),
          BertForwardPass.fromGgufFile(file, config),
          BertClassificationHead.fromGgufFile(file, config.embeddingDim()),
          config.contextLength());
    } catch (IOException failure) {
      arena.close();
      throw new UncheckedIOException("Failed to load reranking model: " + modelPath, failure);
    } catch (RuntimeException | Error failure) {
      arena.close();
      throw failure;
    }
  }

  @Override
  public synchronized double score(String query, String document) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(document, "document");
    checkOpen();
    GgufTokenizer.TokenizedPair pair = tokenizer.encodePair(query, document, contextLength);
    return classifier.score(encoder.encodeCls(pair.tokens(), pair.tokenTypes()));
  }

  @Override
  public synchronized void close() {
    if (!closed) {
      closed = true;
      arena.close();
    }
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("reranking model is closed");
    }
  }
}
