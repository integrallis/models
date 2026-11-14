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
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Pure-Java BERT cross-encoder loaded directly from a corrected GGUF artifact. */
public final class GgufRerankingModel implements RerankingModel {

  // Atomic because DebertaV2Engine.score is not synchronized, so a plain long would
  // under-count under concurrent scoring and a synchronized accessor would imply a
  // safety the writes do not have. Incremented only on the truncation path.
  private final java.util.concurrent.atomic.AtomicLong truncatedPairs =
      new java.util.concurrent.atomic.AtomicLong();
  private final java.util.concurrent.atomic.AtomicLong droppedTokens =
      new java.util.concurrent.atomic.AtomicLong();

  private final Arena arena;
  private final GgufTokenizer tokenizer;
  private final BertForwardPass encoder;
  private final BertClassificationHead classifier;
  private final int contextLength;
  private final java.util.concurrent.atomic.AtomicBoolean closed =
      new java.util.concurrent.atomic.AtomicBoolean();

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
    if (pair.droppedTokens() > 0) {
      truncatedPairs.incrementAndGet();
      droppedTokens.addAndGet(pair.droppedTokens());
    }
    return classifier.score(encoder.encodeCls(pair.tokens(), pair.tokenTypes()));
  }

  @Override
  public List<Double> scoreBatch(String query, List<String> documents) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(documents, "documents");
    checkOpen();
    if (documents.isEmpty()) {
      return List.of();
    }
    int batchSize = documents.size();
    GgufTokenizer.TokenizedPair[] pairs = new GgufTokenizer.TokenizedPair[batchSize];
    int maxLen = 0;
    for (int i = 0; i < batchSize; i++) {
      pairs[i] =
          tokenizer.encodePair(
              query,
              Objects.requireNonNull(documents.get(i), "documents must not contain null"),
              contextLength);
      if (pairs[i].droppedTokens() > 0) {
        truncatedPairs.incrementAndGet();
        droppedTokens.addAndGet(pairs[i].droppedTokens());
      }
      maxLen = Math.max(maxLen, pairs[i].tokens().length);
    }
    int[][] tokens = new int[batchSize][maxLen];
    int[][] tokenTypes = new int[batchSize][maxLen];
    int[] lengths = new int[batchSize];
    for (int i = 0; i < batchSize; i++) {
      int len = pairs[i].tokens().length;
      lengths[i] = len;
      System.arraycopy(pairs[i].tokens(), 0, tokens[i], 0, len);
      System.arraycopy(pairs[i].tokenTypes(), 0, tokenTypes[i], 0, len);
    }
    float[][] clsVectors = encoder.encodeBatchCls(tokens, tokenTypes, lengths);
    List<Double> scores = new ArrayList<>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      double score = classifier.score(clsVectors[i]);
      if (!Double.isFinite(score)) {
        throw new IllegalStateException("reranking model returned a non-finite score: " + score);
      }
      scores.add(score);
    }
    return List.copyOf(scores);
  }

  /**
   * How many pairs this model truncated, and how many tokens it dropped, since it was opened.
   *
   * <p>Truncation was silent. A caller scoring a long query-document pair got a score computed on a
   * prefix with no way to know: measured over LongMemEval-S, 237 of 1,785 pairs (13.3%) exceeded
   * the 512-token window, touching 99 of 119 queries, with the longest pair at 8,755 tokens. See
   * #362. Cumulative rather than per-call so a caller can report a rate over a run without
   * threading a return value through every score.
   *
   * @param truncatedPairs pairs that did not fit the window
   * @param droppedTokens tokens removed across those pairs
   */
  public record TruncationReport(long truncatedPairs, long droppedTokens) {}

  /** Returns the truncation seen so far. Never null; zeroed only by opening a new model. */
  public TruncationReport truncationReport() {
    return new TruncationReport(truncatedPairs.get(), droppedTokens.get());
  }

  @Override
  public synchronized void close() {
    if (closed.compareAndSet(false, true)) {
      arena.close();
    }
  }

  private void checkOpen() {
    if (closed.get()) {
      throw new IllegalStateException("reranking model is closed");
    }
  }
}
