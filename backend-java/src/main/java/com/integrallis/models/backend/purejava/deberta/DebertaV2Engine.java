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
package com.integrallis.models.backend.purejava.deberta;

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import com.integrallis.models.backend.purejava.tokenizer.DebertaV2Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Objects;

/** Package boundary for the pure-Java DeBERTa-v2 reranking implementation. */
public final class DebertaV2Engine {

  // Atomic because DebertaV2Engine.score is not synchronized, so a plain long would
  // under-count under concurrent scoring and a synchronized accessor would imply a
  // safety the writes do not have. Incremented only on the truncation path.
  private final java.util.concurrent.atomic.AtomicLong truncatedPairs =
      new java.util.concurrent.atomic.AtomicLong();
  private final java.util.concurrent.atomic.AtomicLong droppedTokens =
      new java.util.concurrent.atomic.AtomicLong();

  private final DebertaV2Tokenizer tokenizer;
  private final DebertaV2ForwardPass forwardPass;

  private DebertaV2Engine(DebertaV2Tokenizer tokenizer, DebertaV2ForwardPass forwardPass) {
    this.tokenizer = tokenizer;
    this.forwardPass = forwardPass;
  }

  /** Loads a supported Hugging Face DeBERTa-v2 sequence classifier. */
  public static DebertaV2Engine load(Path directory) throws IOException {
    Objects.requireNonNull(directory, "directory");
    DebertaV2Config config = DebertaV2Config.parse(directory.resolve("config.json"));
    DebertaV2Tokenizer tokenizer =
        DebertaV2Tokenizer.fromJson(directory.resolve("tokenizer.json"), config.maxPositions());
    DebertaV2Weights weights;
    try (Arena arena = Arena.ofConfined()) {
      weights =
          DebertaV2Weights.load(
              new SafetensorsTensorSource(SafetensorsBundle.open(directory, arena)), config);
    }
    return new DebertaV2Engine(tokenizer, new DebertaV2ForwardPass(config, weights));
  }

  /** Scores one query-document pair. */
  public double score(String query, String document) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(document, "document");
    DebertaV2Tokenizer.TokenizedPair pair = tokenizer.encodePair(query, document);
    if (pair.droppedTokens() > 0) {
      truncatedPairs.incrementAndGet();
      droppedTokens.addAndGet(pair.droppedTokens());
    }
    return forwardPass.score(pair.tokens());
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

  /** Returns the truncation seen so far. Never null; zeroed only by opening a new engine. */
  public TruncationReport truncationReport() {
    return new TruncationReport(truncatedPairs.get(), droppedTokens.get());
  }
}
