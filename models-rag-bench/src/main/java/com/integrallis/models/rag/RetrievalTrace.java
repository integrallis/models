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
package com.integrallis.models.rag;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

final class RetrievalTrace {
  private final RagRetriever retriever;
  private final int topK;
  private final AtomicReference<List<RetrievedDocument>> hits = new AtomicReference<>();
  private final AtomicLong elapsedNanos = new AtomicLong(-1);

  RetrievalTrace(RagRetriever retriever, int topK) {
    this.retriever = retriever;
    this.topK = topK;
  }

  void clear() {
    hits.set(null);
    elapsedNanos.set(-1);
  }

  List<RetrievedDocument> retrieve(String question) {
    long start = System.nanoTime();
    try {
      List<RetrievedDocument> retrieved = retriever.retrieve(question, topK);
      hits.set(retrieved);
      elapsedNanos.set(System.nanoTime() - start);
      return retrieved;
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  List<RetrievedDocument> hits() {
    List<RetrievedDocument> value = hits.get();
    if (value == null) {
      throw new IllegalStateException("framework did not invoke retrieval");
    }
    return value;
  }

  double elapsedMillis() {
    long nanos = elapsedNanos.get();
    if (nanos < 0) {
      throw new IllegalStateException("framework did not invoke retrieval");
    }
    return nanos / 1_000_000.0;
  }
}
