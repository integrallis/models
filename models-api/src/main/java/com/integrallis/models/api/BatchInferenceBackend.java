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
package com.integrallis.models.api;

import java.util.Objects;

/** Optional backend capability for batching independent autoregressive sessions. */
public interface BatchInferenceBackend extends InferenceBackend {

  /** Returns the largest independent-session batch accepted by this loaded model. */
  int maxBatchSize();

  /** Opens empty request state that shares this backend's loaded model. */
  InferenceSession openSession();

  /** Runs one session step and returns stable logits. */
  float[] forward(InferenceSession session, int token, int position);

  /** Runs one session step whose returned storage may be reused by the next backend invocation. */
  default float[] forwardTransient(InferenceSession session, int token, int position) {
    return forward(session, token, position);
  }

  /** Processes a contiguous prompt for one independent session. */
  default float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
    Objects.requireNonNull(session, "session");
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition < 0) {
      throw new IllegalArgumentException("startPosition must be >= 0");
    }

    float[] logits = null;
    for (int index = 0; index < tokens.length; index++) {
      logits = forwardTransient(session, tokens[index], Math.addExact(startPosition, index));
    }
    return logits;
  }

  /**
   * Returns whether this backend combines different-length independent-session prompts in shared
   * physical model passes rather than using the sequential compatibility implementation.
   */
  default boolean supportsRaggedPrefillBatch() {
    return false;
  }

  /**
   * Processes one different-length prompt per independent session and returns stable final logits
   * in session order.
   *
   * <p>The compatibility implementation preserves independent state but evaluates sessions one at a
   * time. Check {@link #supportsRaggedPrefillBatch()} before assuming weight-read amortization.
   */
  default LogitBatch prefillBatch(InferenceSession[] sessions, int[][] tokenBatches) {
    Objects.requireNonNull(sessions, "sessions");
    Objects.requireNonNull(tokenBatches, "tokenBatches");
    if (sessions.length == 0) {
      throw new IllegalArgumentException("sessions must not be empty");
    }
    if (tokenBatches.length != sessions.length) {
      throw new IllegalArgumentException(
          "tokenBatches.length must equal sessions.length: "
              + tokenBatches.length
              + " != "
              + sessions.length);
    }
    if (sessions.length > maxBatchSize()) {
      throw new IllegalArgumentException(
          "session batch exceeds capacity " + maxBatchSize() + ": " + sessions.length);
    }

    int vocabularySize = metadata().vocabSize();
    float[] batchLogits = new float[Math.multiplyExact(sessions.length, vocabularySize)];
    for (int index = 0; index < sessions.length; index++) {
      InferenceSession session = Objects.requireNonNull(sessions[index], "sessions[" + index + "]");
      int[] tokens = Objects.requireNonNull(tokenBatches[index], "tokenBatches[" + index + "]");
      if (session.isClosed()) {
        throw new IllegalStateException("session " + index + " is closed");
      }
      for (int prior = 0; prior < index; prior++) {
        if (sessions[prior] == session) {
          throw new IllegalArgumentException("sessions must be distinct");
        }
      }
      float[] logits = prefill(session, tokens, session.checkpoint());
      if (logits.length < vocabularySize) {
        throw new IllegalStateException(
            "backend returned " + logits.length + " logits for vocabulary size " + vocabularySize);
      }
      System.arraycopy(logits, 0, batchLogits, index * vocabularySize, vocabularySize);
    }
    return new LogitBatch(sessions.length, vocabularySize, batchLogits);
  }

  /**
   * Processes one prompt per session and may reuse the returned storage on the next backend call.
   */
  default LogitBatch prefillBatchTransient(InferenceSession[] sessions, int[][] tokenBatches) {
    return prefillBatch(sessions, tokenBatches);
  }

  /** Runs one token for every session and returns stable session-major logits. */
  LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens);

  /**
   * Runs one token for every session and may reuse the returned storage on the next backend call.
   */
  default LogitBatch forwardBatchTransient(InferenceSession[] sessions, int[] tokens) {
    return forwardBatch(sessions, tokens);
  }

  /** Discards session state at and after {@code checkpoint}. */
  void rewind(InferenceSession session, int checkpoint);

  /** Clears one session without changing any other sequence or the default backend state. */
  void reset(InferenceSession session);
}
