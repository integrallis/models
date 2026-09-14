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

import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.LogitBatch;
import java.io.IOException;
import java.util.Optional;
import java.util.OptionalLong;

/** Architecture-neutral decoder contract retained inside the pure-Java backend. */
interface PureJavaDecoder extends AutoCloseable {

  interface Session {
    int checkpoint();

    default OptionalLong allocatedStateBytes() {
      return OptionalLong.empty();
    }
  }

  interface SharedPrefix {
    int checkpoint();

    long sharedBytes();
  }

  int maxBatchSize();

  float[] forward(int token, int position);

  float[] forwardTransient(int token, int position);

  float[] prefill(int[] tokens, int startPosition);

  /**
   * Runs the stack over a sequence and returns the final position's hidden state, skipping the
   * vocabulary projection.
   *
   * <p>Architectures without an embedding path may refuse; callers should check {@link
   * #supportsHiddenState()} first.
   */
  default float[] prefillHiddenState(int[] tokens, int startPosition) {
    throw new UnsupportedOperationException(
        "this decoder does not expose hidden states for embedding");
  }

  /** Runs one step and returns its hidden state, skipping the vocabulary projection. */
  default float[] hiddenState(int token, int position) {
    throw new UnsupportedOperationException(
        "this decoder does not expose hidden states for embedding");
  }

  /** Whether {@link #prefillHiddenState(int[], int)} is implemented for this architecture. */
  default boolean supportsHiddenState() {
    return false;
  }

  /**
   * Encodes a whole sequence into one vector using the model's own pooling and projection.
   *
   * <p>Distinct from {@link #prefillHiddenState(int[], int)}, which returns one position's state
   * and leaves pooling to the caller. Models that own their embedding pipeline — an encoder with a
   * declared pooling type, a sentence-transformer with a dense head — cannot express it that way,
   * because the vector they were trained to produce is a function of the whole sequence.
   *
   * @param tokens the tokenized text
   * @return the pooled embedding, not normalized
   */
  default float[] embedSequence(int[] tokens) {
    throw new UnsupportedOperationException(
        "this architecture does not encode whole sequences; pool its hidden states instead");
  }

  /** Whether {@link #embedSequence(int[])} is implemented for this architecture. */
  default boolean supportsSequenceEmbedding() {
    return false;
  }

  default boolean supportsContrastiveEncoding() {
    return false;
  }

  default int contrastiveDimension() {
    throw new UnsupportedOperationException("this architecture has no contrastive head");
  }

  default float[] encodeContrastive(int[] tokens) {
    throw new UnsupportedOperationException("this architecture has no contrastive head");
  }

  default boolean supportsConfidenceScoring() {
    return false;
  }

  default float scoreConfidence(int[] tokens) {
    throw new UnsupportedOperationException("this architecture has no confidence head");
  }

  Session openSession();

  default boolean supportsSharedPrefixes() {
    return false;
  }

  default boolean supportsActivatedBranch() {
    return false;
  }

  default Optional<ActivatedAdapterMetadata> activatedAdapter() {
    return Optional.empty();
  }

  default void activateAdapter(Session session) {
    throw new UnsupportedOperationException("this decoder has no activated adapter branch");
  }

  default SharedPrefix freezePrefix(Session source) {
    throw new UnsupportedOperationException("this decoder cannot share KV-cache prefixes");
  }

  default Session fork(SharedPrefix prefix, boolean activated) {
    throw new UnsupportedOperationException("this decoder cannot share KV-cache prefixes");
  }

  default boolean sharesPrefixStorage(Session first, Session second) {
    return false;
  }

  float[] forward(Session session, int token, int position);

  float[] forwardTransient(Session session, int token, int position);

  float[] prefill(Session session, int[] tokens, int startPosition);

  default float[] hiddenState(Session session, int token, int position) {
    throw new UnsupportedOperationException("this decoder does not expose session hidden states");
  }

  default float[] hiddenStateTransient(Session session, int token, int position) {
    return hiddenState(session, token, position);
  }

  default float[] prefillHiddenState(Session session, int[] tokens, int startPosition) {
    throw new UnsupportedOperationException("this decoder does not expose session hidden states");
  }

  default boolean supportsRaggedPrefillBatch() {
    return false;
  }

  default LogitBatch prefillBatch(Session[] sessions, int[][] tokenBatches) {
    throw new UnsupportedOperationException("this decoder does not batch ragged session prefill");
  }

  default LogitBatch prefillBatchTransient(Session[] sessions, int[][] tokenBatches) {
    return prefillBatch(sessions, tokenBatches);
  }

  LogitBatch forwardBatch(Session[] sessions, int[] tokens);

  LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens);

  void rewind(Session session, int checkpoint);

  void reset(Session session);

  int checkpoint();

  LogitBatch verify(int[] tokens, int startPosition);

  LogitBatch verifyTransient(int[] tokens, int startPosition);

  void rewind(int checkpoint);

  void reset();

  @Override
  void close() throws IOException;
}
