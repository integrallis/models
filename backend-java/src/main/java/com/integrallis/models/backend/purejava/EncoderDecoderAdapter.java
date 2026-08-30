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

import com.integrallis.models.api.LogitBatch;
import java.util.Objects;

/**
 * Presents an encoder through the backend's decoder contract.
 *
 * <p>Almost every method here refuses. That is the honest adaptation: an encoder has no language
 * model head and no notion of a next token, so there is no meaningful value to return from {@code
 * forward} or {@code verify}. Refusing loudly beats returning a zero vector or the pooled embedding
 * under a generative name, either of which would let a caller build something that appears to
 * generate and does not.
 */
final class EncoderDecoderAdapter implements PureJavaDecoder {

  private final SequenceEncoder encoder;

  EncoderDecoderAdapter(SequenceEncoder encoder) {
    this.encoder = Objects.requireNonNull(encoder, "encoder");
  }

  @Override
  public float[] embedSequence(int[] tokens) {
    return encoder.encode(tokens);
  }

  @Override
  public boolean supportsSequenceEmbedding() {
    return true;
  }

  @Override
  public int maxBatchSize() {
    return 1;
  }

  @Override
  public void rewind(int checkpoint) {
    // An encoder holds no state between sequences, so there is nothing to rewind to.
  }

  @Override
  public void reset() {
    // Likewise nothing to reset: every sequence is encoded from scratch.
  }

  @Override
  public int checkpoint() {
    return 0;
  }

  @Override
  public void close() {}

  @Override
  public float[] forward(int token, int position) {
    throw generationUnsupported();
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    throw generationUnsupported();
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    throw generationUnsupported();
  }

  @Override
  public Session openSession() {
    throw generationUnsupported();
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    throw generationUnsupported();
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    throw generationUnsupported();
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    throw generationUnsupported();
  }

  @Override
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    throw generationUnsupported();
  }

  @Override
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    throw generationUnsupported();
  }

  @Override
  public void rewind(Session session, int checkpoint) {
    throw generationUnsupported();
  }

  @Override
  public void reset(Session session) {
    throw generationUnsupported();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    throw generationUnsupported();
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    throw generationUnsupported();
  }

  private static UnsupportedOperationException generationUnsupported() {
    return new UnsupportedOperationException(
        "this model is an encoder: it has no language model head and cannot generate tokens");
  }
}
