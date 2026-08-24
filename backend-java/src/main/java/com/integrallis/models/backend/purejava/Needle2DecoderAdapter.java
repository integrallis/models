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
import com.integrallis.models.backend.purejava.cact.CactHeader;
import com.integrallis.models.backend.purejava.cact.Needle2ForwardPass;
import com.integrallis.models.backend.purejava.cact.Needle2Weights;
import java.util.Objects;

/** Adapts Needle 2 sequence state to the architecture-neutral backend contract. */
final class Needle2DecoderAdapter implements PureJavaDecoder {

  private final class Needle2Session implements Session {
    private final Needle2ForwardPass forwardPass = new Needle2ForwardPass(weights, capacity);

    @Override
    public int checkpoint() {
      return forwardPass.checkpoint();
    }
  }

  private final Needle2Weights weights;
  private final int capacity;
  private final Needle2ForwardPass defaultSequence;

  Needle2DecoderAdapter(Needle2Weights weights, int capacity) {
    this.weights = Objects.requireNonNull(weights, "weights");
    this.capacity = capacity;
    this.defaultSequence = new Needle2ForwardPass(weights, capacity);
  }

  CactHeader header() {
    return weights.header();
  }

  @Override
  public int maxBatchSize() {
    return 1;
  }

  @Override
  public float[] forward(int token, int position) {
    return defaultSequence.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return defaultSequence.forward(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return prefill(defaultSequence, tokens, startPosition);
  }

  @Override
  public Session openSession() {
    return new Needle2Session();
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    return requireSession(session).forwardPass.forward(token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    return requireSession(session).forwardPass.forward(token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    return prefill(requireSession(session).forwardPass, tokens, startPosition);
  }

  @Override
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    return forwardBatchTransient(sessions, tokens).snapshot();
  }

  @Override
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    Objects.requireNonNull(sessions, "sessions");
    Objects.requireNonNull(tokens, "tokens");
    if (sessions.length != 1 || tokens.length != 1) {
      throw new IllegalArgumentException(
          "Needle 2 currently supports one independent session per batch");
    }
    Needle2Session session = requireSession(sessions[0]);
    float[] logits = session.forwardPass.forward(tokens[0], session.forwardPass.checkpoint());
    return new LogitBatch(1, weights.header().vocabularySize(), logits);
  }

  @Override
  public void rewind(Session session, int checkpoint) {
    requireSession(session).forwardPass.rewind(checkpoint);
  }

  @Override
  public void reset(Session session) {
    requireSession(session).forwardPass.reset();
  }

  @Override
  public int checkpoint() {
    return defaultSequence.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return verifyTransient(tokens, startPosition).snapshot();
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    validatePrefill(defaultSequence, tokens, startPosition);
    int vocabularySize = weights.header().vocabularySize();
    float[] rows = new float[Math.multiplyExact(tokens.length, vocabularySize)];
    for (int index = 0; index < tokens.length; index++) {
      float[] logits = defaultSequence.forward(tokens[index], startPosition + index);
      System.arraycopy(logits, 0, rows, index * vocabularySize, vocabularySize);
    }
    return new LogitBatch(tokens.length, vocabularySize, rows);
  }

  @Override
  public void rewind(int checkpoint) {
    defaultSequence.rewind(checkpoint);
  }

  @Override
  public void reset() {
    defaultSequence.reset();
  }

  @Override
  public void close() {}

  private static float[] prefill(Needle2ForwardPass forwardPass, int[] tokens, int startPosition) {
    validatePrefill(forwardPass, tokens, startPosition);
    float[] logits = null;
    for (int index = 0; index < tokens.length; index++) {
      logits = forwardPass.forward(tokens[index], startPosition + index);
    }
    return logits;
  }

  private static void validatePrefill(
      Needle2ForwardPass forwardPass, int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != forwardPass.checkpoint()) {
      throw new IllegalArgumentException(
          "position must be sequential: expected "
              + forwardPass.checkpoint()
              + ", got "
              + startPosition);
    }
  }

  private Needle2Session requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (!(session instanceof Needle2DecoderAdapter.Needle2Session needleSession)) {
      throw new IllegalArgumentException("session belongs to a different decoder");
    }
    return needleSession;
  }
}
