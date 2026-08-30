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
import com.integrallis.models.backend.purejava.qwen35.Qwen35ForwardPass;
import java.util.Objects;

/** Adapts Qwen3.5 hybrid recurrent/attention state to the backend contract. */
final class Qwen35DecoderAdapter implements PureJavaDecoder {

  private final class Qwen35Session implements Session {
    private final Qwen35ForwardPass.Session delegate = forwardPass.openSession(capacity);

    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }
  }

  private final Qwen35ForwardPass forwardPass;
  private final int capacity;
  private final Qwen35Session defaultSession;

  Qwen35DecoderAdapter(Qwen35ForwardPass forwardPass, int capacity) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
    this.capacity = capacity;
    this.defaultSession = new Qwen35Session();
  }

  @Override
  public int maxBatchSize() {
    return 1;
  }

  @Override
  public float[] forward(int token, int position) {
    return forwardPass.forward(defaultSession.delegate, token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return forward(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return forwardPass.prefill(defaultSession.delegate, tokens, startPosition);
  }

  @Override
  public Session openSession() {
    return new Qwen35Session();
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    return forwardPass.forward(requireSession(session).delegate, token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    return forward(session, token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    return forwardPass.prefill(requireSession(session).delegate, tokens, startPosition);
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
          "Qwen3.5 compatibility execution currently supports one session per batch");
    }
    Qwen35Session session = requireSession(sessions[0]);
    float[] logits =
        forwardPass.forward(session.delegate, tokens[0], session.delegate.checkpoint());
    return new LogitBatch(1, forwardPass.config().vocabSize(), logits);
  }

  @Override
  public void rewind(Session session, int checkpoint) {
    forwardPass.rewind(requireSession(session).delegate, checkpoint);
  }

  @Override
  public void reset(Session session) {
    forwardPass.reset(requireSession(session).delegate);
  }

  @Override
  public int checkpoint() {
    return defaultSession.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return verifyTransient(tokens, startPosition).snapshot();
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != checkpoint()) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + checkpoint() + ", got " + startPosition);
    }
    int vocabularySize = forwardPass.config().vocabSize();
    float[] rows = new float[Math.multiplyExact(tokens.length, vocabularySize)];
    for (int index = 0; index < tokens.length; index++) {
      float[] logits = forward(tokens[index], startPosition + index);
      System.arraycopy(logits, 0, rows, index * vocabularySize, vocabularySize);
    }
    return new LogitBatch(tokens.length, vocabularySize, rows);
  }

  @Override
  public void rewind(int checkpoint) {
    forwardPass.rewind(defaultSession.delegate, checkpoint);
  }

  @Override
  public void reset() {
    forwardPass.reset(defaultSession.delegate);
  }

  @Override
  public void close() {}

  private Qwen35Session requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (!(session instanceof Qwen35DecoderAdapter.Qwen35Session qwen35Session)) {
      throw new IllegalArgumentException("session belongs to a different decoder");
    }
    return qwen35Session;
  }
}
