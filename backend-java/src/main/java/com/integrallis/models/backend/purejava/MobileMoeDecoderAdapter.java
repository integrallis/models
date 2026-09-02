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
import com.integrallis.models.backend.purejava.mobilemoe.MobileMoeForwardPass;
import java.util.Objects;

/** Adapts MobileMoE sequence state to the architecture-neutral backend contract. */
final class MobileMoeDecoderAdapter implements PureJavaDecoder {

  private final class MobileMoeSession implements Session {
    private final MobileMoeForwardPass.Session delegate = forwardPass.openSession();

    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }
  }

  private final MobileMoeForwardPass forwardPass;
  private final MobileMoeSession defaultSession;

  MobileMoeDecoderAdapter(MobileMoeForwardPass forwardPass) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
    defaultSession = new MobileMoeSession();
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
    return forwardPass.forwardTransient(defaultSession.delegate, token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return prefill(defaultSession.delegate, tokens, startPosition);
  }

  @Override
  public Session openSession() {
    return new MobileMoeSession();
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    return forwardPass.forward(requireSession(session).delegate, token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    return forwardPass.forwardTransient(requireSession(session).delegate, token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    return prefill(requireSession(session).delegate, tokens, startPosition);
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
      throw new IllegalArgumentException("MobileMoE currently supports one session per batch");
    }
    MobileMoeSession session = requireSession(sessions[0]);
    float[] logits =
        forwardPass.forwardTransient(session.delegate, tokens[0], session.delegate.checkpoint());
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
    validate(defaultSession.delegate, tokens, startPosition);
    int vocabularySize = forwardPass.config().vocabSize();
    float[] rows = new float[Math.multiplyExact(tokens.length, vocabularySize)];
    for (int index = 0; index < tokens.length; index++) {
      float[] logits =
          forwardPass.forwardTransient(
              defaultSession.delegate, tokens[index], startPosition + index);
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

  private float[] prefill(MobileMoeForwardPass.Session session, int[] tokens, int startPosition) {
    validate(session, tokens, startPosition);
    return forwardPass.prefill(session, tokens, startPosition);
  }

  private static void validate(
      MobileMoeForwardPass.Session session, int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != session.checkpoint()) {
      throw new IllegalArgumentException(
          "position must be sequential: expected "
              + session.checkpoint()
              + ", got "
              + startPosition);
    }
  }

  private MobileMoeSession requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (!(session instanceof MobileMoeDecoderAdapter.MobileMoeSession mobileMoeSession)) {
      throw new IllegalArgumentException("session belongs to a different decoder");
    }
    return mobileMoeSession;
  }
}
