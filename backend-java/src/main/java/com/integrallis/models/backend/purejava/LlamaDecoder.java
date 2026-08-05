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
import com.integrallis.models.backend.purejava.llama.LlamaForwardPass;
import java.util.Objects;

/** Adapts the Llama-family graph to the backend's architecture-neutral decoder contract. */
final class LlamaDecoder implements PureJavaDecoder {

  private final class LlamaSession implements Session {
    private final LlamaForwardPass.Session delegate;

    private LlamaSession(LlamaForwardPass.Session delegate) {
      this.delegate = delegate;
    }

    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }
  }

  private final LlamaForwardPass forwardPass;
  private LlamaForwardPass.Session[] sessionBatch = new LlamaForwardPass.Session[0];

  LlamaDecoder(LlamaForwardPass forwardPass) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
  }

  @Override
  public int maxBatchSize() {
    return forwardPass.maxSessionBatchSize();
  }

  @Override
  public float[] forward(int token, int position) {
    return forwardPass.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return forwardPass.forwardTransient(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return forwardPass.prefill(tokens, startPosition);
  }

  @Override
  public float[] prefillHiddenState(int[] tokens, int startPosition) {
    return forwardPass.prefillHiddenState(tokens, startPosition);
  }

  @Override
  public float[] hiddenState(int token, int position) {
    return forwardPass.hiddenState(token, position);
  }

  @Override
  public boolean supportsHiddenState() {
    return true;
  }

  @Override
  public Session openSession() {
    return new LlamaSession(forwardPass.openSession());
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    return forwardPass.forward(requireSession(session), token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    return forwardPass.forwardTransient(requireSession(session), token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    return forwardPass.prefill(requireSession(session), tokens, startPosition);
  }

  @Override
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    return forwardPass.forwardBatch(unwrapSessions(sessions), tokens);
  }

  @Override
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    return forwardPass.forwardBatchTransient(unwrapSessions(sessions), tokens);
  }

  @Override
  public void rewind(Session session, int checkpoint) {
    forwardPass.rewind(requireSession(session), checkpoint);
  }

  @Override
  public void reset(Session session) {
    forwardPass.reset(requireSession(session));
  }

  @Override
  public int checkpoint() {
    return forwardPass.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return forwardPass.verify(tokens, startPosition);
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    return forwardPass.verifyTransient(tokens, startPosition);
  }

  @Override
  public void rewind(int checkpoint) {
    forwardPass.rewind(checkpoint);
  }

  @Override
  public void reset() {
    forwardPass.reset();
  }

  @Override
  public void close() {}

  private LlamaForwardPass.Session[] unwrapSessions(Session[] sessions) {
    Objects.requireNonNull(sessions, "sessions");
    if (sessionBatch.length != sessions.length) {
      sessionBatch = new LlamaForwardPass.Session[sessions.length];
    }
    for (int index = 0; index < sessions.length; index++) {
      sessionBatch[index] = requireSession(sessions[index]);
    }
    return sessionBatch;
  }

  private LlamaForwardPass.Session requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (!(session instanceof LlamaDecoder.LlamaSession llamaSession)) {
      throw new IllegalArgumentException("session belongs to a different decoder");
    }
    return llamaSession.delegate;
  }
}
