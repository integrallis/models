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
import com.integrallis.models.backend.purejava.gemma4.Gemma4Decoder;
import java.io.IOException;
import java.util.Objects;

/** Adapts the Gemma 4 graph to the backend's architecture-neutral decoder contract. */
final class Gemma4DecoderAdapter implements PureJavaDecoder {

  private final class Gemma4Session implements Session {
    private final Gemma4Decoder.Session delegate;

    private Gemma4Session(Gemma4Decoder.Session delegate) {
      this.delegate = delegate;
    }

    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }
  }

  private final Gemma4Decoder decoder;
  private Gemma4Decoder.Session[] sessionBatch = new Gemma4Decoder.Session[0];

  Gemma4DecoderAdapter(Gemma4Decoder decoder) {
    this.decoder = Objects.requireNonNull(decoder, "decoder");
  }

  @Override
  public int maxBatchSize() {
    return decoder.maxBatchSize();
  }

  int prefillBatchSize() {
    return decoder.prefillBatchSize();
  }

  @Override
  public float[] forward(int token, int position) {
    return decoder.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return decoder.forwardTransient(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return decoder.prefill(tokens, startPosition);
  }

  @Override
  public float[] prefillHiddenState(int[] tokens, int startPosition) {
    return decoder.prefillHiddenState(tokens, startPosition);
  }

  @Override
  public float[] hiddenState(int token, int position) {
    return decoder.hiddenState(token, position);
  }

  @Override
  public boolean supportsHiddenState() {
    return true;
  }

  @Override
  public Session openSession() {
    return new Gemma4Session(decoder.openSession());
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    return decoder.forward(requireSession(session), token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    return decoder.forwardTransient(requireSession(session), token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    return decoder.prefill(requireSession(session), tokens, startPosition);
  }

  @Override
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    return decoder.forwardBatch(unwrapSessions(sessions), tokens);
  }

  @Override
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    return decoder.forwardBatchTransient(unwrapSessions(sessions), tokens);
  }

  @Override
  public void rewind(Session session, int checkpoint) {
    decoder.rewind(requireSession(session), checkpoint);
  }

  @Override
  public void reset(Session session) {
    decoder.reset(requireSession(session));
  }

  @Override
  public int checkpoint() {
    return decoder.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return decoder.verify(tokens, startPosition);
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    return decoder.verifyTransient(tokens, startPosition);
  }

  @Override
  public void rewind(int checkpoint) {
    decoder.rewind(checkpoint);
  }

  @Override
  public void reset() {
    decoder.reset();
  }

  @Override
  public void close() throws IOException {
    decoder.close();
  }

  private Gemma4Decoder.Session[] unwrapSessions(Session[] sessions) {
    Objects.requireNonNull(sessions, "sessions");
    if (sessionBatch.length != sessions.length) {
      sessionBatch = new Gemma4Decoder.Session[sessions.length];
    }
    for (int index = 0; index < sessions.length; index++) {
      sessionBatch[index] = requireSession(sessions[index]);
    }
    return sessionBatch;
  }

  private Gemma4Decoder.Session requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (!(session instanceof Gemma4DecoderAdapter.Gemma4Session gemma4Session)) {
      throw new IllegalArgumentException("session belongs to a different decoder");
    }
    return gemma4Session.delegate;
  }
}
