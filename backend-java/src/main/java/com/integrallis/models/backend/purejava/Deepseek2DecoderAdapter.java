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
import com.integrallis.models.backend.purejava.deepseek2.Deepseek2ForwardPass;
import java.util.Objects;

/**
 * Adapts the DeepSeek-V2 decoder to the backend contract.
 *
 * <p>Single-sequence: AltUp's router produces its mixing coefficients per token, so the state is
 * per decoder rather than per session and a second concurrent session is refused instead of
 * silently sharing one.
 *
 * <p><b>Rewinding truncates.</b> Unlike the recurrent and ring-buffered decoders here, this one's
 * key-value cache is a plain array indexed by absolute position and attention reads only positions
 * up to the current one. Moving the position back is therefore an exact rewind -- the entries above
 * it are simply overwritten on the way forward again -- so no token history and no replay is
 * needed. The saving is not small: replaying a long shared prefix is what makes prefix reuse
 * expensive elsewhere.
 */
final class Deepseek2DecoderAdapter implements PureJavaDecoder {

  private final class Deepseek2Session implements Session {
    @Override
    public int checkpoint() {
      return forwardPass.nextPosition();
    }
  }

  private final Deepseek2ForwardPass forwardPass;
  private final Deepseek2Session defaultSession;
  private boolean sessionHandedOut;

  Deepseek2DecoderAdapter(Deepseek2ForwardPass forwardPass) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
    this.defaultSession = new Deepseek2Session();
  }

  @Override
  public int maxBatchSize() {
    return 1;
  }

  @Override
  public float[] forward(int token, int position) {
    return forwardPass.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return forward(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    // Serial: AltUp's coefficients are per token, so a batched prefill would still advance one
    // token
    // at a time. Through forward(int, int) so these tokens are recorded for a later rewind.
    float[] logits = null;
    for (int index = 0; index < tokens.length; index++) {
      logits = forward(tokens[index], startPosition + index);
    }
    return logits;
  }

  @Override
  public Session openSession() {
    if (sessionHandedOut) {
      throw new UnsupportedOperationException(
          "DeepSeek-V2 holds one key-value cache per decoder, so it supports a single session");
    }
    sessionHandedOut = true;
    return defaultSession;
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    requireDefault(session);
    return forward(token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    requireDefault(session);
    return forward(token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    requireDefault(session);
    return prefill(tokens, startPosition);
  }

  @Override
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    if (sessions.length != 1 || tokens.length != 1) {
      throw new UnsupportedOperationException(
          "DeepSeek-V2 decodes one sequence at a time; batched sessions are not supported");
    }
    requireDefault(sessions[0]);
    float[] logits = forward(tokens[0], forwardPass.nextPosition());
    return new LogitBatch(1, logits.length, logits.clone());
  }

  @Override
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    return forwardBatch(sessions, tokens);
  }

  @Override
  public void rewind(Session session, int checkpoint) {
    requireDefault(session);
    int position = forwardPass.nextPosition();
    if (checkpoint == position) {
      return;
    }
    if (checkpoint < 0 || checkpoint > position) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + position + ": " + checkpoint);
    }
    forwardPass.rewindTo(checkpoint);
  }

  @Override
  public void reset(Session session) {
    requireDefault(session);
    reset();
  }

  @Override
  public int checkpoint() {
    return forwardPass.nextPosition();
  }

  /**
   * Refused: speculative verification is not implemented here.
   *
   * <p>Rewinding here is cheap, so this could be supported. It is refused for now because nothing
   * has measured it on this decoder, and a speculative path that has never been compared against
   * serial decoding is a way to produce different text rather than faster text.
   */
  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    throw new UnsupportedOperationException(
        "DeepSeek-V2 speculative verification is not implemented");
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    return verify(tokens, startPosition);
  }

  @Override
  public void reset() {
    forwardPass.reset();
  }

  @Override
  public void rewind(int checkpoint) {
    rewind(defaultSession, checkpoint);
  }

  @Override
  public void close() {
    // Nothing mapped and nothing pooled beyond the weights, which the arena owns.
  }

  private void requireDefault(Session session) {
    if (session != defaultSession) {
      throw new IllegalArgumentException("session was not opened by this DeepSeek-V2 decoder");
    }
  }
}
